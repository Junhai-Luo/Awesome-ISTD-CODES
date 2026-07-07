import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def canonical_networks(presets):
    alias_values = set(presets.get("network_alias", {}).values())
    names = []
    for name in presets.get("network_pretrained_map", {}).keys():
        if name in alias_values or name in ("acm_fpn", "acm_unet"):
            names.append(name)
    return names


def resolve_dataset_override(config, key, dataset_name, default_value):
    overrides = config.get(f"{key}_by_dataset", {})
    if isinstance(overrides, dict):
        for name in (dataset_name, str(dataset_name).upper(), str(dataset_name).lower()):
            if name in overrides and str(overrides[name]).strip() != "":
                return overrides[name]
    return config.get(key, default_value)


def build_profile_model(network_name, presets, num_classes=1):
    name = str(network_name).lower()
    network_presets = presets.get("network_presets", {})

    if name == "acm_fpn":
        from nets.factory import build_acm_detector

        preset = network_presets.get("acm_fpn", {})
        return build_acm_detector(
            backbone_mode=preset.get("backbone_mode", "FPN"),
            fuse_mode=preset.get("fuse_mode", "AsymBi"),
            blocks_per_layer=preset.get("blocks_per_layer", 4),
            num_classes=num_classes,
        )

    if name in ("acm_unet", "acm_unet_saliency"):
        from nets.factory import build_acm_detector

        preset = network_presets.get(name, network_presets.get("acm_unet", {}))
        return build_acm_detector(
            backbone_mode=preset.get("backbone_mode", "UNet"),
            fuse_mode=preset.get("fuse_mode", "AsymBi"),
            blocks_per_layer=preset.get("blocks_per_layer", 4),
            num_classes=num_classes,
            det_mode=preset.get("det_mode", "feature"),
        )

    if name in ("alcnet", "alcnet_det", "alc"):
        from nets.factory import build_alc_detector

        preset = network_presets.get("alcnet", {})
        return build_alc_detector(
            fuse_mode=preset.get("fuse_mode", "AsymBi"),
            blocks_per_layer=preset.get("blocks_per_layer", 4),
            num_classes=num_classes,
        )

    if name in ("dnanet", "dnanet_det", "dna"):
        from nets.dna.model_DNANet_det import DNANetDet

        preset = network_presets.get("dnanet", {})
        return DNANetDet(
            input_channels=3,
            num_classes=num_classes,
            channel_size=preset.get("channel_size", "three"),
            backbone=preset.get("backbone", "resnet_18"),
        )

    if name in ("uiunet", "uiu"):
        from nets.uiu.detection import UIUNETDet

        preset = network_presets.get("uiunet", {})
        return UIUNETDet(
            in_ch=3,
            num_classes=num_classes,
            fuse_mode=preset.get("fuse_mode", "AsymBi"),
        )

    from nets.factory import build_network

    return build_network(name, num_classes=num_classes, num_frame=5)


def resolve_input_size(network_name, dataset_name, predict_config, presets, override_size=None):
    if override_size is not None:
        return int(override_size)

    predict_cfg = predict_config.get("predict", {})
    network_presets = presets.get("network_presets", {})
    name = str(network_name).lower()

    if name in ("dnanet", "dnanet_det", "dna"):
        return int(
            resolve_dataset_override(
                predict_cfg,
                "dna_input_size",
                dataset_name,
                network_presets.get("dnanet", {}).get("base_size", 256),
            )
        )
    if name in ("uiunet", "uiu"):
        return int(predict_cfg.get("input_size", network_presets.get("uiunet", {}).get("base_size", 512)))
    if name in ("alcnet", "alcnet_det", "alc"):
        return int(predict_cfg.get("input_size", network_presets.get("alcnet", {}).get("base_size", 512)))
    return int(predict_cfg.get("input_size", 512))


def make_dummy_input(network_name, batch_size, input_size, num_frame, device):
    name = str(network_name).lower()
    if name in ("sstnet", "tridos", "slowfastnet", "slowfastnet_9520", "dqaligner", "dqaligner_saliency"):
        shape = (batch_size, 3, int(num_frame), input_size, input_size)
    else:
        shape = (batch_size, 3, input_size, input_size)
    try:
        import torch
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required for profiling. Please run this script in the same environment used for training. "
            f"Original error: {exc}"
        )

    return torch.randn(*shape, device=device), shape


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_macs(model, dummy_input):
    try:
        from thop import profile
    except Exception as exc:
        return math.nan, "thop unavailable: %s" % exc

    try:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(macs), ""
    except Exception as exc:
        return math.nan, "thop failed: %s" % exc


def benchmark_fps(model, dummy_input, device, warmup=10, repeat=50):
    import torch

    if repeat <= 0:
        return math.nan, math.nan, ""

    try:
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            for _ in range(repeat):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        latency_ms = elapsed * 1000.0 / repeat
        fps = repeat / elapsed if elapsed > 0 else math.nan
        return fps, latency_ms, ""
    except Exception as exc:
        return math.nan, math.nan, "fps failed: %s" % exc


def _format_float(value, digits=6):
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "network",
        "input_shape",
        "device",
        "params_m",
        "trainable_params_m",
        "flops_g",
        "macs_g",
        "flops_g_2x_macs",
        "fps",
        "latency_ms",
        "status",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("params_m", "trainable_params_m", "flops_g", "macs_g", "flops_g_2x_macs", "fps", "latency_ms"):
                out[key] = _format_float(out.get(key))
            writer.writerow(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile DETLAB networks: Params, FLOPs/MACs, and model FPS.")
    parser.add_argument("--presets", default="configs/experiment_presets.json")
    parser.add_argument("--predict-config", default="configs/predict_experiment_config.json")
    parser.add_argument("--output", default="result/profile/profile_networks.csv")
    parser.add_argument("--networks", nargs="+", default=["all"], help="Networks to profile, or all.")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Datasets to profile, or all.")
    parser.add_argument("--input-size", type=int, default=None, help="Override square input size.")
    parser.add_argument("--num-frame", type=int, default=5, help="Temporal window for SST/Tridos.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-flops", action="store_true", help="Skip FLOPs/MACs profiling.")
    parser.add_argument("--no-fps", action="store_true", help="Skip FPS benchmark.")
    return parser.parse_args()


def main():
    args = parse_args()
    presets = load_json(args.presets)
    predict_config = load_json(args.predict_config)

    dataset_names = list(presets.get("dataset_presets", {}).keys()) if args.datasets == ["all"] else args.datasets
    network_names = canonical_networks(presets) if args.networks == ["all"] else args.networks

    try:
        import torch
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required for profiling. Please run this script in the same environment used for training. "
            f"Original error: {exc}"
        )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    rows = []

    for dataset_name in dataset_names:
        for network_name in network_names:
            note_parts = []
            status = "ok"
            try:
                model = build_profile_model(network_name, presets, num_classes=args.num_classes)
                model.eval().to(device)

                input_size = resolve_input_size(
                    network_name,
                    dataset_name,
                    predict_config,
                    presets,
                    override_size=args.input_size,
                )
                dummy_input, input_shape = make_dummy_input(
                    network_name,
                    args.batch_size,
                    input_size,
                    args.num_frame,
                    device,
                )

                params, trainable_params = count_params(model)

                macs = math.nan
                if not args.no_flops:
                    macs, note = profile_macs(model, dummy_input)
                    if note:
                        note_parts.append(note)

                fps = math.nan
                latency_ms = math.nan
                if not args.no_fps:
                    fps, latency_ms, note = benchmark_fps(
                        model,
                        dummy_input,
                        device,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    if note:
                        note_parts.append(note)

                rows.append(
                    {
                        "dataset": dataset_name,
                        "network": network_name,
                        "input_shape": "x".join(str(v) for v in input_shape),
                        "device": str(device),
                        "params_m": params / 1e6,
                        "trainable_params_m": trainable_params / 1e6,
                        "flops_g": macs / 1e9 if not math.isnan(macs) else math.nan,
                        "macs_g": macs / 1e9 if not math.isnan(macs) else math.nan,
                        "flops_g_2x_macs": (2.0 * macs) / 1e9 if not math.isnan(macs) else math.nan,
                        "fps": fps,
                        "latency_ms": latency_ms,
                        "status": status,
                        "note": " | ".join(note_parts),
                    }
                )
                print(
                    "%s/%s params=%.3fM flops=%sG fps=%s"
                    % (
                        dataset_name,
                        network_name,
                        params / 1e6,
                        _format_float(macs / 1e9 if not math.isnan(macs) else math.nan, 3),
                        _format_float(fps, 3),
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "network": network_name,
                        "input_shape": "",
                        "device": str(device),
                        "params_m": math.nan,
                        "trainable_params_m": math.nan,
                        "flops_g": math.nan,
                        "macs_g": math.nan,
                        "flops_g_2x_macs": math.nan,
                        "fps": math.nan,
                        "latency_ms": math.nan,
                        "status": "failed",
                        "note": str(exc),
                    }
                )
                print("[Profile Skip] %s/%s: %s" % (dataset_name, network_name, exc))

    out_path = Path(args.output)
    write_rows(out_path, rows)
    print("Saved profile CSV to:", out_path)
    print("Note: flops_g follows profile/tools/profile_sota_models.py and records thop.profile output directly.")
    print("Note: fps follows profile/tools/profile_sota_models.py and measures batch forward passes per second.")


if __name__ == "__main__":
    main()
