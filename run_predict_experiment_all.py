import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime


DEFAULT_CONFIG_PATH = "configs/predict_experiment_config.json"
DEFAULT_PRESETS_PATH = "configs/experiment_presets.json"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


DEFAULT_WEIGHT_NAMES = {
    "loss": "best_epoch_weights.pth",
    "last": "last_epoch_weights.pth",
}


METRIC_WEIGHT_POLICIES = {"ap50", "ap50:95"}

WEIGHT_POLICY_ALIASES = {
    "ap50": "ap50",
    "map50": "ap50",
    "best-ap50": "ap50",
    "ap50:95": "ap50:95",
    "map50:95": "ap50:95",
    "ap5095": "ap50:95",
    "map5095": "ap50:95",
    "best-map": "ap50:95",
    "best": "ap50:95",
    "loss": "loss",
    "val-loss": "loss",
    "val_loss": "loss",
    "last": "last",
    "latest-epoch": "last",
}


def _path_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1


def _slug(value):
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _format_tag_float(value):
    if value is None:
        return "na"
    return _slug(f"{float(value):g}")


def make_auto_run_tag(weight_policy, confidence, nms_iou):
    policy = normalize_weight_policy(weight_policy)
    return "wp-%s_conf%s_nms%s" % (
        _slug(policy),
        _format_tag_float(confidence),
        _format_tag_float(nms_iou),
    )


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_command(cmd, dry_run=False, env=None):
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def merge_non_empty(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        out[k] = v
    return out


def resolve_dataset_path(path_value, root_prefix):
    path_value = str(path_value or "").strip()
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value.replace("\\", "/")
    if root_prefix:
        root_parts = [p for p in root_prefix.replace("\\", "/").split("/") if p]
        path_parts = [p for p in path_value.replace("\\", "/").split("/") if p]
        max_overlap = min(len(root_parts), len(path_parts))
        for overlap in range(max_overlap, 0, -1):
            if root_parts[-overlap:] == path_parts[:overlap]:
                tail = path_parts[overlap:]
                joined = os.path.join(root_prefix, *tail) if tail else root_prefix
                return joined.replace("\\", "/")
        return os.path.join(root_prefix, path_value).replace("\\", "/")
    return path_value.replace("\\", "/")


def resolve_pretrained(network_name, predict_model_path, train_model_path, network_alias, network_pretrained_map):
    net_key = network_alias.get(str(network_name).lower(), str(network_name).lower())
    default_path = network_pretrained_map.get(net_key, "")
    if predict_model_path:
        return predict_model_path, default_path, net_key
    if train_model_path:
        return train_model_path, default_path, net_key
    return default_path, default_path, net_key


def is_acm_network(net_key):
    return str(net_key).lower() in ("acm_fpn", "acm_unet", "acm_unet_saliency")


def is_dna_network(net_key):
    return str(net_key).lower() in ("dnanet", "dnanet_saliency")


def is_dna_saliency_network(net_key):
    return str(net_key).lower() == "dnanet_saliency"


def is_alc_network(net_key):
    return str(net_key).lower() in ("alcnet", "alcnet_saliency")


def is_alc_saliency_network(net_key):
    return str(net_key).lower() == "alcnet_saliency"


def is_uiu_network(net_key):
    return str(net_key).lower() in ("uiunet", "uiunet_saliency")


def is_uiu_saliency_network(net_key):
    return str(net_key).lower() == "uiunet_saliency"


def is_sctrans_network(net_key):
    return str(net_key).lower() in ("sctransnet", "sctransnet_det")


def resolve_dataset_override(config, key, dataset_name, default_value):
    overrides = config.get(f"{key}_by_dataset", {})
    if isinstance(overrides, dict):
        for name in (dataset_name, str(dataset_name).upper(), str(dataset_name).lower()):
            if name in overrides and str(overrides[name]).strip() != "":
                return overrides[name]
    return config.get(key, default_value)


def _make_output_dir(predict_cfg, dataset_name, network_name):
    timestamp = "%s_pid%d" % (datetime.now().strftime("%Y%m%d_%H%M%S_%f"), os.getpid())
    output_parts = [
        predict_cfg["output_dir"] if predict_cfg.get("output_dir") else predict_cfg.get("predict_output_root", "result/predict"),
        dataset_name,
        network_name,
    ]
    run_tag = _slug(predict_cfg.get("run_tag", ""))
    if run_tag:
        output_parts.append(run_tag)
    output_parts.append(timestamp)
    return os.path.join(*output_parts)


def _append_predict_flags(cmd, predict_cfg):
    if predict_cfg.get("auto_input_from_json", False):
        cmd.append("--auto_input_from_json")
    if predict_cfg.get("save_eval_json", False):
        cmd.append("--save_eval_json")
    if predict_cfg.get("run_eval", False):
        cmd.append("--run_eval")
    if predict_cfg.get("save_failed_images", False):
        cmd.append("--save_failed_images")
        cmd.extend(["--failed_iou", str(predict_cfg.get("failed_iou", 0.5))])
    if predict_cfg.get("cpu", False):
        cmd.append("--cpu")


def _base_predict_cmd(predict_cfg, model_path, output_dir, input_size, network_name, confidence, nms_iou):
    cmd = [
        sys.executable,
        "infer/predict_from_coco_json.py",
        "--json_path",
        predict_cfg["json_path"],
        "--dataset_img_path",
        predict_cfg["dataset_img_path"],
        "--model_path",
        model_path,
        "--classes_path",
        predict_cfg.get("classes_path", "model_data/classes.txt"),
        "--output_dir",
        output_dir,
        "--input_size",
        str(input_size),
        "--network_name",
        network_name,
        "--confidence",
        str(confidence),
        "--nms_iou",
        str(nms_iou),
        "--output_mode",
        str(predict_cfg.get("output_mode", "all")).lower(),
    ]
    if predict_cfg.get("vis_confidence") is not None:
        cmd.extend(["--vis_confidence", str(predict_cfg.get("vis_confidence"))])
    if predict_cfg.get("vis_max_boxes") is not None:
        cmd.extend(["--vis_max_boxes", str(predict_cfg.get("vis_max_boxes"))])
    return cmd


def run_predict(config, presets, dry_run=False):
    dataset_presets = presets["dataset_presets"]
    network_alias = presets.get("network_alias", {})
    network_pretrained_map = presets.get("network_pretrained_map", {})
    network_presets = presets.get("network_presets", {})
    global_cfg = presets.get("global", {})

    dataset_name = config["dataset"]
    preset = dataset_presets.get(dataset_name)
    if preset is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(dataset_presets.keys())}")

    predict_cfg = merge_non_empty(preset.get("default_predict", {}), config.get("predict", {}))
    dataset_root_prefix = (
        str(config.get("dataset_root_prefix", "")).strip()
        or str(global_cfg.get("dataset_root_prefix", "")).strip()
    )
    predict_cfg["dataset_img_path"] = resolve_dataset_path(
        predict_cfg.get("dataset_img_path", ""), dataset_root_prefix
    )

    model_path, bound_default, net_key = resolve_pretrained(
        config.get("network", "sstnet"),
        predict_cfg.get("model_path", ""),
        config.get("train_model_path", ""),
        network_alias,
        network_pretrained_map,
    )
    if net_key not in network_pretrained_map:
        raise ValueError(
            "Unknown network '%s'. Use one of: %s"
            % (config.get("network", "sstnet"), ", ".join(sorted(network_pretrained_map.keys())))
        )
    predict_cfg["model_path"] = model_path

    missing = [k for k in ("json_path", "dataset_img_path", "model_path") if not predict_cfg.get(k)]
    if missing:
        raise ValueError(f"Missing predict fields in config json: {missing}")

    network_name = config.get("network", "sstnet")
    output_dir = _make_output_dir(predict_cfg, dataset_name, network_name)
    print(f"Network: {net_key} | Predict model: {model_path or '(none)'}")
    if bound_default and model_path != bound_default:
        print(f"Note: overriding bound default ({bound_default}) with {model_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT if not env.get("PYTHONPATH") else f"{PROJECT_ROOT}{os.pathsep}{env['PYTHONPATH']}"

    confidence = predict_cfg.get("confidence", 0.001)
    nms_iou = predict_cfg.get("nms_iou", 0.65)

    if is_uiu_network(net_key):
        preset_cfg = network_presets.get(net_key, network_presets.get("uiunet", {}))
        uiu_det_mode = str(predict_cfg.get("uiu_det_mode", preset_cfg.get("det_mode", "")) or "")
        if not uiu_det_mode:
            uiu_det_mode = "saliency" if is_uiu_saliency_network(net_key) else "feature"
        cmd = _base_predict_cmd(
            predict_cfg,
            model_path,
            output_dir,
            predict_cfg.get("input_size", preset_cfg.get("base_size", 512)),
            network_name,
            predict_cfg.get("confidence", preset_cfg.get("eval_confidence", 0.3)),
            predict_cfg.get("nms_iou", preset_cfg.get("eval_nms_iou", 0.45)),
        )
        cmd.extend([
            "--num_classes",
            str(predict_cfg.get("num_classes", preset_cfg.get("num_classes", 1))),
            "--uiu_fuse_mode",
            str(predict_cfg.get("uiu_fuse_mode", preset_cfg.get("fuse_mode", "AsymBi"))),
            "--uiu_det_mode",
            uiu_det_mode,
        ])
    elif is_dna_network(net_key):
        preset_cfg = network_presets.get(net_key, network_presets.get("dnanet", {}))
        input_size = resolve_dataset_override(predict_cfg, "dna_input_size", dataset_name, preset_cfg.get("base_size", 512))
        dna_det_mode = str(predict_cfg.get("dna_det_mode", preset_cfg.get("det_mode", "")) or "")
        if not dna_det_mode:
            dna_det_mode = "saliency" if is_dna_saliency_network(net_key) else "feature"
        cmd = _base_predict_cmd(
            predict_cfg,
            model_path,
            output_dir,
            input_size,
            network_name,
            predict_cfg.get("confidence", preset_cfg.get("eval_confidence", 0.3)),
            predict_cfg.get("nms_iou", preset_cfg.get("eval_nms_iou", 0.45)),
        )
        cmd.extend([
            "--num_classes",
            str(predict_cfg.get("num_classes", preset_cfg.get("num_classes", 1))),
            "--dna_channel_size",
            str(predict_cfg.get("dna_channel_size", preset_cfg.get("channel_size", "three"))),
            "--dna_backbone",
            str(predict_cfg.get("dna_backbone", preset_cfg.get("backbone", "resnet_18"))),
            "--dna_det_mode",
            dna_det_mode,
        ])
    elif is_alc_network(net_key):
        preset_cfg = network_presets.get(net_key, network_presets.get("alcnet", {}))
        alc_det_mode = str(predict_cfg.get("alc_det_mode", preset_cfg.get("det_mode", "")) or "")
        if not alc_det_mode:
            alc_det_mode = "saliency" if is_alc_saliency_network(net_key) else "feature"
        cmd = _base_predict_cmd(
            predict_cfg,
            model_path,
            output_dir,
            predict_cfg.get("input_size", preset_cfg.get("base_size", 512)),
            network_name,
            predict_cfg.get("confidence", preset_cfg.get("eval_confidence", 0.3)),
            predict_cfg.get("nms_iou", preset_cfg.get("eval_nms_iou", 0.45)),
        )
        cmd.extend([
            "--fuse_mode",
            str(preset_cfg.get("fuse_mode", "AsymBi")),
            "--blocks_per_layer",
            str(int(preset_cfg.get("blocks_per_layer", 4))),
            "--num_classes",
            str(predict_cfg.get("num_classes", preset_cfg.get("num_classes", 1))),
            "--alc_det_mode",
            alc_det_mode,
        ])
    elif is_acm_network(net_key):
        preset_cfg = network_presets.get(net_key, {})
        acm_name = str(net_key).lower()
        default_acm_det_mode = "saliency" if "saliency" in acm_name else "feature"
        default_acm_backbone = "UNet" if ("unet" in acm_name or "saliency" in acm_name) else "FPN"
        acm_det_mode = str(predict_cfg.get("acm_det_mode", preset_cfg.get("det_mode", default_acm_det_mode)) or default_acm_det_mode)
        cmd = _base_predict_cmd(
            predict_cfg,
            model_path,
            output_dir,
            predict_cfg.get("input_size", preset_cfg.get("base_size", 512)),
            network_name,
            confidence,
            nms_iou,
        )
        cmd.extend([
            "--backbone_mode",
            str(predict_cfg.get("acm_backbone_mode", preset_cfg.get("backbone_mode", default_acm_backbone))),
            "--acm_det_mode",
            acm_det_mode,
            "--fuse_mode",
            str(predict_cfg.get("acm_fuse_mode", preset_cfg.get("fuse_mode", "AsymBi"))),
            "--blocks_per_layer",
            str(int(predict_cfg.get("acm_blocks_per_layer", preset_cfg.get("blocks_per_layer", 4)))),
            "--num_classes",
            str(predict_cfg.get("num_classes", preset_cfg.get("num_classes", 1))),
        ])
    elif is_sctrans_network(net_key):
        preset_cfg = network_presets.get(net_key, network_presets.get("sctransnet", {}))
        cmd = _base_predict_cmd(
            predict_cfg,
            model_path,
            output_dir,
            predict_cfg.get("input_size", preset_cfg.get("base_size", 512)),
            network_name,
            predict_cfg.get("confidence", preset_cfg.get("eval_confidence", 0.3)),
            predict_cfg.get("nms_iou", preset_cfg.get("eval_nms_iou", 0.45)),
        )
        cmd.extend([
            "--num_classes",
            str(predict_cfg.get("num_classes", preset_cfg.get("num_classes", 1))),
        ])
    else:
        cmd = _base_predict_cmd(
            predict_cfg,
            model_path,
            output_dir,
            predict_cfg.get("input_size", 512),
            network_name,
            confidence,
            nms_iou,
        )
        cmd.extend([
            "--num_frame",
            str(predict_cfg.get("num_frame", 5)),
            "--batch_size",
            str(predict_cfg.get("batch_size", 1)),
        ])
        if bool(predict_cfg.get("letterbox", False)):
            cmd.append("--letterbox")

    _append_predict_flags(cmd, predict_cfg)
    print("Predict outputs dir:", output_dir)
    run_command(cmd, dry_run=dry_run, env=env)


def _list_loss_dirs(network_log_dir):
    if not os.path.isdir(network_log_dir):
        return []
    dirs = {}
    network_log_dir_abs = os.path.abspath(network_log_dir)
    for root, dirnames, _ in os.walk(network_log_dir, followlinks=True):
        for name in dirnames:
            if not (name.startswith("loss_") or "loss" in name.lower()):
                continue
            path = os.path.join(root, name)
            if os.path.isdir(path) and os.path.abspath(path) != network_log_dir_abs:
                dirs[path] = _path_mtime(path)
    return sorted(dirs.keys(), key=lambda p: (dirs[p], os.path.basename(p)), reverse=True)


def _list_scanned_children(network_log_dir, max_items=30):
    if not os.path.isdir(network_log_dir):
        return ["<missing dir: %s>" % network_log_dir]
    items = []
    for root, dirnames, filenames in os.walk(network_log_dir, followlinks=True):
        rel = os.path.relpath(root, network_log_dir)
        rel = "." if rel == "." else rel
        sample_files = [name for name in filenames if name.endswith(".pth") or name.endswith(".json")][:5]
        items.append("%s dirs=%s files=%s" % (rel, dirnames[:5], sample_files))
        if len(items) >= max_items:
            break
    return items


def _latest_epoch_weight(run_dir):
    if not os.path.isdir(run_dir):
        return ""
    pattern = re.compile(r"^ep(\d+)-loss.*-val_loss.*\.pth$")
    candidates = []
    for name in os.listdir(run_dir):
        match = pattern.match(name)
        if not match:
            continue
        path = os.path.join(run_dir, name)
        candidates.append((int(match.group(1)), _path_mtime(path), path))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][2]


def _recursive_named_weights(root_dir, filename):
    if not os.path.isdir(root_dir):
        return []
    candidates = []
    for root, _, filenames in os.walk(root_dir):
        if filename not in filenames:
            continue
        path = os.path.join(root, filename)
        candidates.append((_path_mtime(path), path))
    return sorted(candidates, reverse=True)


def _recursive_latest_epoch_weight(root_dir):
    if not os.path.isdir(root_dir):
        return ""
    pattern = re.compile(r"^ep(\d+)-loss.*-val_loss.*\.pth$")
    candidates = []
    for root, _, filenames in os.walk(root_dir):
        for name in filenames:
            match = pattern.match(name)
            if not match:
                continue
            path = os.path.join(root, name)
            candidates.append((_path_mtime(path), int(match.group(1)), path))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][2]


def _epoch_weight(run_dir, epoch):
    pattern = re.compile(r"^ep%03d-loss.*-val_loss.*\.pth$" % int(epoch))
    candidates = []
    for name in os.listdir(run_dir):
        if pattern.match(name):
            path = os.path.join(run_dir, name)
            candidates.append((_path_mtime(path), path))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def _read_metric_value(path, keys):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    for key in keys:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _metric_keys_for_policy(policy):
    if policy == "ap50:95":
        return ("AP@[0.50:0.95]", "AP50:95", "mAP50:95", "mAP@[0.50:0.95]")
    if policy == "ap50":
        return ("AP@0.50", "AP50", "mAP@0.50", "mAP")
    raise ValueError("Unknown metric weight policy: %s" % policy)


def normalize_weight_policy(policy):
    normalized = WEIGHT_POLICY_ALIASES.get(str(policy).lower())
    if not normalized:
        raise ValueError("Unknown weight policy: %s" % policy)
    return normalized


def _best_metric_epoch_weight(run_dir, policy):
    if not os.path.isdir(run_dir):
        return ""
    metric_pattern = re.compile(r"^epoch_(\d+)_metrics\.json$")
    keys = _metric_keys_for_policy(policy)
    candidates = []

    for name in os.listdir(run_dir):
        match = metric_pattern.match(name)
        if not match:
            continue
        epoch = int(match.group(1))
        metrics_path = os.path.join(run_dir, name)
        value = _read_metric_value(metrics_path, keys)
        if value is None:
            continue
        weight_path = _epoch_weight(run_dir, epoch)
        if not weight_path:
            continue
        candidates.append((value, epoch, _path_mtime(metrics_path), weight_path))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def _recursive_best_metric_epoch_weight(root_dir, policy):
    if not os.path.isdir(root_dir):
        return ""
    metric_pattern = re.compile(r"^epoch_(\d+)_metrics\.json$")
    keys = _metric_keys_for_policy(policy)
    candidates = []

    for root, _, filenames in os.walk(root_dir):
        for name in filenames:
            match = metric_pattern.match(name)
            if not match:
                continue
            epoch = int(match.group(1))
            metrics_path = os.path.join(root, name)
            value = _read_metric_value(metrics_path, keys)
            if value is None:
                continue
            weight_path = _epoch_weight(root, epoch)
            if not weight_path:
                continue
            candidates.append((value, epoch, _path_mtime(metrics_path), weight_path))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def _resolve_weight_in_dir(weight_dir, policy):
    policy = normalize_weight_policy(policy)
    if policy == "last":
        return _latest_epoch_weight(weight_dir) or os.path.join(weight_dir, DEFAULT_WEIGHT_NAMES[policy])
    if policy in DEFAULT_WEIGHT_NAMES:
        return os.path.join(weight_dir, DEFAULT_WEIGHT_NAMES[policy])
    if policy in METRIC_WEIGHT_POLICIES:
        return _best_metric_epoch_weight(weight_dir, policy)
    raise ValueError("Unknown weight policy: %s" % policy)


def resolve_latest_weight(train_output_root, dataset_name, network_name, policy="best"):
    policy = normalize_weight_policy(policy)
    network_log_dir = os.path.join(train_output_root, dataset_name, network_name)
    checked = []

    run_dirs = _list_loss_dirs(network_log_dir)
    for run_dir in run_dirs:
        path = _resolve_weight_in_dir(run_dir, policy)
        if path:
            checked.append(path)
        elif policy in METRIC_WEIGHT_POLICIES:
            checked.append(
                os.path.join(run_dir, "epoch_XXX_metrics.json -> epXXX-loss*-val_loss*.pth")
            )
        if path and os.path.exists(path):
            return path

    checked_text = "\n  ".join(checked[:10]) if checked else "(none)"
    if not run_dirs:
        checked_text += "\n  Run dirs found: 0\n  Scanned children:\n  " + "\n  ".join(_list_scanned_children(network_log_dir))
    raise FileNotFoundError(
        "No %s weight found under %s/%s/%s. Checked:\n  %s"
        % (policy, train_output_root, dataset_name, network_name, checked_text)
    )


def canonical_networks(presets):
    return list(presets.get("network_pretrained_map", {}).keys())


def parse_args():
    parser = argparse.ArgumentParser("Predict runner with automatic latest trained weights")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to predict config json")
    parser.add_argument("--presets", default=DEFAULT_PRESETS_PATH, help="Path to presets json")
    parser.add_argument("--dataset", help="Dataset to run, e.g. DAUB/IRDST/ITSDT_15K")
    parser.add_argument("--network", help="Network to run, e.g. sstnet/tridos/dnanet")
    parser.add_argument("--model-path", help="Explicit model path. Overrides auto latest weight.")
    parser.add_argument(
        "--json-path",
        "--json_path",
        "--gt-json",
        "--gt_json",
        dest="json_path",
        help="Override COCO GT json path used for prediction image list and evaluation.",
    )
    parser.add_argument("--dataset-img-path", help="Override predict dataset image root path")
    parser.add_argument("--dataset-root-prefix", help="Override dataset root prefix")
    parser.add_argument("--train-output-root", default="logs", help="Root where training logs are saved")
    parser.add_argument(
        "--weight-policy",
        type=normalize_weight_policy,
        choices=["ap50", "ap50:95", "loss", "last"],
        default="ap50",
        help=(
            "Which weight to use from the latest loss_* directory. "
            "Use metric names: ap50 selects the highest AP@0.50 epoch, "
            "ap50:95 selects the highest AP@[0.50:0.95] epoch, "
            "loss selects best_epoch_weights.pth by lowest val loss, "
            "last selects the latest saved epoch. "
            "Compatible aliases: map50/best-ap50 -> ap50, map5095/best-map/best -> ap50:95, latest-epoch -> last."
        ),
    )
    parser.add_argument(
        "--all-latest",
        action="store_true",
        help="Run all dataset/network combinations that have a matching latest trained weight",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        help="Networks to use with --all-latest. Default: all canonical networks in presets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to use with --all-latest. Default: all dataset presets.",
    )
    parser.add_argument(
        "--output-mode",
        "--output_mode",
        dest="output_mode",
        choices=["all", "vis_only", "eval_only"],
        help="Override predict output mode",
    )
    parser.add_argument("--confidence", type=float, help="Override predict confidence threshold")
    parser.add_argument("--nms_iou", "--nms-iou", dest="nms_iou", type=float, help="Override predict NMS IoU threshold")
    parser.add_argument("--vis-confidence", "--vis_confidence", dest="vis_confidence", type=float, help="Override visualization confidence threshold")
    parser.add_argument("--vis-max-boxes", "--vis_max_boxes", dest="vis_max_boxes", type=int, help="Limit drawn boxes per image; 0 means no limit")
    parser.add_argument("--save-failed-images", "--save_failed_images", dest="save_failed_images", action="store_true", help="Save image paths whose GT objects are not matched by predictions during eval")
    parser.add_argument("--failed-iou", "--failed_iou", dest="failed_iou", type=float, help="IoU threshold used to decide failed prediction images")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, help="Batch size for video-network prediction.")
    parser.add_argument(
        "--run-tag",
        default="",
        help=(
            "Optional output grouping tag. Default is generated from weight policy, confidence and NMS, "
            "for example wp-ap50_conf0p001_nms0p65."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing weights in --all-latest mode")
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved commands")
    return parser.parse_args()


def run_one(args, presets, base_config, dataset_name, network_name, model_path, run_tag):
    config = dict(base_config)
    config["dataset"] = dataset_name
    config["network"] = network_name
    config.setdefault("predict", {})
    config["predict"] = dict(config["predict"])
    config["predict"]["model_path"] = model_path
    if run_tag:
        config["predict"]["run_tag"] = run_tag

    if args.output_mode:
        config["predict"]["output_mode"] = args.output_mode
    if args.confidence is not None:
        config["predict"]["confidence"] = args.confidence
    if args.nms_iou is not None:
        config["predict"]["nms_iou"] = args.nms_iou
    if args.vis_confidence is not None:
        config["predict"]["vis_confidence"] = args.vis_confidence
    if args.vis_max_boxes is not None:
        config["predict"]["vis_max_boxes"] = args.vis_max_boxes
    if args.save_failed_images:
        config["predict"]["save_failed_images"] = True
    if args.failed_iou is not None:
        config["predict"]["failed_iou"] = args.failed_iou
    if args.batch_size is not None:
        config["predict"]["batch_size"] = args.batch_size
    if args.json_path is not None:
        config["predict"]["json_path"] = args.json_path
    if args.dataset_img_path is not None:
        config["predict"]["dataset_img_path"] = args.dataset_img_path
    if args.dataset_root_prefix is not None:
        config["dataset_root_prefix"] = args.dataset_root_prefix

    print("=" * 80)
    print("Auto predict:", dataset_name, network_name)
    if run_tag:
        print("Run tag:", run_tag)
    print("Selected weight:", model_path)
    run_predict(config, presets, dry_run=args.dry_run)


def main():
    args = parse_args()
    presets = load_json(args.presets)
    base_config = load_json(args.config)
    predict_cfg = dict(base_config.get("predict", {}))
    effective_confidence = args.confidence if args.confidence is not None else predict_cfg.get("confidence", 0.001)
    effective_nms_iou = args.nms_iou if args.nms_iou is not None else predict_cfg.get("nms_iou", 0.65)
    run_tag = _slug(args.run_tag) if args.run_tag else make_auto_run_tag(
        args.weight_policy,
        effective_confidence,
        effective_nms_iou,
    )

    if args.all_latest:
        datasets = args.datasets or list(presets.get("dataset_presets", {}).keys())
        networks = args.networks or canonical_networks(presets)
        failures = []

        for dataset_name in datasets:
            for network_name in networks:
                try:
                    model_path = resolve_latest_weight(
                        args.train_output_root,
                        dataset_name,
                        network_name,
                        policy=args.weight_policy,
                    )
                except FileNotFoundError as exc:
                    message = "[Skip] %s/%s: %s" % (dataset_name, network_name, exc)
                    if args.strict:
                        failures.append(message)
                    else:
                        print(message)
                    continue

                run_one(args, presets, base_config, dataset_name, network_name, model_path, run_tag)

        if failures:
            raise FileNotFoundError("\n".join(failures))
        return

    dataset_name = args.dataset or base_config.get("dataset", "")
    network_name = args.network or base_config.get("network", "")
    if not dataset_name or not network_name:
        raise ValueError("Single-run mode requires --dataset and --network or config defaults.")

    model_path = args.model_path
    if not model_path:
        model_path = resolve_latest_weight(
            args.train_output_root,
            dataset_name,
            network_name,
            policy=args.weight_policy,
        )

    run_one(args, presets, base_config, dataset_name, network_name, model_path, run_tag)


if __name__ == "__main__":
    main()
