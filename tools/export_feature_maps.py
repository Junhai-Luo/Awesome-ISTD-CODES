import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = None


SEQUENCE_NETWORKS = {
    "sstnet",
    "tridos",
    "slowfastnet",
    "slowfastnet_9520",
    "dqaligner",
    "dqaligner_det",
    "dqaligner_saliency",
    "dqaligner_saliency_det",
}


AUTO_LAYER_PATTERNS = [
    r"(^|\.)head\.stems\.\d+$",
    r"(^|\.)head\.reg_convs\.\d+$",
    r"(^|\.)head\.cls_convs\.\d+$",
    r"(^|\.)head\.reg_convs$",
    r"(^|\.)head\.cls_convs$",
    r"(^|\.)head\.reg_conv$",
    r"(^|\.)head\.cls_conv$",
    r"(^|\.)head\.stem$",
    r"(^|\.)det_head_s8\.reg_conv$",
    r"(^|\.)det_head_s8\.cls_conv$",
    r"(^|\.)det_head_s8\.stem$",
    r"(^|\.)downsample$",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export feature-map heatmaps from intermediate layers during a prediction-style forward pass."
    )
    parser.add_argument("--network", required=True, help="Network name accepted by nets.factory.build_network.")
    parser.add_argument("--model-path", default="", help="Checkpoint path. If omitted, uses randomly initialized weights.")
    parser.add_argument("--image-path", default="", help="Direct image path.")
    parser.add_argument("--image-list", default="", help="Txt file with one image path per line. Extra columns are ignored.")
    parser.add_argument("--json-path", default="", help="COCO json path. Use with --index or --image-id.")
    parser.add_argument("--dataset-img-path", default="", help="Dataset image root used to resolve COCO file_name.")
    parser.add_argument("--index", type=int, default=0, help="Image index in COCO json when --image-id is not set.")
    parser.add_argument("--indices", default="", help="Comma/range COCO indices, for example 0,1,10-20.")
    parser.add_argument("--image-id", type=int, default=None, help="COCO image id to export.")
    parser.add_argument("--image-ids", default="", help="Comma/range COCO image ids, for example 1,2,100-120.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit exported images after selection. 0 means no limit.")
    parser.add_argument("--output-dir", default="result/feature_maps", help="Output root directory.")
    parser.add_argument("--input-size", type=int, default=512, help="Model input size.")
    parser.add_argument("--num-frame", type=int, default=5, help="Temporal frame count for sequence networks.")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of detection classes.")
    parser.add_argument("--layers", nargs="*", default=["auto"], help="Layer names, suffixes, substrings, or auto.")
    parser.add_argument("--topk", type=int, default=8, help="How many strongest channels to export per layer.")
    parser.add_argument("--device", default="", help="cuda, cuda:0, or cpu. Default auto-selects CUDA when available.")
    parser.add_argument("--letterbox", action=argparse.BooleanOptionalAction, default=True, help="Use letterbox resize.")
    parser.add_argument("--list-layers", action="store_true", help="Print all hookable module names and exit.")
    return parser.parse_args()


def is_sequence_network(name):
    return str(name or "").lower() in SEQUENCE_NETWORKS


def load_checkpoint(model, model_path, device):
    if not model_path:
        print("No --model-path given; exporting features from randomly initialized weights.")
        return
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    model_state = model.state_dict()
    loadable = {}
    skipped = []
    for key, value in ckpt.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if torch.is_tensor(value) and clean_key in model_state and tuple(model_state[clean_key].shape) == tuple(value.shape):
            loadable[clean_key] = value
        else:
            skipped.append(clean_key)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print(
        f"Loaded checkpoint: {model_path}\n"
        f"  loaded={len(loadable)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}"
    )


def load_image_from_json(json_path, dataset_img_path, index=0, image_id=None):
    from infer.predict_from_coco_json import resolve_dataset_image_path

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    if not images:
        raise ValueError(f"No images found in COCO json: {json_path}")

    selected = None
    if image_id is not None:
        for item in images:
            if int(item.get("id")) == int(image_id):
                selected = item
                break
        if selected is None:
            raise ValueError(f"image_id={image_id} not found in {json_path}")
    else:
        if index < 0 or index >= len(images):
            raise IndexError(f"--index {index} out of range, json has {len(images)} images")
        selected = images[index]

    json_dir = os.path.dirname(os.path.abspath(json_path))
    image_path = resolve_dataset_image_path(selected.get("file_name", ""), dataset_img_path, json_dir)
    return image_path, selected


def parse_int_selector(value):
    if not value:
        return []
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start.strip())
            end = int(end.strip())
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(part))
    return out


def collect_targets(args):
    targets = []

    if args.image_path:
        targets.append({"image_path": args.image_path, "image_meta": None, "source": "image_path"})

    if args.image_list:
        with open(args.image_list, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                image_path = line.split()[0]
                targets.append(
                    {
                        "image_path": image_path,
                        "image_meta": {"line_index": line_idx, "raw_line": line},
                        "source": "image_list",
                    }
                )

    if args.json_path:
        if args.image_ids:
            for image_id in parse_int_selector(args.image_ids):
                image_path, image_meta = load_image_from_json(
                    args.json_path, args.dataset_img_path, index=0, image_id=image_id
                )
                targets.append({"image_path": image_path, "image_meta": image_meta, "source": "json"})
        elif args.indices:
            for index in parse_int_selector(args.indices):
                image_path, image_meta = load_image_from_json(
                    args.json_path, args.dataset_img_path, index=index, image_id=None
                )
                targets.append({"image_path": image_path, "image_meta": image_meta, "source": "json"})
        elif not targets:
            image_path, image_meta = load_image_from_json(
                args.json_path, args.dataset_img_path, args.index, args.image_id
            )
            targets.append({"image_path": image_path, "image_meta": image_meta, "source": "json"})

    if not targets:
        raise ValueError("Provide --image-path, --image-list, or --json-path.")

    if args.max_images and args.max_images > 0:
        targets = targets[: args.max_images]
    return targets


def image_to_tensor(image, input_size, letterbox):
    from utils.utils import cvtColor, preprocess_input, resize_image

    image = cvtColor(image)
    resized = resize_image(image, (input_size, input_size), letterbox)
    arr = np.asarray(resized, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr), resized


def build_input_tensor(image_path, network_name, input_size, num_frame, letterbox):
    from utils.dataloader_for_sequence import _history_frame_paths

    if is_sequence_network(network_name):
        frame_paths = _history_frame_paths(image_path, num_frame)
        tensors = []
        preview = None
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor, resized = image_to_tensor(image, input_size, letterbox)
            tensors.append(tensor)
            preview = resized
        stacked = torch.stack(tensors, dim=1).unsqueeze(0)
        return stacked, preview, frame_paths

    image = Image.open(image_path).convert("RGB")
    tensor, resized = image_to_tensor(image, input_size, letterbox)
    return tensor.unsqueeze(0), resized, [image_path]


def named_modules_without_root(model):
    return [(name, module) for name, module in model.named_modules() if name]


def is_auto_layer(name):
    return any(re.search(pattern, name) for pattern in AUTO_LAYER_PATTERNS)


def resolve_layers(model, requested):
    modules = dict(named_modules_without_root(model))
    names = list(modules.keys())
    requested = requested or ["auto"]

    if len(requested) == 1 and requested[0].lower() == "auto":
        selected = [name for name in names if is_auto_layer(name)]
        if selected:
            return selected
        fallback = [
            name
            for name in names
            if "head" in name and "pred" not in name and not isinstance(modules[name], torch.nn.ModuleList)
        ]
        return fallback[-4:]

    selected = []
    for item in requested:
        if item in modules:
            selected.append(item)
            continue
        suffix_matches = [name for name in names if name.endswith(item)]
        if suffix_matches:
            selected.extend(suffix_matches)
            continue
        substring_matches = [name for name in names if item in name]
        if substring_matches:
            selected.extend(substring_matches)
            continue
        raise ValueError(f"Layer '{item}' not found. Use --list-layers to inspect available names.")

    deduped = []
    seen = set()
    for name in selected:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(output, dict):
        for item in output.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def normalize_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(arr.min()) if arr.size else 0.0
    max_v = float(arr.max()) if arr.size else 0.0
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - min_v) / (max_v - min_v) * 255.0).clip(0, 255).astype(np.uint8)


def colorize_heatmap(gray):
    gray = normalize_uint8(gray)
    try:
        import matplotlib.cm as cm

        colored = cm.get_cmap("jet")(gray.astype(np.float32) / 255.0)[..., :3]
        return (colored * 255.0).astype(np.uint8)
    except Exception:
        return np.stack([gray, np.zeros_like(gray), 255 - gray], axis=-1)


def tensor_to_chw(tensor):
    feature = tensor.detach().float().cpu()
    if feature.dim() == 4:
        feature = feature[0]
    elif feature.dim() == 5:
        feature = feature[0].mean(dim=1)
    elif feature.dim() == 3:
        pass
    else:
        return None
    if feature.dim() != 3:
        return None
    return feature


def save_layer_outputs(layer_name, tensor, output_dir, preview_image, topk):
    feature = tensor_to_chw(tensor)
    if feature is None:
        return {
            "layer": layer_name,
            "shape": list(tensor.shape),
            "saved": False,
            "reason": "unsupported tensor rank",
        }

    abs_feature = feature.abs()
    heatmap = abs_feature.mean(dim=0).numpy()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", layer_name)
    layer_dir = output_dir / safe_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    heat_rgb = colorize_heatmap(heatmap)
    heat_img = Image.fromarray(heat_rgb).resize(preview_image.size, Image.BICUBIC)
    heat_path = layer_dir / "heatmap.png"
    heat_img.save(heat_path)

    overlay = Image.blend(preview_image.convert("RGB"), heat_img.convert("RGB"), alpha=0.45)
    overlay_path = layer_dir / "overlay.png"
    overlay.save(overlay_path)

    channel_scores = abs_feature.flatten(1).mean(dim=1).numpy()
    topk = max(0, min(int(topk), int(abs_feature.shape[0])))
    top_indices = np.argsort(-channel_scores)[:topk].tolist() if topk else []
    channel_files = []
    for rank, channel_idx in enumerate(top_indices, start=1):
        channel_map = feature[channel_idx].numpy()
        channel_img = Image.fromarray(colorize_heatmap(channel_map)).resize(preview_image.size, Image.BICUBIC)
        path = layer_dir / f"channel_rank{rank:02d}_c{channel_idx:04d}.png"
        channel_img.save(path)
        channel_files.append(str(path))

    return {
        "layer": layer_name,
        "shape": list(tensor.shape),
        "feature_chw": list(feature.shape),
        "saved": True,
        "heatmap": str(heat_path),
        "overlay": str(overlay_path),
        "top_channels": top_indices,
        "top_channel_files": channel_files,
        "min": float(feature.min().item()),
        "max": float(feature.max().item()),
        "mean": float(feature.mean().item()),
    }


def export_one_target(args, model, modules, layers, device, target, run_timestamp, ordinal):
    image_path = target["image_path"]
    image_meta = target.get("image_meta")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    input_tensor, preview_image, frame_paths = build_input_tensor(
        image_path=image_path,
        network_name=args.network,
        input_size=args.input_size,
        num_frame=args.num_frame,
        letterbox=args.letterbox,
    )
    input_tensor = input_tensor.to(device)

    activations = {}
    handles = []
    for layer_name in layers:
        def make_hook(name):
            def hook(_module, _inputs, output):
                tensor = first_tensor(output)
                if tensor is not None:
                    activations[name] = tensor.detach()

            return hook

        handles.append(modules[layer_name].register_forward_hook(make_hook(layer_name)))

    with torch.no_grad():
        _ = model(input_tensor)

    for handle in handles:
        handle.remove()

    image_stem = Path(image_path).stem
    image_tag = f"{ordinal:06d}_{image_stem}_{run_timestamp}"
    output_dir = Path(args.output_dir) / args.network / image_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_image.save(output_dir / "input_resized.png")

    summary = {
        "network": args.network,
        "model_path": args.model_path,
        "image_path": image_path,
        "image_meta": image_meta,
        "source": target.get("source"),
        "frame_paths": frame_paths,
        "input_shape": list(input_tensor.shape),
        "input_size": args.input_size,
        "letterbox": args.letterbox,
        "selected_layers": layers,
        "layers": [],
    }
    for layer_name in layers:
        if layer_name not in activations:
            summary["layers"].append(
                {"layer": layer_name, "saved": False, "reason": "hook did not receive tensor output"}
            )
            continue
        summary["layers"].append(
            save_layer_outputs(layer_name, activations[layer_name], output_dir, preview_image, args.topk)
        )

    summary_path = output_dir / "feature_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {"image_path": image_path, "output_dir": str(output_dir), "summary": str(summary_path)}


def main():
    global torch

    args = parse_args()
    import torch as _torch
    from nets.factory import build_network

    torch = _torch
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    model = build_network(args.network, num_classes=args.num_classes, num_frame=args.num_frame)
    model.to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    if args.list_layers:
        for name, module in named_modules_without_root(model):
            print(f"{name}\t{module.__class__.__name__}")
        return

    layers = resolve_layers(model, args.layers)
    if not layers:
        raise ValueError("No layers selected. Use --list-layers and pass --layers explicitly.")

    modules = dict(named_modules_without_root(model))
    targets = collect_targets(args)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_summary = {
        "network": args.network,
        "model_path": args.model_path,
        "input_size": args.input_size,
        "letterbox": args.letterbox,
        "selected_layers": layers,
        "count": len(targets),
        "items": [],
    }
    print(f"Selected layers: {', '.join(layers)}")
    print(f"Exporting {len(targets)} image(s).")
    for ordinal, target in enumerate(targets):
        item = export_one_target(args, model, modules, layers, device, target, run_timestamp, ordinal)
        batch_summary["items"].append(item)
        print(f"[{ordinal + 1}/{len(targets)}] {item['image_path']} -> {item['output_dir']}")

    summary_root = Path(args.output_dir) / args.network
    summary_root.mkdir(parents=True, exist_ok=True)
    batch_summary_path = summary_root / f"feature_batch_summary_{run_timestamp}.json"
    with open(batch_summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)

    print(f"Batch summary: {batch_summary_path}")


if __name__ == "__main__":
    main()
