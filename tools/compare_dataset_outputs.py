import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _image_stats(image):
    image = _to_numpy(image).astype(np.float32)
    return {
        "shape": list(image.shape),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
    }


def _box_stats(boxes):
    boxes = _to_numpy(boxes).astype(np.float32)
    if boxes.size == 0:
        boxes = np.zeros((0, 5), dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(-1, 5)

    if len(boxes) == 0:
        return {
            "shape": list(boxes.shape),
            "count": 0,
            "first": [],
            "cx_mean": None,
            "cy_mean": None,
            "w_mean": None,
            "h_mean": None,
            "w_min": None,
            "h_min": None,
        }

    return {
        "shape": list(boxes.shape),
        "count": int(len(boxes)),
        "first": boxes[: min(5, len(boxes))].round(4).tolist(),
        "cx_mean": float(np.mean(boxes[:, 0])),
        "cy_mean": float(np.mean(boxes[:, 1])),
        "w_mean": float(np.mean(boxes[:, 2])),
        "h_mean": float(np.mean(boxes[:, 3])),
        "w_min": float(np.min(boxes[:, 2])),
        "h_min": float(np.min(boxes[:, 3])),
    }


def _compare_arrays(left, right):
    left = _to_numpy(left).astype(np.float32)
    right = _to_numpy(right).astype(np.float32)
    if left.shape != right.shape:
        return {"same_shape": False, "max_abs": None, "mean_abs": None}
    diff = np.abs(left - right)
    return {
        "same_shape": True,
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
    }


def _set_detlab_precision(mode, coord_type="float"):
    geometry = "int" if mode == "truncate" else "float"
    os.environ["DETLAB_BOX_COORD_TYPE"] = coord_type
    os.environ["DATA_DETLAB_BOX_COORD_TYPE"] = coord_type
    os.environ["ACM_BOX_COORD_TYPE"] = coord_type
    os.environ["ALC_BOX_COORD_TYPE"] = coord_type
    os.environ["UIU_BOX_COORD_TYPE"] = coord_type
    os.environ["DETLAB_BOX_GEOMETRY_DTYPE"] = geometry
    os.environ["DATA_DETLAB_BOX_GEOMETRY_DTYPE"] = geometry
    os.environ["ACM_BOX_GEOMETRY_DTYPE"] = geometry
    os.environ["ALC_BOX_GEOMETRY_DTYPE"] = geometry
    os.environ["UIU_BOX_GEOMETRY_DTYPE"] = geometry


def _make_dataset(args, train, mosaic=False, mixup=False):
    from utils.acm.data_detlab import DetlabTxtDetDataset

    return DetlabTxtDetDataset(
        txt_path=args.train_txt,
        input_size=args.input_size,
        image_root=args.image_root,
        train=train,
        mosaic=mosaic,
        mixup=mixup,
        mosaic_prob=args.mosaic_prob,
        mixup_prob=args.mixup_prob,
        epoch_length=args.epochs,
        special_aug_ratio=args.special_aug_ratio,
    )


def _parse_indices(text, length):
    if text:
        return [int(item) for item in text.split(",") if item.strip()]
    count = min(length, 20)
    if count <= 0:
        return []
    return np.linspace(0, length - 1, count, dtype=int).tolist()


def _sample_pair(high_ds, truncate_ds, index, seed):
    _set_seed(seed)
    high_image, high_boxes = high_ds[index]
    _set_seed(seed)
    truncate_image, truncate_boxes = truncate_ds[index]
    return high_image, high_boxes, truncate_image, truncate_boxes


def main():
    parser = argparse.ArgumentParser(description="Compare DETLAB high-precision and truncate box outputs.")
    parser.add_argument("--train-txt", default="data_txt/IRDST_train.txt")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--indices", default="", help="Comma separated sample indices. Default: 20 evenly spaced samples.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", choices=["letterbox", "random-single", "mosaic"], default="letterbox")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mosaic-prob", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--special-aug-ratio", type=float, default=0.7)
    parser.add_argument("--box-coord-type", choices=["float", "int"], default="float")
    parser.add_argument("--out-dir", default="result/dataset_compare")
    args = parser.parse_args()

    if args.mode == "letterbox":
        train = False
        mosaic = False
        mixup = False
    elif args.mode == "random-single":
        train = True
        mosaic = False
        mixup = False
    else:
        train = True
        mosaic = True
        mixup = False

    _set_detlab_precision("high", coord_type=args.box_coord_type)
    high_ds = _make_dataset(args, train=train, mosaic=mosaic, mixup=mixup)
    _set_detlab_precision("truncate", coord_type=args.box_coord_type)
    truncate_ds = _make_dataset(args, train=train, mosaic=mosaic, mixup=mixup)
    _set_detlab_precision("high", coord_type=args.box_coord_type)

    indices = _parse_indices(args.indices, min(len(high_ds), len(truncate_ds)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []

    for pos, index in enumerate(indices):
        seed = args.seed + pos
        high_image, high_boxes, truncate_image, truncate_boxes = _sample_pair(high_ds, truncate_ds, index, seed)
        image_diff = _compare_arrays(high_image, truncate_image)
        box_diff = _compare_arrays(high_boxes, truncate_boxes)
        high_img_stats = _image_stats(high_image)
        truncate_img_stats = _image_stats(truncate_image)
        high_box_stats = _box_stats(high_boxes)
        truncate_box_stats = _box_stats(truncate_boxes)

        row = {
            "index": index,
            "seed": seed,
            "image_same_shape": image_diff["same_shape"],
            "image_max_abs": image_diff["max_abs"],
            "image_mean_abs": image_diff["mean_abs"],
            "box_same_shape": box_diff["same_shape"],
            "box_max_abs": box_diff["max_abs"],
            "box_mean_abs": box_diff["mean_abs"],
            "high_image_mean": high_img_stats["mean"],
            "truncate_image_mean": truncate_img_stats["mean"],
            "high_image_std": high_img_stats["std"],
            "truncate_image_std": truncate_img_stats["std"],
            "high_box_count": high_box_stats["count"],
            "truncate_box_count": truncate_box_stats["count"],
            "high_w_mean": high_box_stats["w_mean"],
            "truncate_w_mean": truncate_box_stats["w_mean"],
            "high_h_mean": high_box_stats["h_mean"],
            "truncate_h_mean": truncate_box_stats["h_mean"],
        }
        rows.append(row)
        details.append(
            {
                "index": index,
                "seed": seed,
                "image_diff": image_diff,
                "box_diff": box_diff,
                "high_image": high_img_stats,
                "truncate_image": truncate_img_stats,
                "high_boxes": high_box_stats,
                "truncate_boxes": truncate_box_stats,
            }
        )

    summary_path = out_dir / "dataset_precision_compare_summary.csv"
    detail_path = out_dir / "dataset_precision_compare_details.json"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print("Compared samples:", len(rows))
    print("Summary:", summary_path)
    print("Details:", detail_path)


if __name__ == "__main__":
    main()
