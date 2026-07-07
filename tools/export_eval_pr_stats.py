import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path


DEFAULT_PRESETS_PATH = "configs/experiment_presets.json"
DEFAULT_PREDICT_ROOT = "result/predict"
EVAL_FILE_NAMES = ("eval_results.json", "eval_result.json")


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


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def slug(value):
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def format_tag_float(value):
    if value is None:
        return "na"
    return slug(f"{float(value):g}")


def normalize_weight_policy(policy):
    normalized = WEIGHT_POLICY_ALIASES.get(str(policy).lower())
    if not normalized:
        raise ValueError("Unknown weight policy: %s" % policy)
    return normalized


def run_tag_candidates(weight_policy, confidence, nms_iou):
    policy = normalize_weight_policy(weight_policy)
    conf = format_tag_float(confidence)
    nms = format_tag_float(nms_iou)
    names = ["wp-%s_conf%s_nms%s" % (slug(policy), conf, nms)]
    if policy == "ap50":
        names.append("wp-map50_conf%s_nms%s" % (conf, nms))
    elif policy == "ap50:95":
        names.extend(
            [
                "wp-map50-95_conf%s_nms%s" % (conf, nms),
                "wp-map5095_conf%s_nms%s" % (conf, nms),
            ]
        )
    return list(dict.fromkeys(names))


def iou_label(iou_threshold):
    return ("%.2f" % float(iou_threshold)).replace(".", "p")


def xywh_iou(box_a, box_b):
    ax1, ay1, aw, ah = [float(v) for v in box_a[:4]]
    bx1, by1, bw, bh = [float(v) for v in box_b[:4]]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def safe_median(values):
    if not values:
        return None
    return float(statistics.median(values))


def load_gt_annotations(gt_json_path, class_agnostic=False):
    dataset = load_json(gt_json_path)
    gt_by_key = {}
    total_gt = 0

    for ann in dataset.get("annotations", []):
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        image_id = int(ann["image_id"])
        category_id = 0 if class_agnostic else int(ann.get("category_id", 0))
        key = (image_id, category_id)
        gt_by_key.setdefault(key, []).append(
            {
                "ann_id": int(ann.get("id", total_gt + 1)),
                "bbox": [float(v) for v in bbox[:4]],
                "matched": False,
            }
        )
        total_gt += 1

    image_count = len(dataset.get("images", []))
    categories = dataset.get("categories", [])
    return gt_by_key, total_gt, image_count, len(categories)


def load_predictions(eval_json_path, class_agnostic=False):
    raw_predictions = load_json(eval_json_path)
    predictions = []
    for index, pred in enumerate(raw_predictions):
        bbox = pred.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        try:
            score = float(pred.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        image_id = int(pred["image_id"])
        category_id = 0 if class_agnostic else int(pred.get("category_id", 0))
        predictions.append(
            {
                "input_index": index,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(v) for v in bbox[:4]],
                "score": score,
            }
        )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def voc_ap_from_points(points):
    if not points:
        return 0.0
    recalls = [float(row["recall"]) for row in points]
    precisions = [float(row["precision"]) for row in points]
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return float(ap)


def evaluate_predictions(eval_json_path, gt_json_path, iou_threshold=0.5, class_agnostic=False):
    gt_by_key, total_gt, image_count, category_count = load_gt_annotations(gt_json_path, class_agnostic=class_agnostic)
    predictions = load_predictions(eval_json_path, class_agnostic=class_agnostic)

    tp = 0
    fp = 0
    tp_scores = []
    fp_scores = []
    rows = []

    for rank, pred in enumerate(predictions, start=1):
        key = (pred["image_id"], pred["category_id"])
        candidates = gt_by_key.get(key, [])
        best_iou = 0.0
        best_gt = None
        for gt in candidates:
            if gt["matched"]:
                continue
            iou = xywh_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt is not None and best_iou >= iou_threshold:
            best_gt["matched"] = True
            tp += 1
            status = "TP"
            matched_ann_id = best_gt["ann_id"]
            tp_scores.append(pred["score"])
        else:
            fp += 1
            status = "FP"
            matched_ann_id = ""
            fp_scores.append(pred["score"])

        fn = max(total_gt - tp, 0)
        precision = float(tp / (tp + fp)) if (tp + fp) else 1.0
        recall = float(tp / total_gt) if total_gt else 0.0
        bbox = pred["bbox"]
        rows.append(
            {
                "rank": rank,
                "score_threshold": float(pred["score"]),
                "precision": precision,
                "recall": recall,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "status": status,
                "image_id": int(pred["image_id"]),
                "category_id": int(pred["category_id"]),
                "matched_iou": float(best_iou),
                "matched_ann_id": matched_ann_id,
                "bbox_x": float(bbox[0]),
                "bbox_y": float(bbox[1]),
                "bbox_w": float(bbox[2]),
                "bbox_h": float(bbox[3]),
            }
        )

    final_fn = max(total_gt - tp, 0)
    summary = {
        "eval_json": str(eval_json_path),
        "gt_json": str(gt_json_path),
        "iou_threshold": float(iou_threshold),
        "class_agnostic": bool(class_agnostic),
        "num_images": int(image_count),
        "num_categories": int(category_count),
        "num_gt": int(total_gt),
        "num_predictions": int(len(predictions)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(final_fn),
        "precision_final": float(tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall_final": float(tp / total_gt) if total_gt else 0.0,
        "ap": voc_ap_from_points(rows),
        "tp_score_median": safe_median(tp_scores),
        "fp_score_median": safe_median(fp_scores),
    }
    return rows, summary


def write_pr_outputs(run_dir, rows, summary, iou_threshold):
    label = iou_label(iou_threshold)
    curve_path = run_dir / "pr_curve.json"
    stats_json_path = run_dir / ("pr_stats_iou%s.json" % label)
    stats_csv_path = run_dir / ("pr_stats_iou%s.csv" % label)
    details_path = run_dir / ("pr_match_details_iou%s.csv" % label)

    for stale_path in (
        run_dir / ("pr_points_iou%s.csv" % label),
        run_dir / ("pr_summary_iou%s.json" % label),
        run_dir / ("pr_summary_iou%s.csv" % label),
    ):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass

    curve = {
        "iou_threshold": float(iou_threshold),
        "points": [
            {
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "score_threshold": float(row["score_threshold"]),
            }
            for row in rows
        ],
    }
    dump_json(curve_path, curve)

    detail_fields = [
        "rank",
        "score_threshold",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "status",
        "image_id",
        "category_id",
        "matched_iou",
        "matched_ann_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(rows)

    dump_json(stats_json_path, summary)

    summary_fields = [
        "dataset",
        "network",
        "run_tag",
        "run_name",
        "eval_json",
        "gt_json",
        "iou_threshold",
        "num_images",
        "num_gt",
        "num_predictions",
        "tp",
        "fp",
        "fn",
        "precision_final",
        "recall_final",
        "ap",
        "tp_score_median",
        "fp_score_median",
    ]
    with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({key: summary.get(key, "") for key in summary_fields})

    return curve_path, stats_json_path, stats_csv_path, details_path


def load_presets_gt_map(presets_path):
    if not presets_path:
        return {}
    path = Path(presets_path)
    if not path.exists():
        return {}
    presets = load_json(path)
    out = {}
    for dataset, cfg in presets.get("dataset_presets", {}).items():
        gt_path = cfg.get("default_predict", {}).get("json_path", "")
        if gt_path:
            out[str(dataset)] = str(gt_path)
    return out


def parse_gt_map_items(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--gt-map entries must look like DATASET=path/to/gt.json")
        dataset, path = item.split("=", 1)
        dataset = dataset.strip()
        path = path.strip()
        if not dataset or not path:
            raise ValueError("--gt-map entries must look like DATASET=path/to/gt.json")
        out[dataset] = path
    return out


def resolve_gt_json(dataset, gt_json, gt_map, presets_gt_map):
    if gt_json:
        return Path(gt_json)
    if dataset in gt_map:
        return Path(gt_map[dataset])
    if dataset in presets_gt_map:
        return Path(presets_gt_map[dataset])
    return None


def find_eval_file(run_dir):
    run_dir = Path(run_dir)
    for name in EVAL_FILE_NAMES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def parse_run_context(eval_path, predict_root):
    eval_path = Path(eval_path)
    run_dir = eval_path.parent
    context = {
        "dataset": "",
        "network": "",
        "run_tag": "",
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
    }
    try:
        rel_parts = run_dir.relative_to(Path(predict_root)).parts
    except ValueError:
        rel_parts = ()
    if len(rel_parts) >= 1:
        context["dataset"] = rel_parts[0]
    if len(rel_parts) >= 2:
        context["network"] = rel_parts[1]
    if len(rel_parts) >= 3:
        context["run_tag"] = rel_parts[2]
    if len(rel_parts) >= 4:
        context["run_name"] = rel_parts[3]
    return context


def discover_eval_files(predict_root):
    root = Path(predict_root)
    if not root.exists():
        return []
    paths = []
    for name in EVAL_FILE_NAMES:
        paths.extend(root.rglob(name))
    return sorted(set(paths), key=lambda item: str(item))


def filter_eval_files(eval_files, predict_root, datasets=None, networks=None, run_tags=None):
    datasets = set(datasets or [])
    networks = set(networks or [])
    run_tags = set(run_tags or [])
    out = []
    for eval_file in eval_files:
        context = parse_run_context(eval_file, predict_root)
        if datasets and context["dataset"] not in datasets:
            continue
        if networks and context["network"] not in networks:
            continue
        if run_tags and context["run_tag"] not in run_tags:
            continue
        out.append(eval_file)
    return out


def select_latest_by_group(eval_files, predict_root, group_keys):
    latest = {}
    for eval_file in eval_files:
        context = parse_run_context(eval_file, predict_root)
        key = tuple(context.get(item, "") for item in group_keys)
        mtime = eval_file.stat().st_mtime
        marker = (mtime, context.get("run_name", ""), str(eval_file))
        if key not in latest or marker > latest[key][0]:
            latest[key] = (marker, eval_file)
    return [item[1] for item in sorted(latest.values(), key=lambda value: value[0])]


def process_eval_file(eval_file, gt_json, predict_root, args):
    run_dir = Path(eval_file).parent
    context = parse_run_context(eval_file, predict_root)
    rows, summary = evaluate_predictions(
        eval_file,
        gt_json,
        iou_threshold=args.iou,
        class_agnostic=args.class_agnostic,
    )
    summary.update(context)
    curve_path, stats_json_path, stats_csv_path, details_path = write_pr_outputs(run_dir, rows, summary, args.iou)
    print(
        "Processed %s | TP=%d FP=%d FN=%d AP=%.6f"
        % (run_dir, summary["tp"], summary["fp"], summary["fn"], summary["ap"])
    )
    return summary, curve_path, stats_json_path, stats_csv_path, details_path


def write_global_summary(path, summaries):
    if not path:
        return
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "network",
        "run_tag",
        "run_name",
        "run_dir",
        "eval_json",
        "gt_json",
        "iou_threshold",
        "num_images",
        "num_gt",
        "num_predictions",
        "tp",
        "fp",
        "fn",
        "precision_final",
        "recall_final",
        "ap",
        "tp_score_median",
        "fp_score_median",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fields})
    print("Saved global summary:", path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export pr_curve.json plus separate TP/FP/FN score stats from prediction eval_results.json files."
    )
    parser.add_argument("--predict-root", default=DEFAULT_PREDICT_ROOT, help="Prediction output root.")
    parser.add_argument("--presets", default=DEFAULT_PRESETS_PATH, help="Dataset preset json used to infer GT json paths.")
    parser.add_argument("--eval-json", help="Process one eval_results.json directly.")
    parser.add_argument("--run-dir", help="Process one prediction run directory containing eval_results.json.")
    parser.add_argument("--gt-json", "--gt_json", dest="gt_json", help="GT COCO json path. Required for direct eval-json unless inferable.")
    parser.add_argument("--gt-map", action="append", default=[], help="Dataset-specific GT mapping, e.g. DAUB=data_json/DAUB_test.json.")
    parser.add_argument("--dataset", help="Dataset name to process.")
    parser.add_argument("--network", help="Network name to process.")
    parser.add_argument("--datasets", nargs="+", help="Datasets to process with --all-latest or --all.")
    parser.add_argument("--networks", nargs="+", help="Networks to process with --all-latest or --all.")
    parser.add_argument("--weight-policy", default="ap50", help="Prediction run-tag weight policy, e.g. ap50/ap50:95/loss/last.")
    parser.add_argument("--confidence", type=float, default=0.001, help="Confidence value encoded in auto run tag.")
    parser.add_argument("--nms-iou", "--nms_iou", dest="nms_iou", type=float, default=0.65, help="NMS value encoded in auto run tag.")
    parser.add_argument("--run-tag", help="Exact run-tag folder name. Overrides weight-policy/confidence/nms tag matching.")
    parser.add_argument("--all", action="store_true", help="Process every matching eval_results.json under predict-root.")
    parser.add_argument("--all-latest", action="store_true", help="Process the latest matching prediction folder for each dataset/network.")
    parser.add_argument("--latest-per-run-tag", action="store_true", help="With --all-latest, keep one latest run per dataset/network/run-tag.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching.")
    parser.add_argument("--class-agnostic", action="store_true", help="Ignore category_id when matching predictions to GT.")
    parser.add_argument("--summary-csv", default="", help="Optional global summary CSV path.")
    parser.add_argument("--dry-run", action="store_true", help="Only show selected eval files.")
    return parser.parse_args()


def main():
    args = parse_args()
    predict_root = Path(args.predict_root)
    gt_map = parse_gt_map_items(args.gt_map)
    presets_gt_map = load_presets_gt_map(args.presets)

    if args.run_dir:
        eval_file = find_eval_file(args.run_dir)
        if eval_file is None:
            raise FileNotFoundError("No eval_results.json/eval_result.json under: %s" % args.run_dir)
        eval_files = [eval_file]
    elif args.eval_json:
        eval_files = [Path(args.eval_json)]
    else:
        all_eval_files = discover_eval_files(predict_root)
        datasets = args.datasets or ([args.dataset] if args.dataset else None)
        networks = args.networks or ([args.network] if args.network else None)
        if args.run_tag:
            run_tags = [args.run_tag]
        else:
            run_tags = run_tag_candidates(args.weight_policy, args.confidence, args.nms_iou)

        eval_files = filter_eval_files(
            all_eval_files,
            predict_root,
            datasets=datasets,
            networks=networks,
            run_tags=run_tags,
        )

        if args.all_latest or not args.all:
            group_keys = ["dataset", "network", "run_tag"] if args.latest_per_run_tag else ["dataset", "network"]
            eval_files = select_latest_by_group(eval_files, predict_root, group_keys)

    if not eval_files:
        raise FileNotFoundError("No matching eval_results.json/eval_result.json files found.")

    print("Selected eval files:")
    for eval_file in eval_files:
        print(" ", eval_file)
    if args.dry_run:
        return

    summaries = []
    for eval_file in eval_files:
        context = parse_run_context(eval_file, predict_root)
        dataset = args.dataset or context.get("dataset", "")
        gt_json = resolve_gt_json(dataset, args.gt_json, gt_map, presets_gt_map)
        if gt_json is None:
            raise ValueError(
                "Cannot infer GT json for %s. Pass --gt-json or --gt-map %s=path."
                % (eval_file, dataset or "DATASET")
            )
        if not gt_json.exists():
            raise FileNotFoundError("GT json not found: %s" % gt_json)
        summary, _, _, _, _ = process_eval_file(eval_file, gt_json, predict_root, args)
        summaries.append(summary)

    summary_csv = args.summary_csv
    if not summary_csv and len(summaries) > 1:
        summary_csv = str(Path(args.predict_root) / ("pr_stats_summary_iou%s.csv" % iou_label(args.iou)))
    write_global_summary(summary_csv, summaries)


if __name__ == "__main__":
    main()
