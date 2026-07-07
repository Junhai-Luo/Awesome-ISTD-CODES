import argparse
import bisect
import csv
import datetime
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


METRIC_KEYS = [
    "AP@[0.50:0.95]",
    "AP@0.50",
    "AP@0.75",
    "Precision@0.5",
    "Recall@0.5",
    "F1@0.5",
    "AR@100",
]


def _slug(value):
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def _format_label_float(value):
    if value is None:
        return "na"
    return _slug(f"{float(value):g}")


def _weight_policy_tag(policy):
    aliases = {
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
    return aliases.get(str(policy or "ap50").lower(), str(policy or "ap50").lower())


def make_auto_run_tag(weight_policy, confidence, nms_iou):
    return "wp-%s_conf%s_nms%s" % (
        _slug(_weight_policy_tag(weight_policy)),
        _format_label_float(confidence),
        _format_label_float(nms_iou),
    )


def _make_run_output_dir(base_dir, args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = resolve_run_tag(args)
    parts = [
        timestamp,
        _slug(args.select),
        f"iou{_format_label_float(args.roc_iou)}",
    ]
    if run_tag:
        parts.append(run_tag)
    else:
        parts.extend(
            [
                f"conf{_format_label_float(args.confidence)}",
                f"nms{_format_label_float(args.nms_iou)}",
            ]
        )
    if args.tag:
        parts.append(_slug(args.tag))

    name = "_".join(part for part in parts if part)
    out_dir = base_dir / name
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{name}_{suffix:02d}"
        suffix += 1
    return out_dir


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def _format_score(value):
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def _point_count(value):
    try:
        return max(2, int(value))
    except (TypeError, ValueError):
        return 101


def _sample_pr_points(pr_points, point_count=101):
    """COCO-style 101-point PR sampling on evenly spaced recall thresholds."""
    point_count = _point_count(point_count)
    if not pr_points:
        return [(0.0, 1.0, 1.0)] + [(i / (point_count - 1), 0.0, 0.0) for i in range(1, point_count)]

    ordered = sorted(pr_points, key=lambda item: (item[0], -item[1]))
    recalls = [max(0.0, min(1.0, float(item[0]))) for item in ordered]
    precisions = [max(0.0, min(1.0, float(item[1]))) for item in ordered]
    scores = [float(item[2]) for item in ordered]

    best_precisions = [0.0] * len(ordered)
    best_scores = [0.0] * len(ordered)
    best_precision = 0.0
    best_score = 0.0
    for idx in range(len(ordered) - 1, -1, -1):
        if precisions[idx] >= best_precision:
            best_precision = precisions[idx]
            best_score = scores[idx]
        best_precisions[idx] = best_precision
        best_scores[idx] = best_score

    sampled = []
    for idx in range(point_count):
        recall = idx / (point_count - 1)
        point_idx = bisect.bisect_left(recalls, recall)
        if point_idx < len(ordered):
            sampled.append((recall, best_precisions[point_idx], best_scores[point_idx]))
        else:
            sampled.append((recall, 0.0, 0.0))
    sampled[0] = (0.0, max(sampled[0][1], 1.0), sampled[0][2])
    return sampled


def _sample_froc_points(points, point_count=101, max_fppi=None):
    point_count = _point_count(point_count)
    if not points:
        return [(0.0, 0.0, 1.0)] * point_count

    ordered = sorted(points, key=lambda item: item[0])
    target_max = max_fppi
    if target_max is None:
        target_max = max(float(item[0]) for item in ordered)
    target_max = max(0.0, float(target_max))
    grid = [target_max * idx / (point_count - 1) for idx in range(point_count)]

    sampled = []
    src_idx = 0
    last_recall = 0.0
    last_score = 1.0
    for idx, fppi in enumerate(grid):
        if idx == 0:
            sampled.append((0.0, 0.0, 1.0))
            continue
        while src_idx < len(ordered) and float(ordered[src_idx][0]) <= fppi:
            last_recall = max(0.0, min(1.0, float(ordered[src_idx][1])))
            last_score = float(ordered[src_idx][2])
            src_idx += 1
        sampled.append((fppi, last_recall, last_score))
    return sampled


def _load_json_metrics(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {}
    for key in ("Precision@0.5", "Recall@0.5", "F1@0.5"):
        row[key] = _safe_float(data.get(key))
    for key, value in data.get("stats", {}).items():
        row[key] = _safe_float(value)
    return row


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_predict_defaults(config_path):
    try:
        data = _load_json(Path(config_path))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("predict", {}) if isinstance(data, dict) else {}


def resolve_run_tag(args):
    tag = str(getattr(args, "run_tag", "") or "").strip()
    if tag and tag.lower() != "auto":
        return _slug(tag)

    defaults = _load_predict_defaults(getattr(args, "config", "configs/predict_experiment_config.json"))
    confidence = args.confidence if args.confidence is not None else defaults.get("confidence", 0.001)
    nms_iou = args.nms_iou if args.nms_iou is not None else defaults.get("nms_iou", 0.65)
    if tag.lower() == "auto" or args.weight_policy or args.confidence is not None or args.nms_iou is not None:
        return make_auto_run_tag(args.weight_policy or "ap50", confidence, nms_iou)
    return ""


def _load_summary_metrics(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    row = {}
    first = re.search(
        r"Precision:\s*([0-9.eE+-]+),\s*Recall:\s*([0-9.eE+-]+),\s*F1:\s*([0-9.eE+-]+)",
        text,
    )
    if first:
        row["Precision@0.5"] = _safe_float(first.group(1))
        row["Recall@0.5"] = _safe_float(first.group(2))
        row["F1@0.5"] = _safe_float(first.group(3))
    for key in METRIC_KEYS:
        match = re.search(re.escape(key) + r"\s*:\s*([0-9.eE+-]+)", text)
        if match:
            row[key] = _safe_float(match.group(1))
    return row


def _infer_names(root, run_dir):
    rel = run_dir.relative_to(root)
    parts = rel.parts
    dataset = parts[0] if len(parts) >= 1 else "unknown"
    network = parts[1] if len(parts) >= 2 else "unknown"
    if len(parts) >= 4:
        run_tag = parts[2]
        run_name = parts[3]
    else:
        run_tag = ""
        run_name = parts[2] if len(parts) >= 3 else run_dir.name
    return dataset, network, run_name, run_tag


def collect_runs(root, run_tag=""):
    rows = []
    metric_files = list(root.rglob("eval_metrics.json"))
    seen_dirs = {path.parent for path in metric_files}

    for path in sorted(metric_files):
        run_dir = path.parent
        dataset, network, run_name, row_run_tag = _infer_names(root, run_dir)
        if run_tag and row_run_tag != run_tag:
            continue
        metrics = _load_json_metrics(path)
        rows.append(
            {
                "dataset": dataset,
                "network": network,
                "run": run_name,
                "run_tag": row_run_tag,
                "run_dir": str(run_dir),
                "mtime": path.stat().st_mtime,
                **metrics,
            }
        )

    for path in sorted(root.rglob("result_summary.txt")):
        if path.parent in seen_dirs:
            continue
        run_dir = path.parent
        dataset, network, run_name, row_run_tag = _infer_names(root, run_dir)
        if run_tag and row_run_tag != run_tag:
            continue
        metrics = _load_summary_metrics(path)
        rows.append(
            {
                "dataset": dataset,
                "network": network,
                "run": run_name,
                "run_tag": row_run_tag,
                "run_dir": str(run_dir),
                "mtime": path.stat().st_mtime,
                **metrics,
            }
        )

    return rows


def _resolve_gt_json(dataset, presets_path):
    presets = _load_json(Path(presets_path))
    preset = presets.get("dataset_presets", {}).get(dataset, {})
    json_path = preset.get("default_predict", {}).get("json_path", "")
    if not json_path:
        return None
    path = Path(json_path)
    if path.exists():
        return path
    preset_relative = Path(presets_path).resolve().parent.parent / json_path
    return preset_relative if preset_relative.exists() else path


def _bbox_iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _roc_points_from_coco(gt_json_path, det_json_path, iou_threshold=0.5):
    gt_data = _load_json(Path(gt_json_path))
    det_data = _load_json(Path(det_json_path))

    images = gt_data.get("images", [])
    annotations = gt_data.get("annotations", [])
    num_images = max(1, len(images))

    gt_by_key = defaultdict(list)
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        key = (int(ann.get("image_id")), int(ann.get("category_id", 1)))
        gt_by_key[key].append({"bbox": [float(v) for v in bbox], "matched": False})

    total_gt = sum(len(v) for v in gt_by_key.values())
    detections = []
    for det in det_data:
        bbox = det.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        detections.append(
            {
                "image_id": int(det.get("image_id")),
                "category_id": int(det.get("category_id", 1)),
                "bbox": [float(v) for v in bbox],
                "score": _safe_float(det.get("score", 0.0)),
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)

    tp = 0
    fp = 0
    points = [(0.0, 0.0, 1.0)]
    pr_points = [(0.0, 1.0, 1.0)]
    for det in detections:
        key = (det["image_id"], det["category_id"])
        best_gt = None
        best_iou = 0.0
        for gt in gt_by_key.get(key, []):
            if gt["matched"]:
                continue
            iou = _bbox_iou_xywh(det["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt is not None and best_iou >= iou_threshold:
            best_gt["matched"] = True
            tp += 1
        else:
            fp += 1

        recall = tp / total_gt if total_gt else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        fppi = fp / num_images
        points.append((fppi, recall, det["score"]))
        pr_points.append((recall, precision, det["score"]))

    return {
        "points": points,
        "pr_points": pr_points,
        "num_images": num_images,
        "num_gt": total_gt,
        "num_det": len(detections),
    }


def select_rows(rows, mode):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["network"])].append(row)

    selected = []
    for _, group in grouped.items():
        mode = _weight_policy_tag(mode)
        if mode == "ap50":
            key = lambda r: (_safe_float(r.get("AP@0.50")), r.get("mtime", 0))
        elif mode == "ap50:95":
            key = lambda r: (_safe_float(r.get("AP@[0.50:0.95]")), r.get("mtime", 0))
        else:
            key = lambda r: r.get("mtime", 0)
        selected.append(max(group, key=key))
    return sorted(selected, key=lambda r: (r["dataset"], r["network"]))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "network", "run_tag", "run", "run_dir"] + METRIC_KEYS
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            for key in METRIC_KEYS:
                out[key] = _format_score(_safe_float(row.get(key)))
            writer.writerow(out)


def _metric_matrix(rows, metric):
    datasets = sorted({r["dataset"] for r in rows})
    networks = sorted({r["network"] for r in rows})
    lookup = {(r["dataset"], r["network"]): _safe_float(r.get(metric)) for r in rows}
    matrix = []
    for network in networks:
        matrix.append([lookup.get((dataset, network), math.nan) for dataset in datasets])
    return datasets, networks, matrix


def plot_heatmaps(rows, out_path, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(metrics)
    cols = 2
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(max(10, cols * 5), max(4, rows_n * 3.8)))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, metric in zip(axes, metrics):
        datasets, networks, matrix = _metric_matrix(rows, metric)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(metric)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_yticks(range(len(networks)))
        ax.set_yticklabels(networks)
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                text = "NA" if math.isnan(value) else f"{value * 100:.1f}"
                color = "white" if not math.isnan(value) and value < 0.55 else "black"
                ax.text(x, y, text, ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("Prediction Metrics by Dataset and Network", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bars(rows, out_path, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in rows})
    networks = sorted({r["network"] for r in rows})
    lookup = {(r["dataset"], r["network"]): r for r in rows}

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(10, len(datasets) * 2.8), 3.3 * len(metrics)))
    axes = axes if isinstance(axes, (list, tuple)) else list(getattr(axes, "flat", [axes]))

    group_width = 0.82
    bar_width = group_width / max(len(networks), 1)
    base_x = list(range(len(datasets)))

    for ax, metric in zip(axes, metrics):
        for idx, network in enumerate(networks):
            xs = [x - group_width / 2 + idx * bar_width + bar_width / 2 for x in base_x]
            ys = [_safe_float(lookup.get((dataset, network), {}).get(metric)) for dataset in datasets]
            ax.bar(xs, [0 if math.isnan(y) else y for y in ys], width=bar_width, label=network)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(base_x)
        ax.set_xticklabels(datasets)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=min(len(networks), 4), fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _metric_plot_name(metric):
    aliases = {
        "AP@0.50": ("map50", "mAP50 / AP@0.50"),
        "AP@[0.50:0.95]": ("map50_95", "mAP50:95 / AP@[0.50:0.95]"),
        "AP@0.75": ("map75", "mAP75 / AP@0.75"),
        "Precision@0.5": ("precision", "Precision @ IoU 0.5"),
        "Recall@0.5": ("recall", "Recall @ IoU 0.5"),
        "F1@0.5": ("f1", "F1 @ IoU 0.5"),
        "AR@100": ("ar100", "AR@100"),
    }
    if metric in aliases:
        return aliases[metric]
    return _slug(metric).lower() or "metric", metric


def _dataset_order(rows):
    present = {r["dataset"] for r in rows}
    preferred = [d for d in ("DAUB", "IRDST", "ITSDT_15K") if d in present]
    preferred.extend(sorted(present - set(preferred)))
    return preferred


def _write_metric_rank_csv(rows, metric, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "network", "run_tag", "run", "run_dir", metric, f"{metric}_percent"]
    ordered = sorted(
        rows,
        key=lambda r: (r["dataset"], -1.0 if math.isnan(_safe_float(r.get(metric))) else -_safe_float(r.get(metric)), r["network"]),
    )
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            value = _safe_float(row.get(metric))
            writer.writerow(
                {
                    "dataset": row.get("dataset", ""),
                    "network": row.get("network", ""),
                    "run_tag": row.get("run_tag", ""),
                    "run": row.get("run", ""),
                    "run_dir": row.get("run_dir", ""),
                    metric: _format_score(value),
                    f"{metric}_percent": "" if math.isnan(value) else f"{value * 100.0:.4f}",
                }
            )


def plot_metric_rankings(rows, out_dir, metrics, select_label="latest", run_tag=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_order = _dataset_order(rows)
    written = []

    for metric in metrics:
        metric_rows = [r for r in rows if not math.isnan(_safe_float(r.get(metric)))]
        if not metric_rows:
            continue

        slug, title_label = _metric_plot_name(metric)
        counts = [len([r for r in metric_rows if r["dataset"] == dataset]) for dataset in dataset_order]
        active = [(dataset, count) for dataset, count in zip(dataset_order, counts) if count > 0]
        if not active:
            continue

        height = max(5.5, sum(max(count * 0.48, 2.5) for _, count in active) + 1.2)
        fig, axes = plt.subplots(
            nrows=len(active),
            ncols=1,
            figsize=(13.5, height),
            sharex=True,
            gridspec_kw={"height_ratios": [max(count, 4) for _, count in active]},
        )
        if len(active) == 1:
            axes = [axes]

        cmap = plt.get_cmap("viridis")
        for ax, (dataset, _count) in zip(axes, active):
            sub = [r for r in metric_rows if r["dataset"] == dataset]
            sub = sorted(sub, key=lambda r: (_safe_float(r.get(metric)), r["network"]))
            names = [r["network"] for r in sub]
            values = [_safe_float(r.get(metric)) * 100.0 for r in sub]
            colors = [cmap(0.25 + 0.65 * min(max(value / 100.0, 0.0), 1.0)) for value in values]
            bars = ax.barh(names, values, color=colors, edgecolor="#263238", linewidth=0.45, height=0.66)

            ax.set_title(f"{dataset}  {title_label} Ranking", loc="left", fontsize=15, fontweight="bold", pad=8)
            ax.set_xlim(0, 103)
            ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.28)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=11)
            ax.tick_params(axis="x", labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#78909c")
            ax.spines["bottom"].set_color("#78909c")
            for bar, value in zip(bars, values):
                x = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2
                if x >= 88:
                    ax.text(x - 1.2, y, f"{value:.1f}", va="center", ha="right", fontsize=10, color="white", fontweight="bold")
                else:
                    ax.text(x + 1.0, y, f"{value:.1f}", va="center", ha="left", fontsize=10, color="#263238", fontweight="bold")

        axes[-1].set_xlabel(f"{title_label} (%)", fontsize=12, fontweight="bold")
        fig.suptitle(f"{select_label.capitalize()} Detection Performance by Dataset ({title_label})", fontsize=18, fontweight="bold", y=0.995)
        source = "Source: selected metrics"
        if run_tag:
            source += f", run tag {run_tag}"
        fig.text(0.01, 0.01, source, fontsize=9, color="#607d8b")
        fig.tight_layout(rect=[0.02, 0.025, 0.995, 0.975])

        png_path = out_dir / f"{slug}_ranked_by_dataset.png"
        pdf_path = out_dir / f"{slug}_ranked_by_dataset.pdf"
        csv_path = out_dir / f"{slug}_ranked_data.csv"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        _write_metric_rank_csv(metric_rows, metric, csv_path)
        written.extend([str(png_path), str(pdf_path), str(csv_path)])

    return written


def write_roc_csv(path, curves):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "network", "run_tag", "run", "fppi", "recall", "score", "num_images", "num_gt", "num_det"],
        )
        writer.writeheader()
        for curve in curves:
            row_base = {
                "dataset": curve["dataset"],
                "network": curve["network"],
                "run_tag": curve.get("run_tag", ""),
                "run": curve["run"],
                "num_images": curve["num_images"],
                "num_gt": curve["num_gt"],
                "num_det": curve["num_det"],
            }
            for fppi, recall, score in curve["points"]:
                writer.writerow(
                    {
                        **row_base,
                        "fppi": _format_score(fppi),
                        "recall": _format_score(recall),
                        "score": _format_score(score),
                    }
                )


def write_pr_csv(path, curves):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "network", "run_tag", "run", "recall", "precision", "score", "num_images", "num_gt", "num_det"],
        )
        writer.writeheader()
        for curve in curves:
            row_base = {
                "dataset": curve["dataset"],
                "network": curve["network"],
                "run_tag": curve.get("run_tag", ""),
                "run": curve["run"],
                "num_images": curve["num_images"],
                "num_gt": curve["num_gt"],
                "num_det": curve["num_det"],
            }
            for recall, precision, score in curve["pr_points"]:
                writer.writerow(
                    {
                        **row_base,
                        "recall": _format_score(recall),
                        "precision": _format_score(precision),
                        "score": _format_score(score),
                    }
                )


def collect_roc_curves(rows, presets_path, iou_threshold, curve_points=101, max_fppi=10.0, verbose=True):
    curves = []
    gt_cache = {}
    iterator = rows
    if tqdm is not None:
        iterator = tqdm(rows, desc="Build PR/FROC", unit="run")
    for idx, row in enumerate(iterator, start=1):
        run_dir = Path(row["run_dir"])
        det_path = run_dir / "eval_results.json"
        if not det_path.exists():
            if verbose:
                print(f"[PR/FROC Skip] {row['dataset']}/{row['network']} missing eval_results.json: {run_dir}")
            continue

        dataset = row["dataset"]
        if verbose:
            size_mb = det_path.stat().st_size / (1024 * 1024)
            print(
                "[PR/FROC %d/%d] %s/%s run=%s eval_results=%.2f MB"
                % (idx, len(rows), dataset, row["network"], row["run"], size_mb),
                flush=True,
            )
        if dataset not in gt_cache:
            gt_cache[dataset] = _resolve_gt_json(dataset, presets_path)
        gt_path = gt_cache[dataset]
        if gt_path is None or not Path(gt_path).exists():
            print(f"[ROC Skip] Missing GT json for dataset {dataset}: {gt_path}")
            continue

        try:
            roc = _roc_points_from_coco(gt_path, det_path, iou_threshold=iou_threshold)
        except Exception as exc:
            print(f"[ROC Skip] {run_dir}: {exc}")
            continue
        roc["raw_num_points"] = len(roc["points"])
        roc["raw_num_pr_points"] = len(roc["pr_points"])
        roc["points"] = _sample_froc_points(roc["points"], curve_points, max_fppi=max_fppi)
        roc["pr_points"] = _sample_pr_points(roc["pr_points"], curve_points)
        if verbose:
            print(
                "[PR/FROC Done] %s/%s detections=%d gt=%d images=%d raw_points=%d sampled_points=%d"
                % (
                    dataset,
                    row["network"],
                    roc["num_det"],
                    roc["num_gt"],
                    roc["num_images"],
                    roc["raw_num_pr_points"],
                    len(roc["pr_points"]),
                ),
                flush=True,
            )
        curves.append({**row, **roc})
    return curves


def plot_pr_curves(curves, out_dir, iou_threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for curve in curves:
        grouped[curve["dataset"]].append(curve)

    for dataset, dataset_curves in grouped.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for curve in sorted(dataset_curves, key=lambda c: c["network"]):
            xs = [p[0] for p in curve["pr_points"]]
            ys = [p[1] for p in curve["pr_points"]]
            ax.step(xs, ys, where="post", linewidth=2, label=curve["network"])

        ax.set_title(f"{dataset} Precision-Recall @ IoU {iou_threshold:g}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.margins(x=0, y=0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"pr_{dataset}_iou{iou_threshold:g}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for curve in sorted(curves, key=lambda c: (c["dataset"], c["network"])):
        xs = [p[0] for p in curve["pr_points"]]
        ys = [p[1] for p in curve["pr_points"]]
        ax.step(xs, ys, where="post", linewidth=1.8, label=f"{curve['dataset']}/{curve['network']}")
    ax.set_title(f"Precision-Recall @ IoU {iou_threshold:g}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.margins(x=0, y=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_all_iou{iou_threshold:g}.png", dpi=180)
    plt.close(fig)


def plot_roc_curves(curves, out_dir, iou_threshold, max_fppi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for curve in curves:
        grouped[curve["dataset"]].append(curve)

    for dataset, dataset_curves in grouped.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for curve in sorted(dataset_curves, key=lambda c: c["network"]):
            xs = [p[0] for p in curve["points"]]
            ys = [p[1] for p in curve["points"]]
            if max_fppi is not None:
                filtered = [(x, y) for x, y in zip(xs, ys) if x <= max_fppi]
                if filtered:
                    xs, ys = zip(*filtered)
                else:
                    xs, ys = [0.0], [0.0]
            ax.plot(xs, ys, linewidth=2, label=curve["network"])

        ax.set_title(f"{dataset} ROC-like FROC @ IoU {iou_threshold:g}")
        ax.set_xlabel("False positives per image")
        ax.set_ylabel("Recall / TPR")
        ax.set_ylim(0, 1)
        if max_fppi is not None:
            ax.set_xlim(0, max_fppi)
        ax.margins(x=0, y=0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"roc_{dataset}_iou{iou_threshold:g}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for curve in sorted(curves, key=lambda c: (c["dataset"], c["network"])):
        xs = [p[0] for p in curve["points"]]
        ys = [p[1] for p in curve["points"]]
        if max_fppi is not None:
            filtered = [(x, y) for x, y in zip(xs, ys) if x <= max_fppi]
            if filtered:
                xs, ys = zip(*filtered)
            else:
                xs, ys = [0.0], [0.0]
        ax.plot(xs, ys, linewidth=1.8, label=f"{curve['dataset']}/{curve['network']}")
    ax.set_title(f"ROC-like FROC @ IoU {iou_threshold:g}")
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Recall / TPR")
    ax.set_ylim(0, 1)
    if max_fppi is not None:
        ax.set_xlim(0, max_fppi)
    ax.margins(x=0, y=0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_all_iou{iou_threshold:g}.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize prediction metrics under result/predict.")
    parser.add_argument("--root", default="result/predict", help="Prediction output root.")
    parser.add_argument("--output-dir", default="", help="Summary output directory.")
    parser.add_argument("--config", default="configs/predict_experiment_config.json", help="Predict config json used for auto run-tag defaults.")
    parser.add_argument(
        "--select",
        choices=("latest", "ap50", "ap50:95", "best-ap50", "best-map"),
        default="latest",
        help="Which run to visualize for each dataset/network pair. Use ap50 or ap50:95; best-ap50/best-map are compatible aliases.",
    )
    parser.add_argument(
        "--metrics",
        default="AP@[0.50:0.95],AP@0.50,AP@0.75,Precision@0.5,Recall@0.5,F1@0.5",
        help="Comma-separated metrics to plot.",
    )
    parser.add_argument("--presets", default="configs/experiment_presets.json", help="Dataset preset json for ROC GT lookup.")
    parser.add_argument("--roc-iou", type=float, default=0.5, help="IoU threshold used for ROC/FROC matching.")
    parser.add_argument("--max-fppi", type=float, default=10.0, help="Max false positives per image shown on ROC plots. Use negative to disable.")
    parser.add_argument("--confidence", type=float, default=None, help="Optional confidence label for the output folder name.")
    parser.add_argument("--nms_iou", "--nms-iou", dest="nms_iou", type=float, default=None, help="Optional NMS IoU label for the output folder name.")
    parser.add_argument("--weight-policy", default="ap50", help="Weight policy label used when --run-tag auto. Canonical values: ap50, ap50:95, loss, last.")
    parser.add_argument("--run-tag", default="", help="Filter prediction runs by tag. Use 'auto' to build wp-*_conf*_nms* from weight policy/config.")
    parser.add_argument("--tag", default="", help="Optional extra label for the output folder name.")
    parser.add_argument("--no-roc", action="store_true", help="Disable ROC/FROC plots.")
    parser.add_argument("--no-pr", action="store_true", help="Disable precision-recall plots.")
    parser.add_argument("--curve-points", type=int, default=101, help="Number of sampled points saved and plotted for PR/ROC curves.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logs while building PR/FROC curves.")
    args = parser.parse_args()

    root = Path(args.root)
    run_tag = resolve_run_tag(args)
    base_out_dir = Path(args.output_dir) if args.output_dir else root / "_summary"
    out_dir = _make_run_output_dir(base_out_dir, args)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    all_rows = collect_runs(root, run_tag=run_tag)
    if not all_rows:
        tag_text = f" with run_tag={run_tag}" if run_tag else ""
        raise SystemExit(f"No eval_metrics.json or result_summary.txt found under {root}{tag_text}")

    selected_rows = select_rows(all_rows, args.select)
    if not args.quiet:
        print(f"[Scan] root={root}")
        print(f"[Scan] run_tag={run_tag or '(none)'}")
        print(f"[Scan] matched runs={len(all_rows)}")
        print(f"[Scan] selected runs={len(selected_rows)} ({args.select})")
        for row in selected_rows:
            det_path = Path(row["run_dir"]) / "eval_results.json"
            det_text = "has eval_results.json" if det_path.exists() else "no eval_results.json"
            print(f"[Selected] {row['dataset']}/{row['network']} run={row['run']} {det_text}")

    write_csv(out_dir / "all_runs_metrics.csv", sorted(all_rows, key=lambda r: (r["dataset"], r["network"], r["run"])))
    write_csv(out_dir / f"{args.select}_metrics.csv", selected_rows)
    ranked_metric_outputs = plot_metric_rankings(
        selected_rows,
        out_dir,
        metrics,
        select_label=args.select,
        run_tag=run_tag,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "output_dir": str(out_dir),
        "run_tag_filter": run_tag,
        "select": args.select,
        "metrics": metrics,
        "roc_iou": args.roc_iou,
        "max_fppi": args.max_fppi,
        "weight_policy_label": args.weight_policy,
        "confidence_label": args.confidence,
        "nms_iou_label": args.nms_iou,
        "tag": args.tag,
        "no_roc": args.no_roc,
        "no_pr": args.no_pr,
        "curve_points": _point_count(args.curve_points),
        "quiet": args.quiet,
        "ranked_metric_outputs": ranked_metric_outputs,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_roc or not args.no_pr:
        max_fppi = None if args.max_fppi is not None and args.max_fppi < 0 else args.max_fppi
        roc_curves = collect_roc_curves(
            selected_rows,
            args.presets,
            args.roc_iou,
            curve_points=args.curve_points,
            max_fppi=max_fppi,
            verbose=not args.quiet,
        )
        if roc_curves:
            if not args.no_roc:
                if not args.quiet:
                    print("[Write] ROC/FROC csv and plots...", flush=True)
                write_roc_csv(out_dir / f"{args.select}_roc_iou{args.roc_iou:g}.csv", roc_curves)
                plot_roc_curves(roc_curves, out_dir, args.roc_iou, max_fppi)
            if not args.no_pr:
                if not args.quiet:
                    print("[Write] PR csv and plots...", flush=True)
                write_pr_csv(out_dir / f"{args.select}_pr_iou{args.roc_iou:g}.csv", roc_curves)
                plot_pr_curves(roc_curves, out_dir, args.roc_iou)
        else:
            print("No eval_results.json-backed runs found for ROC/FROC or PR plots.")

    print(f"Found runs: {len(all_rows)}")
    print(f"Selected runs: {len(selected_rows)} ({args.select})")
    print(f"Saved CSV and figures to: {out_dir}")


if __name__ == "__main__":
    main()
