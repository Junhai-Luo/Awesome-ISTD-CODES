import argparse
import colorsys
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from nets.factory import build_network
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression
from utils.coco_compat import ensure_coco_dataset_compat
from utils.dataloader_for_sequence import _history_frame_paths

VIS_BOX_COLOR = (255, 0, 0)
VIS_BOX_THICKNESS = 2


def is_acm_network(name):
    return str(name).lower().startswith("acm")


def is_acm_saliency_network(name):
    lower = str(name).lower()
    return lower == "acm_unet_saliency"


def is_dna_network(name):
    lower = str(name).lower()
    return lower in ("dnanet", "dnanet_saliency")


def is_dna_saliency_network(name):
    lower = str(name).lower()
    return lower == "dnanet_saliency"


def _dna_state_key_candidates(key):
    key = str(key)
    keys = [key]
    if key.startswith("module."):
        keys.append(key[len("module."):])
    for item in list(keys):
        if not item.startswith("backbone."):
            keys.append("backbone." + item)
    return keys


def is_alc_network(name):
    lower = str(name).lower()
    return lower in ("alcnet", "alcnet_saliency")


def is_alc_saliency_network(name):
    lower = str(name).lower()
    return lower == "alcnet_saliency"


def is_uiu_network(name):
    lower = str(name).lower()
    return lower in ("uiunet", "uiunet_saliency")


def is_uiu_saliency_network(name):
    lower = str(name).lower()
    return lower == "uiunet_saliency"


def is_sctrans_network(name):
    lower = str(name).lower()
    return lower in ("sctransnet", "sctransnet_det")


def normalize_output_mode(mode):
    m = str(mode or "all").lower()
    if m in ("vis", "vis_only"):
        return "vis_only"
    if m in ("json", "eval", "eval_only"):
        return "eval_only"
    return "all"


def _prepare_draw_box(box, canvas_size):
    top, left, bottom, right = [float(v) for v in box]
    if not np.isfinite([top, left, bottom, right]).all():
        return None

    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))

    width, height = canvas_size
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    left = int(np.clip(np.floor(left), 0, max_x))
    right = int(np.clip(np.ceil(right), 0, max_x))
    top = int(np.clip(np.floor(top), 0, max_y))
    bottom = int(np.clip(np.ceil(bottom), 0, max_y))

    if right < left or bottom < top:
        return None
    return top, left, bottom, right


def _draw_detection_box(draw, box, color, thickness):
    top, left, bottom, right = box
    draw.rectangle([left, top, right, bottom], outline=color, width=max(1, int(thickness)))


def _xywh_iou(box_a, box_b):
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax2 = ax + max(aw, 0.0)
    ay2 = ay + max(ah, 0.0)
    bx2 = bx + max(bw, 0.0)
    by2 = by + max(bh, 0.0)

    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return 0.0 if union <= 0 else inter / union


def _format_iou_tag(value):
    return ("%.2f" % float(value)).replace(".", "p")


def save_failed_prediction_report(coco, det_results, output_dir, dataset_img_path, json_dir, iou_threshold, missing_images=None):
    missing_images = missing_images or []
    preds_by_image = {}
    for pred in det_results:
        preds_by_image.setdefault(int(pred["image_id"]), []).append(pred)

    rows = []
    failed_paths = []
    for image_id in coco.getImgIds():
        anns = [
            ann
            for ann in coco.imgToAnns.get(image_id, [])
            if int(ann.get("iscrowd", 0)) == 0 and float(ann.get("area", 0.0)) >= 0.0
        ]
        if not anns:
            continue

        image_info = coco.loadImgs(image_id)[0]
        file_name = str(image_info.get("file_name", ""))
        image_path = resolve_dataset_image_path(file_name, dataset_img_path, json_dir)
        preds = preds_by_image.get(int(image_id), [])
        best_iou = 0.0
        best_score = 0.0
        matched = False

        for ann in anns:
            ann_cat = int(ann.get("category_id", -1))
            ann_box = ann.get("bbox", [0, 0, 0, 0])
            for pred in preds:
                if int(pred.get("category_id", -2)) != ann_cat:
                    continue
                iou = _xywh_iou(ann_box, pred.get("bbox", [0, 0, 0, 0]))
                score = float(pred.get("score", 0.0))
                if iou > best_iou:
                    best_iou = iou
                    best_score = score
                if iou >= float(iou_threshold):
                    matched = True

        if matched:
            continue

        reason = "no_detection" if len(preds) == 0 else "no_iou_match"
        rows.append(
            {
                "image_id": int(image_id),
                "file_name": file_name,
                "image_path": image_path,
                "num_gt": len(anns),
                "num_pred": len(preds),
                "best_iou": best_iou,
                "best_score": best_score,
                "reason": reason,
            }
        )
        failed_paths.append(image_path)

    tag = _format_iou_tag(iou_threshold)
    txt_path = os.path.join(output_dir, "failed_prediction_images_iou%s.txt" % tag)
    csv_path = os.path.join(output_dir, "failed_prediction_images_iou%s.csv" % tag)

    with open(txt_path, "w", encoding="utf-8") as f:
        for path in failed_paths:
            f.write(path + "\n")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image_id,file_name,image_path,num_gt,num_pred,best_iou,best_score,reason\n")
        for row in rows:
            f.write(
                '%d,"%s","%s",%d,%d,%.6f,%.6f,%s\n'
                % (
                    row["image_id"],
                    str(row["file_name"]).replace('"', '""'),
                    str(row["image_path"]).replace('"', '""'),
                    row["num_gt"],
                    row["num_pred"],
                    row["best_iou"],
                    row["best_score"],
                    row["reason"],
                )
            )

    if missing_images:
        missing_path = os.path.join(output_dir, "missing_images.txt")
        with open(missing_path, "w", encoding="utf-8") as f:
            for item in missing_images:
                f.write("%s\t%s\n" % (item.get("file_name", ""), item.get("image_path", "")))

    summary = {
        "iou_threshold": float(iou_threshold),
        "num_failed_images": len(rows),
        "num_missing_images": len(missing_images),
        "txt_path": txt_path,
        "csv_path": csv_path,
    }
    summary_path = os.path.join(output_dir, "failed_prediction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        "Saved failed prediction image list:",
        txt_path,
        "(failed=%d, missing=%d)" % (len(rows), len(missing_images)),
    )
    return summary


def _visual_detection_indices(scores, vis_confidence=None, vis_max_boxes=0):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return []

    if vis_confidence is None:
        keep = np.arange(scores.size)
    else:
        keep = np.where(scores >= float(vis_confidence))[0]

    if keep.size == 0:
        return []

    order = keep[np.argsort(scores[keep])[::-1]]
    if vis_max_boxes and int(vis_max_boxes) > 0:
        order = order[: int(vis_max_boxes)]
    return [int(i) for i in order]


def _normalize_path_parts(path_value):
    path_value = str(path_value).replace("\\", "/")
    return [part for part in path_value.split("/") if part and part not in (".",)]


def _dedupe_root_candidates(root_path, raw_path):
    if not root_path:
        return []

    root_path = os.path.normpath(root_path)
    root_parts = _normalize_path_parts(root_path)
    raw_parts = _normalize_path_parts(raw_path)
    candidates = []

    max_overlap = min(len(root_parts), len(raw_parts))
    for overlap in range(max_overlap, 0, -1):
        if root_parts[-overlap:] != raw_parts[:overlap]:
            continue
        tail = raw_parts[overlap:]
        candidates.append(os.path.join(root_path, *tail) if tail else root_path)
        break
    return candidates


def _join_tail_candidates(root_path, raw_path):
    if not root_path:
        return []

    root_path = os.path.normpath(root_path)
    raw_parts = _normalize_path_parts(raw_path)
    candidates = []
    if raw_parts:
        for keep in range(1, min(len(raw_parts), 6) + 1):
            candidates.append(os.path.join(root_path, *raw_parts[-keep:]))

    ordered = []
    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return ordered


def resolve_dataset_image_path(file_name, dataset_img_path, json_dir):
    file_name = str(file_name).strip()
    candidates = []

    if os.path.isabs(file_name):
        candidates.append(file_name)
    else:
        if dataset_img_path:
            candidates.extend(_dedupe_root_candidates(dataset_img_path, file_name))
            candidates.append(os.path.join(dataset_img_path, file_name))
            candidates.extend(_join_tail_candidates(dataset_img_path, file_name))
        if json_dir:
            candidates.append(os.path.join(json_dir, file_name))
        candidates.append(file_name)

    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm

    return os.path.normpath(candidates[0]) if candidates else os.path.normpath(file_name)


class JsonVideoPredictor:
    def __init__(
        self,
        model_path,
        classes_path,
        input_shape=(512, 512),
        confidence=0.5,
        nms_iou=0.3,
        letterbox_image=False,
        num_frame=5,
        cuda=True,
        network_name="sstnet",
        vis_confidence=0.3,
        vis_max_boxes=0,
    ):
        self.model_path = model_path
        self.classes_path = classes_path
        self.input_shape = list(input_shape)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.num_frame = num_frame
        self.cuda = cuda and torch.cuda.is_available()
        self.network_name = network_name
        self.vis_confidence = vis_confidence
        self.vis_max_boxes = int(vis_max_boxes or 0)

        self.class_names, self.num_classes = get_classes(self.classes_path)
        hsv_tuples = [(x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.net = build_network(self.network_name, num_classes=self.num_classes, num_frame=self.num_frame)
        device = torch.device("cuda" if self.cuda else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

    def history_frames(self, image_path):
        return [Image.open(path) for path in _history_frame_paths(image_path, self.num_frame)]

    def _preprocess_sequence(self, images, use_input_shape):
        image_shape = np.array(np.shape(images[0])[0:2])
        images = [cvtColor(img) for img in images]
        canvas = images[-1].copy()

        image_data = [resize_image(img, (use_input_shape[1], use_input_shape[0]), self.letterbox_image) for img in images]
        image_data = [np.transpose(preprocess_input(np.array(img, dtype="float32")), (2, 0, 1)) for img in image_data]
        image_data = np.stack(image_data, axis=1)
        return image_data, canvas, image_shape

    def _draw_detections(self, canvas, detections, clsid2catid=None, image_id=None, input_shape=None):
        use_input_shape = self.input_shape if input_shape is None else [int(input_shape[0]), int(input_shape[1])]
        if detections is None:
            return canvas, []

        top_label = np.array(detections[:, 6], dtype="int32")
        top_conf = detections[:, 4] * detections[:, 5]
        top_boxes = detections[:, :4]

        results = []
        if clsid2catid is not None and image_id is not None:
            for i, c in enumerate(top_label):
                top, left, bottom, right = top_boxes[i]
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": clsid2catid[int(c)],
                        "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                        "score": float(top_conf[i]),
                    }
                )

        try:
            font = ImageFont.truetype(
                font="model_data/simhei.ttf",
                size=np.floor(3e-2 * canvas.size[1] + 0.5).astype("int32"),
            )
        except Exception:
            font = ImageFont.load_default()
        thickness = int(max((canvas.size[0] + canvas.size[1]) // np.mean(use_input_shape), 1))

        for i in _visual_detection_indices(top_conf, self.vis_confidence, self.vis_max_boxes):
            draw_box = _prepare_draw_box(top_boxes[i], canvas.size)
            if draw_box is None:
                continue
            draw = ImageDraw.Draw(canvas)
            _draw_detection_box(draw, draw_box, VIS_BOX_COLOR, VIS_BOX_THICKNESS)
            del draw
        return canvas, results

    def detect_and_draw(self, images, clsid2catid=None, image_id=None, input_shape=None):
        if input_shape is None:
            use_input_shape = self.input_shape
        else:
            use_input_shape = [int(input_shape[0]), int(input_shape[1])]

        image_data, canvas, image_shape = self._preprocess_sequence(images, use_input_shape)
        image_data = np.expand_dims(image_data, 0)

        with torch.no_grad():
            tensor = torch.from_numpy(image_data)
            if self.cuda:
                tensor = tensor.cuda()
            outputs = self.net(tensor)
            outputs = decode_outputs(outputs, use_input_shape)
            outputs = non_max_suppression(
                outputs,
                self.num_classes,
                use_input_shape,
                image_shape,
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
            )
            if outputs[0] is None:
                return canvas, []

        return self._draw_detections(canvas, outputs[0], clsid2catid=clsid2catid, image_id=image_id, input_shape=use_input_shape)

    def detect_batch(self, batch_images, image_ids, clsid2catid=None, input_shape=None):
        if input_shape is None:
            use_input_shape = self.input_shape
        else:
            use_input_shape = [int(input_shape[0]), int(input_shape[1])]

        prepared = [self._preprocess_sequence(images, use_input_shape) for images in batch_images]
        tensors = np.stack([item[0] for item in prepared], axis=0)
        canvases = [item[1] for item in prepared]
        image_shapes = [item[2] for item in prepared]

        with torch.no_grad():
            tensor = torch.from_numpy(tensors)
            if self.cuda:
                tensor = tensor.cuda()
            decoded = decode_outputs(self.net(tensor), use_input_shape)

        batch_results = []
        for idx, canvas in enumerate(canvases):
            outputs = non_max_suppression(
                decoded[idx : idx + 1],
                self.num_classes,
                use_input_shape,
                image_shapes[idx],
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
            )
            detections = outputs[0] if outputs and outputs[0] is not None else None
            batch_results.append(
                self._draw_detections(
                    canvas,
                    detections,
                    clsid2catid=clsid2catid,
                    image_id=image_ids[idx],
                    input_shape=use_input_shape,
                )
            )
        return batch_results


class AcmJsonPredictor:
    def __init__(
        self,
        model_path,
        input_shape=(512, 512),
        confidence=0.3,
        nms_iou=0.45,
        cuda=True,
        backbone_mode="FPN",
        det_mode="feature",
        fuse_mode="AsymBi",
        blocks_per_layer=4,
        num_classes=1,
        class_names=None,
        vis_confidence=0.3,
        vis_max_boxes=0,
    ):
        from nets.acm.detection import ASKCResNetFPNDet, ASKCResUNetDet, ASKCResUNetSaliencyDet

        self.model_path = model_path
        self.input_shape = list(input_shape)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.cuda = cuda and torch.cuda.is_available()
        self.backbone_mode = backbone_mode
        self.det_mode = str(det_mode or "feature").lower()
        self.fuse_mode = fuse_mode
        self.blocks_per_layer = int(blocks_per_layer)
        self.num_classes = int(num_classes)
        self.vis_confidence = vis_confidence
        self.vis_max_boxes = int(vis_max_boxes or 0)

        if class_names and len(class_names) >= self.num_classes:
            self.class_names = class_names[: self.num_classes]
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]

        hsv_tuples = [(x / max(self.num_classes, 1), 1.0, 1.0) for x in range(max(self.num_classes, 1))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        layer_blocks = [self.blocks_per_layer] * 3
        channels = [8, 16, 32, 64]
        if self.det_mode == "saliency":
            if self.backbone_mode != "UNet":
                raise ValueError("ACM saliency detection requires backbone_mode=UNet.")
            self.net = ASKCResUNetSaliencyDet(layer_blocks, channels, self.fuse_mode, self.num_classes)
        elif self.backbone_mode == "FPN":
            self.net = ASKCResNetFPNDet(layer_blocks, channels, self.fuse_mode, self.num_classes)
        else:
            self.net = ASKCResUNetDet(layer_blocks, channels, self.fuse_mode, self.num_classes)

        device = torch.device("cuda" if self.cuda else "cpu")
        ckpt = torch.load(self.model_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.net.load_state_dict(ckpt["state_dict"], strict=True)
        elif isinstance(ckpt, dict):
            self.net.load_state_dict(ckpt, strict=True)
        else:
            raise ValueError("Unsupported ACM checkpoint format")

        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

    def _preprocess_image(self, image):
        iw, ih = image.size
        w = int(self.input_shape[1])
        h = int(self.input_shape[0])
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(resized, ((w - nw) // 2, (h - nh) // 2))

        arr = np.array(new_image, dtype="float32")
        arr = preprocess_input(arr)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr

    def detect_and_draw(self, image, clsid2catid=None, image_id=None, input_shape=None):
        from utils.acm.det_bbox import decode_outputs as acm_decode_outputs
        from utils.acm.det_bbox import non_max_suppression as acm_non_max_suppression

        if input_shape is None:
            use_input_shape = self.input_shape
        else:
            use_input_shape = [int(input_shape[0]), int(input_shape[1])]

        image = cvtColor(image)
        canvas = image.copy()
        image_shape = (image.height, image.width)
        image_data = self._preprocess_image(image)

        with torch.no_grad():
            tensor = torch.from_numpy(image_data)
            if self.cuda:
                tensor = tensor.cuda()
            outputs = self.net(tensor)
            outputs = acm_decode_outputs(outputs, (use_input_shape[0], use_input_shape[1]))
            detections = acm_non_max_suppression(
                outputs,
                num_classes=self.num_classes,
                input_shape=(use_input_shape[0], use_input_shape[1]),
                image_shape=image_shape,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
                letterbox_image=True,
            )[0]

            if detections is None:
                return canvas, []

            top_label = np.array(detections[:, 6], dtype="int32")
            top_conf = detections[:, 4] * detections[:, 5]
            top_boxes = detections[:, :4]

        results = []
        if clsid2catid is not None and image_id is not None:
            for i, c in enumerate(top_label):
                top, left, bottom, right = top_boxes[i]
                cat_id = clsid2catid[int(c)] if int(c) < len(clsid2catid) else int(c) + 1
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cat_id),
                        "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                        "score": float(top_conf[i]),
                    }
                )

        try:
            font = ImageFont.truetype(
                font="model_data/simhei.ttf",
                size=np.floor(3e-2 * canvas.size[1] + 0.5).astype("int32"),
            )
        except Exception:
            font = ImageFont.load_default()

        thickness = int(max((canvas.size[0] + canvas.size[1]) // np.mean(use_input_shape), 1))
        for i in _visual_detection_indices(top_conf, self.vis_confidence, self.vis_max_boxes):
            draw_box = _prepare_draw_box(top_boxes[i], canvas.size)
            if draw_box is None:
                continue
            draw = ImageDraw.Draw(canvas)
            _draw_detection_box(draw, draw_box, VIS_BOX_COLOR, VIS_BOX_THICKNESS)
            del draw

        return canvas, results


class AlcJsonPredictor:
    def __init__(self, model_path, input_shape=(512, 512), confidence=0.3, nms_iou=0.45, cuda=True, num_classes=1, class_names=None, fuse_mode="AsymBi", blocks_per_layer=4, det_mode="feature", vis_confidence=0.3, vis_max_boxes=0):
        from nets.alc.detection import ALCNetDet, ALCNetSaliencyDet

        self.model_path = model_path
        self.input_shape = list(input_shape)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.cuda = cuda and torch.cuda.is_available()
        self.num_classes = int(num_classes)
        self.fuse_mode = fuse_mode
        self.blocks_per_layer = int(blocks_per_layer)
        self.det_mode = str(det_mode or "feature")
        self.vis_confidence = vis_confidence
        self.vis_max_boxes = int(vis_max_boxes or 0)

        if class_names and len(class_names) >= self.num_classes:
            self.class_names = class_names[: self.num_classes]
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]

        hsv_tuples = [(x / max(self.num_classes, 1), 1.0, 1.0) for x in range(max(self.num_classes, 1))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        layer_blocks = [self.blocks_per_layer] * 3
        channels = [8, 16, 32, 64]
        model_cls = ALCNetSaliencyDet if self.det_mode == "saliency" else ALCNetDet
        self.net = model_cls(in_channels=3, layers=layer_blocks, channels=channels, fuse_mode=self.fuse_mode, num_classes=self.num_classes)

        device = torch.device("cuda" if self.cuda else "cpu")
        ckpt = torch.load(self.model_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported ALCNet checkpoint format")

        current = self.net.state_dict()
        filtered = {k: v for k, v in state.items() if (k in current and current[k].shape == v.shape)}
        if len(filtered) == 0:
            stripped = {
                k[len("module.") :]: v
                for k, v in state.items()
                if isinstance(k, str) and k.startswith("module.")
            }
            filtered = {k: v for k, v in stripped.items() if (k in current and current[k].shape == v.shape)}
        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        skipped = len(state) - len(filtered) if isinstance(state, dict) else 0
        print(
            "ALCNet loaded keys:",
            len(filtered),
            "missing:",
            len(missing),
            "unexpected:",
            len(unexpected),
            "skipped:",
            skipped,
        )
        if len(filtered) == 0:
            raise RuntimeError("ALCNet checkpoint did not match the model architecture.")

        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

    def _preprocess_image(self, image):
        iw, ih = image.size
        w = int(self.input_shape[1])
        h = int(self.input_shape[0])
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(resized, ((w - nw) // 2, (h - nh) // 2))

        arr = np.array(new_image, dtype="float32")
        arr = preprocess_input(arr)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr

    def detect_and_draw(self, image, clsid2catid=None, image_id=None, input_shape=None):
        from utils.acm.det_bbox import decode_outputs as alc_decode_outputs
        from utils.acm.det_bbox import non_max_suppression as alc_non_max_suppression

        if input_shape is None:
            use_input_shape = self.input_shape
        else:
            use_input_shape = [int(input_shape[0]), int(input_shape[1])]

        image = cvtColor(image)
        canvas = image.copy()
        image_shape = (image.height, image.width)
        image_data = self._preprocess_image(image)

        with torch.no_grad():
            tensor = torch.from_numpy(image_data)
            if self.cuda:
                tensor = tensor.cuda()
            outputs = self.net(tensor)
            outputs = alc_decode_outputs(outputs, (use_input_shape[0], use_input_shape[1]))
            detections = alc_non_max_suppression(outputs, num_classes=self.num_classes, input_shape=(use_input_shape[0], use_input_shape[1]), image_shape=image_shape, conf_thres=self.confidence, nms_thres=self.nms_iou, letterbox_image=True)[0]
            if detections is None:
                return canvas, []
            top_label = np.array(detections[:, 6], dtype="int32")
            top_conf = detections[:, 4] * detections[:, 5]
            top_boxes = detections[:, :4]

        results = []
        if clsid2catid is not None and image_id is not None:
            for i, c in enumerate(top_label):
                top, left, bottom, right = top_boxes[i]
                cat_id = clsid2catid[int(c)] if int(c) < len(clsid2catid) else int(c) + 1
                results.append({"image_id": int(image_id), "category_id": int(cat_id), "bbox": [float(left), float(top), float(right - left), float(bottom - top)], "score": float(top_conf[i])})

        try:
            font = ImageFont.truetype(font="model_data/simhei.ttf", size=np.floor(3e-2 * canvas.size[1] + 0.5).astype("int32"))
        except Exception:
            font = ImageFont.load_default()

        thickness = int(max((canvas.size[0] + canvas.size[1]) // np.mean(use_input_shape), 1))
        for i in _visual_detection_indices(top_conf, self.vis_confidence, self.vis_max_boxes):
            draw_box = _prepare_draw_box(top_boxes[i], canvas.size)
            if draw_box is None:
                continue
            draw = ImageDraw.Draw(canvas)
            _draw_detection_box(draw, draw_box, VIS_BOX_COLOR, VIS_BOX_THICKNESS)
            del draw

        return canvas, results


class SCTransNetJsonPredictor:
    def __init__(
        self,
        model_path,
        input_shape=(512, 512),
        confidence=0.3,
        nms_iou=0.45,
        cuda=True,
        num_classes=1,
        class_names=None,
        vis_confidence=0.3,
        vis_max_boxes=0,
    ):
        from nets.sctransnet.Config import get_SCTrans_config
        from nets.sctransnet.SCTransNetDet import SCTransNetDet

        self.model_path = model_path
        self.input_shape = list(input_shape)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.cuda = cuda and torch.cuda.is_available()
        self.num_classes = int(num_classes)
        self.vis_confidence = vis_confidence
        self.vis_max_boxes = int(vis_max_boxes or 0)

        if class_names and len(class_names) >= self.num_classes:
            self.class_names = class_names[: self.num_classes]
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]

        self.net = SCTransNetDet(
            get_SCTrans_config(),
            n_channels=3,
            num_classes=self.num_classes,
            img_size=int(self.input_shape[0]),
        )

        device = torch.device("cuda" if self.cuda else "cpu")
        ckpt = torch.load(self.model_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported SCTransNet checkpoint format")

        current = self.net.state_dict()
        filtered = {k: v for k, v in state.items() if (k in current and current[k].shape == v.shape)}
        if len(filtered) == 0:
            stripped = {
                k[len("module.") :]: v
                for k, v in state.items()
                if isinstance(k, str) and k.startswith("module.")
            }
            filtered = {k: v for k, v in stripped.items() if (k in current and current[k].shape == v.shape)}
        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        skipped = len(state) - len(filtered) if isinstance(state, dict) else 0
        print(
            "SCTransNet loaded keys:",
            len(filtered),
            "missing:",
            len(missing),
            "unexpected:",
            len(unexpected),
            "skipped:",
            skipped,
        )
        if len(filtered) == 0:
            raise RuntimeError("SCTransNet checkpoint did not match the model architecture.")

        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

    def _preprocess_image(self, image):
        iw, ih = image.size
        w = int(self.input_shape[1])
        h = int(self.input_shape[0])
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(resized, ((w - nw) // 2, (h - nh) // 2))

        arr = np.array(new_image, dtype="float32")
        arr = preprocess_input(arr)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr

    def detect_and_draw(self, image, clsid2catid=None, image_id=None, input_shape=None):
        from utils.acm.det_bbox import decode_outputs as det_decode_outputs
        from utils.acm.det_bbox import non_max_suppression as det_non_max_suppression

        if input_shape is None:
            use_input_shape = self.input_shape
        else:
            use_input_shape = [int(input_shape[0]), int(input_shape[1])]

        image = cvtColor(image)
        canvas = image.copy()
        image_shape = (image.height, image.width)
        image_data = self._preprocess_image(image)

        with torch.no_grad():
            tensor = torch.from_numpy(image_data)
            if self.cuda:
                tensor = tensor.cuda()
            outputs = self.net(tensor)
            outputs = det_decode_outputs(outputs, (use_input_shape[0], use_input_shape[1]))
            detections = det_non_max_suppression(
                outputs,
                num_classes=self.num_classes,
                input_shape=(use_input_shape[0], use_input_shape[1]),
                image_shape=image_shape,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
                letterbox_image=True,
            )[0]
            if detections is None:
                return canvas, []
            top_label = np.array(detections[:, 6], dtype="int32")
            top_conf = detections[:, 4] * detections[:, 5]
            top_boxes = detections[:, :4]

        results = []
        if clsid2catid is not None and image_id is not None:
            for i, c in enumerate(top_label):
                top, left, bottom, right = top_boxes[i]
                cat_id = clsid2catid[int(c)] if int(c) < len(clsid2catid) else int(c) + 1
                results.append({"image_id": int(image_id), "category_id": int(cat_id), "bbox": [float(left), float(top), float(right - left), float(bottom - top)], "score": float(top_conf[i])})

        for i in _visual_detection_indices(top_conf, self.vis_confidence, self.vis_max_boxes):
            draw_box = _prepare_draw_box(top_boxes[i], canvas.size)
            if draw_box is None:
                continue
            draw = ImageDraw.Draw(canvas)
            _draw_detection_box(draw, draw_box, VIS_BOX_COLOR, VIS_BOX_THICKNESS)
            del draw

        return canvas, results


class DnaJsonPredictor:
    def __init__(
        self,
        model_path,
        input_shape=(512, 512),
        confidence=0.3,
        nms_iou=0.45,
        cuda=True,
        num_classes=1,
        class_names=None,
        channel_size="three",
        backbone="resnet_18",
        det_mode="feature",
        vis_confidence=0.3,
        vis_max_boxes=0,
    ):
        from nets.dna.model_DNANet_det import DNANetDet, DNANetSaliencyDet

        self.model_path = model_path
        self.input_shape = list(input_shape)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.cuda = cuda and torch.cuda.is_available()
        self.num_classes = int(num_classes)
        self.channel_size = channel_size
        self.backbone = backbone
        self.det_mode = det_mode
        self.vis_confidence = vis_confidence
        self.vis_max_boxes = int(vis_max_boxes or 0)

        if class_names and len(class_names) >= self.num_classes:
            self.class_names = class_names[: self.num_classes]
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]

        hsv_tuples = [(x / max(self.num_classes, 1), 1.0, 1.0) for x in range(max(self.num_classes, 1))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        model_cls = DNANetSaliencyDet if self.det_mode == "saliency" else DNANetDet
        self.net = model_cls(
            input_channels=3,
            num_classes=self.num_classes,
            channel_size=self.channel_size,
            backbone=self.backbone,
        )

        device = torch.device("cuda" if self.cuda else "cpu")
        ckpt = torch.load(self.model_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported DNANet checkpoint format")

        current = self.net.state_dict()
        filtered = {}
        for key, value in state.items():
            for candidate in _dna_state_key_candidates(key):
                if candidate in current and current[candidate].shape == value.shape:
                    filtered[candidate] = value
                    break
        self.net.load_state_dict(filtered, strict=False)

        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

    def _preprocess_image(self, image):
        iw, ih = image.size
        w = int(self.input_shape[1])
        h = int(self.input_shape[0])
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(resized, ((w - nw) // 2, (h - nh) // 2))

        arr = np.array(new_image, dtype="float32")
        arr = preprocess_input(arr)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr

    def detect_and_draw(self, image, clsid2catid=None, image_id=None, input_shape=None):
        from utils.dna.det_bbox import decode_outputs as dna_decode_outputs
        from utils.dna.det_bbox import non_max_suppression as dna_non_max_suppression

        if input_shape is None:
            use_input_shape = self.input_shape
        else:
            use_input_shape = [int(input_shape[0]), int(input_shape[1])]

        image = cvtColor(image)
        canvas = image.copy()
        image_shape = (image.height, image.width)

        image_data = self._preprocess_image(image)
        with torch.no_grad():
            tensor = torch.from_numpy(image_data)
            if self.cuda:
                tensor = tensor.cuda()
            outputs = self.net(tensor)
            outputs = dna_decode_outputs(outputs, (use_input_shape[0], use_input_shape[1]))
            detections = dna_non_max_suppression(
                outputs,
                num_classes=self.num_classes,
                input_shape=(use_input_shape[0], use_input_shape[1]),
                image_shape=image_shape,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
                letterbox_image=True,
            )[0]

            if detections is None:
                return canvas, []

            top_label = np.array(detections[:, 6], dtype="int32")
            top_conf = detections[:, 4] * detections[:, 5]
            top_boxes = detections[:, :4]

        results = []
        if clsid2catid is not None and image_id is not None:
            for i, c in enumerate(top_label):
                top, left, bottom, right = top_boxes[i]
                cat_id = clsid2catid[int(c)] if int(c) < len(clsid2catid) else int(c) + 1
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cat_id),
                        "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                        "score": float(top_conf[i]),
                    }
                )

        try:
            font = ImageFont.truetype(
                font="model_data/simhei.ttf",
                size=np.floor(3e-2 * canvas.size[1] + 0.5).astype("int32"),
            )
        except Exception:
            font = ImageFont.load_default()

        thickness = int(max((canvas.size[0] + canvas.size[1]) // np.mean(use_input_shape), 1))
        for i in _visual_detection_indices(top_conf, self.vis_confidence, self.vis_max_boxes):
            draw_box = _prepare_draw_box(top_boxes[i], canvas.size)
            if draw_box is None:
                continue
            draw = ImageDraw.Draw(canvas)
            _draw_detection_box(draw, draw_box, VIS_BOX_COLOR, VIS_BOX_THICKNESS)
            del draw

        return canvas, results


class UiuJsonPredictor:
    def __init__(
        self,
        model_path,
        input_shape=(512, 512),
        confidence=0.3,
        nms_iou=0.45,
        cuda=True,
        num_classes=1,
        class_names=None,
        fuse_mode="AsymBi",
        det_mode="feature",
        vis_confidence=0.3,
        vis_max_boxes=0,
    ):
        from nets.uiu.detection import UIUNETDet, UIUNETSaliencyDet

        self.model_path = model_path
        self.input_shape = list(input_shape)
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.cuda = cuda and torch.cuda.is_available()
        self.num_classes = int(num_classes)
        self.fuse_mode = fuse_mode
        self.det_mode = str(det_mode or "feature")
        self.vis_confidence = vis_confidence
        self.vis_max_boxes = int(vis_max_boxes or 0)

        if class_names and len(class_names) >= self.num_classes:
            self.class_names = class_names[: self.num_classes]
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]

        hsv_tuples = [(x / max(self.num_classes, 1), 1.0, 1.0) for x in range(max(self.num_classes, 1))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        model_cls = UIUNETSaliencyDet if self.det_mode == "saliency" else UIUNETDet
        self.net = model_cls(in_ch=3, num_classes=self.num_classes, fuse_mode=self.fuse_mode)
        device = torch.device("cuda" if self.cuda else "cpu")
        ckpt = torch.load(self.model_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported UIUNet checkpoint format")

        current = self.net.state_dict()
        filtered = {}
        for key, value in state.items():
            for candidate in self._state_key_candidates(key):
                if candidate in current and current[candidate].shape == value.shape:
                    filtered[candidate] = value
                    break
        self.net.load_state_dict(filtered, strict=False)

        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()

    @staticmethod
    def _state_key_candidates(key):
        key = str(key)
        keys = [key]
        if key.startswith("module."):
            keys.append(key[len("module."):])
        for item in list(keys):
            if not item.startswith("backbone."):
                keys.append("backbone." + item)
        return keys

    def _preprocess_image(self, image):
        iw, ih = image.size
        w = int(self.input_shape[1])
        h = int(self.input_shape[0])
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(resized, ((w - nw) // 2, (h - nh) // 2))

        arr = np.array(new_image, dtype="float32")
        arr = preprocess_input(arr)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr

    def detect_and_draw(self, image, clsid2catid=None, image_id=None, input_shape=None):
        from utils.uiu.det_bbox import decode_outputs as uiu_decode_outputs
        from utils.uiu.det_bbox import non_max_suppression as uiu_non_max_suppression

        if input_shape is None:
            use_input_shape = self.input_shape
        else:
            use_input_shape = [int(input_shape[0]), int(input_shape[1])]

        image = cvtColor(image)
        canvas = image.copy()
        image_shape = (image.height, image.width)

        image_data = self._preprocess_image(image)
        with torch.no_grad():
            tensor = torch.from_numpy(image_data)
            if self.cuda:
                tensor = tensor.cuda()
            outputs = self.net(tensor)
            outputs = uiu_decode_outputs(outputs, (use_input_shape[0], use_input_shape[1]))
            detections = uiu_non_max_suppression(
                outputs,
                num_classes=self.num_classes,
                input_shape=(use_input_shape[0], use_input_shape[1]),
                image_shape=image_shape,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
                letterbox_image=True,
            )[0]

            if detections is None:
                return canvas, []

            top_label = np.array(detections[:, 6], dtype="int32")
            top_conf = detections[:, 4] * detections[:, 5]
            top_boxes = detections[:, :4]

        results = []
        if clsid2catid is not None and image_id is not None:
            for i, c in enumerate(top_label):
                top, left, bottom, right = top_boxes[i]
                cat_id = clsid2catid[int(c)] if int(c) < len(clsid2catid) else int(c) + 1
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cat_id),
                        "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                        "score": float(top_conf[i]),
                    }
                )

        try:
            font = ImageFont.truetype(
                font="model_data/simhei.ttf",
                size=np.floor(3e-2 * canvas.size[1] + 0.5).astype("int32"),
            )
        except Exception:
            font = ImageFont.load_default()

        thickness = int(max((canvas.size[0] + canvas.size[1]) // np.mean(use_input_shape), 1))
        for i in _visual_detection_indices(top_conf, self.vis_confidence, self.vis_max_boxes):
            draw_box = _prepare_draw_box(top_boxes[i], canvas.size)
            if draw_box is None:
                continue
            draw = ImageDraw.Draw(canvas)
            _draw_detection_box(draw, draw_box, VIS_BOX_COLOR, VIS_BOX_THICKNESS)
            del draw

        return canvas, results

def parse_args():
    parser = argparse.ArgumentParser("Batch prediction images from COCO test.json")
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--dataset_img_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--classes_path", default="model_data/classes.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--auto_input_from_json", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms_iou", type=float, default=0.65)
    parser.add_argument("--vis_confidence", "--vis-confidence", dest="vis_confidence", type=float, default=0.3)
    parser.add_argument("--vis_max_boxes", "--vis-max-boxes", dest="vis_max_boxes", type=int, default=0)
    parser.add_argument("--num_frame", type=int, default=5)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--network_name", type=str, default="sstnet")
    parser.add_argument("--letterbox", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output_mode",
        type=str,
        default="all",
        choices=["vis_only", "eval_only", "all", "vis", "json", "eval"],
    )
    parser.add_argument("--save_eval_json", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--save_failed_images", action="store_true")
    parser.add_argument("--failed_iou", "--failed-iou", dest="failed_iou", type=float, default=0.5)

    # ACM specific
    parser.add_argument("--backbone_mode", type=str, default="FPN", choices=["FPN", "UNet"])
    parser.add_argument("--acm_det_mode", type=str, default="auto", choices=["auto", "feature", "saliency"])
    parser.add_argument("--fuse_mode", type=str, default="AsymBi", choices=["BiLocal", "AsymBi", "BiGlobal"])
    parser.add_argument("--blocks_per_layer", type=int, default=4)
    parser.add_argument("--alc_det_mode", type=str, default="auto", choices=["auto", "feature", "saliency"])

    # DNANet specific
    parser.add_argument("--dna_channel_size", type=str, default="three")
    parser.add_argument("--dna_backbone", type=str, default="resnet_18")
    parser.add_argument("--dna_det_mode", type=str, default="auto", choices=["auto", "feature", "saliency"])

    # UIUNet specific
    parser.add_argument("--uiu_fuse_mode", type=str, default="AsymBi")
    parser.add_argument("--uiu_det_mode", type=str, default="auto", choices=["auto", "feature", "saliency"])
    parser.add_argument("--num_classes", type=int, default=1)

    return parser.parse_args()


def build_predictor(args, coco):
    if is_uiu_network(args.network_name):
        categories = coco.dataset.get("categories", [])
        class_names = [str(c.get("name", c.get("id", i + 1))) for i, c in enumerate(categories)]
        num_classes = args.num_classes if args.num_classes > 0 else max(1, len(categories))
        det_mode = args.uiu_det_mode
        if det_mode == "auto":
            det_mode = "saliency" if is_uiu_saliency_network(args.network_name) else "feature"
        return UiuJsonPredictor(
            model_path=args.model_path,
            input_shape=(args.input_size, args.input_size),
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            cuda=not args.cpu,
            num_classes=num_classes,
            class_names=class_names,
            fuse_mode=args.uiu_fuse_mode,
            det_mode=det_mode,
            vis_confidence=args.vis_confidence,
            vis_max_boxes=args.vis_max_boxes,
        )

    if is_dna_network(args.network_name):
        categories = coco.dataset.get("categories", [])
        class_names = [str(c.get("name", c.get("id", i + 1))) for i, c in enumerate(categories)]
        num_classes = args.num_classes if args.num_classes > 0 else max(1, len(categories))
        det_mode = args.dna_det_mode
        if det_mode == "auto":
            det_mode = "saliency" if is_dna_saliency_network(args.network_name) else "feature"
        return DnaJsonPredictor(
            model_path=args.model_path,
            input_shape=(args.input_size, args.input_size),
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            cuda=not args.cpu,
            num_classes=num_classes,
            class_names=class_names,
            channel_size=args.dna_channel_size,
            backbone=args.dna_backbone,
            det_mode=det_mode,
            vis_confidence=args.vis_confidence,
            vis_max_boxes=args.vis_max_boxes,
        )

    if is_alc_network(args.network_name):
        categories = coco.dataset.get("categories", [])
        class_names = [str(c.get("name", c.get("id", i + 1))) for i, c in enumerate(categories)]
        num_classes = args.num_classes if args.num_classes > 0 else max(1, len(categories))
        det_mode = args.alc_det_mode
        if det_mode == "auto":
            det_mode = "saliency" if is_alc_saliency_network(args.network_name) else "feature"
        return AlcJsonPredictor(
            model_path=args.model_path,
            input_shape=(args.input_size, args.input_size),
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            cuda=not args.cpu,
            num_classes=num_classes,
            class_names=class_names,
            fuse_mode=args.fuse_mode,
            blocks_per_layer=args.blocks_per_layer,
            det_mode=det_mode,
            vis_confidence=args.vis_confidence,
            vis_max_boxes=args.vis_max_boxes,
        )

    if is_acm_network(args.network_name):
        categories = coco.dataset.get("categories", [])
        class_names = [str(c.get("name", c.get("id", i + 1))) for i, c in enumerate(categories)]
        num_classes = args.num_classes if args.num_classes > 0 else max(1, len(categories))
        det_mode = args.acm_det_mode
        if det_mode == "auto":
            det_mode = "saliency" if is_acm_saliency_network(args.network_name) else "feature"
        return AcmJsonPredictor(
            model_path=args.model_path,
            input_shape=(args.input_size, args.input_size),
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            cuda=not args.cpu,
            backbone_mode=args.backbone_mode,
            det_mode=det_mode,
            fuse_mode=args.fuse_mode,
            blocks_per_layer=args.blocks_per_layer,
            num_classes=num_classes,
            class_names=class_names,
            vis_confidence=args.vis_confidence,
            vis_max_boxes=args.vis_max_boxes,
        )

    if is_sctrans_network(args.network_name):
        categories = coco.dataset.get("categories", [])
        class_names = [str(c.get("name", c.get("id", i + 1))) for i, c in enumerate(categories)]
        num_classes = args.num_classes if args.num_classes > 0 else max(1, len(categories))
        return SCTransNetJsonPredictor(
            model_path=args.model_path,
            input_shape=(args.input_size, args.input_size),
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            cuda=not args.cpu,
            num_classes=num_classes,
            class_names=class_names,
            vis_confidence=args.vis_confidence,
            vis_max_boxes=args.vis_max_boxes,
        )

    return JsonVideoPredictor(
        model_path=args.model_path,
        classes_path=args.classes_path,
        input_shape=(args.input_size, args.input_size),
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        letterbox_image=args.letterbox,
        num_frame=args.num_frame,
        cuda=not args.cpu,
        network_name=args.network_name,
        vis_confidence=args.vis_confidence,
        vis_max_boxes=args.vis_max_boxes,
    )


def main():
    args = parse_args()
    args.output_mode = normalize_output_mode(args.output_mode)
    os.makedirs(args.output_dir, exist_ok=True)

    coco = ensure_coco_dataset_compat(COCO(args.json_path))
    predictor = build_predictor(args, coco)

    image_ids = coco.getImgIds()
    clsid2catid = coco.getCatIds()
    det_results = []
    missing_images = []
    json_dir = os.path.dirname(os.path.abspath(args.json_path))
    is_single_frame_network = (
        is_acm_network(args.network_name)
        or is_alc_network(args.network_name)
        or is_dna_network(args.network_name)
        or is_uiu_network(args.network_name)
        or is_sctrans_network(args.network_name)
    )
    video_batch_size = max(1, int(args.batch_size))

    if (not is_single_frame_network) and video_batch_size > 1 and not args.auto_input_from_json:
        with tqdm(total=len(image_ids)) as pbar:
            for start in range(0, len(image_ids), video_batch_size):
                batch_ids = []
                batch_files = []
                batch_histories = []

                for image_id in image_ids[start : start + video_batch_size]:
                    image_info = coco.loadImgs(image_id)[0]
                    file_name = image_info["file_name"]
                    image_path = resolve_dataset_image_path(file_name, args.dataset_img_path, json_dir)
                    if not os.path.exists(image_path):
                        print(f"[Skip] Missing image: {image_path}")
                        missing_images.append({"image_id": int(image_id), "file_name": file_name, "image_path": image_path})
                        pbar.update(1)
                        continue
                    batch_ids.append(image_id)
                    batch_files.append(file_name)
                    batch_histories.append(predictor.history_frames(image_path))

                if not batch_ids:
                    continue

                batch_outputs = predictor.detect_batch(batch_histories, batch_ids, clsid2catid=clsid2catid)
                for file_name, (vis, img_results) in zip(batch_files, batch_outputs):
                    det_results.extend(img_results)
                    if args.output_mode != "eval_only":
                        save_path = os.path.join(args.output_dir, file_name)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        vis.save(save_path)
                pbar.update(len(batch_ids))
    else:
        if (not is_single_frame_network) and video_batch_size > 1 and args.auto_input_from_json:
            print("[Batch] auto_input_from_json is enabled; falling back to batch_size=1 for video prediction.")

        for image_id in tqdm(image_ids):
            image_info = coco.loadImgs(image_id)[0]
            file_name = image_info["file_name"]
            image_path = resolve_dataset_image_path(file_name, args.dataset_img_path, json_dir)
            if not os.path.exists(image_path):
                print(f"[Skip] Missing image: {image_path}")
                missing_images.append({"image_id": int(image_id), "file_name": file_name, "image_path": image_path})
                continue

            if is_single_frame_network:
                image = Image.open(image_path)
                if args.auto_input_from_json:
                    auto_h = int(image_info.get("height", args.input_size))
                    auto_w = int(image_info.get("width", args.input_size))
                    vis, img_results = predictor.detect_and_draw(
                        image,
                        clsid2catid=clsid2catid,
                        image_id=image_id,
                        input_shape=(auto_h, auto_w),
                    )
                else:
                    vis, img_results = predictor.detect_and_draw(image, clsid2catid=clsid2catid, image_id=image_id)
            else:
                history_images = predictor.history_frames(image_path)
                if args.auto_input_from_json:
                    auto_h = int(image_info.get("height", predictor.input_shape[0]))
                    auto_w = int(image_info.get("width", predictor.input_shape[1]))
                    vis, img_results = predictor.detect_and_draw(
                        history_images,
                        clsid2catid=clsid2catid,
                        image_id=image_id,
                        input_shape=(auto_h, auto_w),
                    )
                else:
                    vis, img_results = predictor.detect_and_draw(history_images, clsid2catid=clsid2catid, image_id=image_id)

            det_results.extend(img_results)

            if args.output_mode != "eval_only":
                save_path = os.path.join(args.output_dir, file_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                vis.save(save_path)

    need_eval_files = args.output_mode in ("eval_only", "all") or args.save_eval_json or args.run_eval
    if need_eval_files:
        eval_json_path = os.path.join(args.output_dir, "eval_results.json")
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(det_results, f)
        print(f"Saved detection json: {eval_json_path}")

    if args.run_eval:
        if len(det_results) == 0:
            print("No detections found, skip COCO eval.")
        else:
            coco_dt = coco.loadRes(os.path.join(args.output_dir, "eval_results.json"))
            coco_eval = COCOeval(coco, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            precisions = coco_eval.eval["precision"]
            recalls = coco_eval.eval["recall"]
            precision_50 = precisions[0, :, 0, 0, -1]
            recall_50 = recalls[0, 0, 0, -1]
            if recall_50 > 0:
                p_mean = float(np.mean(precision_50[: int(recall_50 * 100)]))
            else:
                p_mean = 0.0
            f1 = 0.0 if (p_mean + recall_50) == 0 else float(2 * recall_50 * p_mean / (p_mean + recall_50))

            metrics = {
                "Precision@0.5": p_mean,
                "Recall@0.5": float(recall_50),
                "F1@0.5": f1,
                "stats": {
                    "AP@[0.50:0.95]": float(coco_eval.stats[0]),
                    "AP@0.50": float(coco_eval.stats[1]),
                    "AP@0.75": float(coco_eval.stats[2]),
                    "AP_small": float(coco_eval.stats[3]),
                    "AP_medium": float(coco_eval.stats[4]),
                    "AP_large": float(coco_eval.stats[5]),
                    "AR@1": float(coco_eval.stats[6]),
                    "AR@10": float(coco_eval.stats[7]),
                    "AR@100": float(coco_eval.stats[8]),
                    "AR_small": float(coco_eval.stats[9]),
                    "AR_medium": float(coco_eval.stats[10]),
                    "AR_large": float(coco_eval.stats[11]),
                },
            }

            metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            with open(os.path.join(args.output_dir, "result_precision50.txt"), "w", encoding="utf-8") as f:
                f.write("\t".join([str(float(p)) for p in precision_50]))

            with open(os.path.join(args.output_dir, "result_summary.txt"), "w", encoding="utf-8") as f:
                f.write("Precision: %.4f, Recall: %.4f, F1: %.4f\n" % (p_mean, recall_50, f1))
                for k, v in metrics["stats"].items():
                    f.write(f"{k}: {v:.6f}\n")

            import matplotlib.pyplot as plt

            plt.figure(1)
            plt.title("PR Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.xlim(0, 100)
            plt.ylim(0, 1.05)
            plt.plot(precision_50)
            plt.savefig(os.path.join(args.output_dir, "p-r.png"))
            plt.close()
            print(f"Saved eval artifacts under: {args.output_dir}")

    if args.save_failed_images and (args.run_eval or need_eval_files):
        save_failed_prediction_report(
            coco,
            det_results,
            args.output_dir,
            args.dataset_img_path,
            json_dir,
            args.failed_iou,
            missing_images=missing_images,
        )

    if args.output_mode == "vis_only":
        print(f"Done. Visualized predictions saved to: {args.output_dir}")
    elif args.output_mode == "eval_only":
        print(f"Done. Eval artifacts saved to: {args.output_dir}")
    else:
        print(f"Done. Visualizations and eval artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()








