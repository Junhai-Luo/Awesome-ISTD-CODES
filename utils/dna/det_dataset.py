import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def build_image_id(image_path):
    normalized = str(image_path).replace("\\", "/").rstrip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:]).rsplit(".", 1)[0]
    return os.path.splitext(os.path.basename(normalized))[0]


def _normalize_parts(path_value):
    path_value = str(path_value).replace("\\", "/")
    return [part for part in path_value.split("/") if part and part not in (".",)]


def _dedupe_root_candidates(image_root, raw_path):
    if not image_root:
        return []

    image_root = os.path.normpath(image_root)
    root_parts = _normalize_parts(image_root)
    raw_parts = _normalize_parts(raw_path)
    candidates = []

    max_overlap = min(len(root_parts), len(raw_parts))
    for overlap in range(max_overlap, 0, -1):
        if root_parts[-overlap:] != raw_parts[:overlap]:
            continue
        tail = raw_parts[overlap:]
        candidates.append(os.path.join(image_root, *tail) if tail else image_root)
        break
    return candidates


def _join_tail_candidates(image_root, raw_path, source_image_root):
    if not image_root:
        return []

    image_root = os.path.normpath(image_root)
    raw_parts = _normalize_parts(raw_path)
    candidates = []

    if source_image_root:
        src_parts = _normalize_parts(source_image_root)
        if src_parts:
            for idx in range(len(raw_parts)):
                if raw_parts[idx : idx + len(src_parts)] == src_parts:
                    tail = raw_parts[idx + len(src_parts) :]
                    if tail:
                        candidates.append(os.path.join(image_root, *tail))
                    break

    if raw_parts:
        for keep in range(1, min(len(raw_parts), 6) + 1):
            candidates.append(os.path.join(image_root, *raw_parts[-keep:]))

    ordered = []
    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return ordered


def _resolve_image_path(raw_path, image_root, txt_dir, suffix, source_image_root=""):
    raw_path = raw_path.strip()
    candidates = []

    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        if image_root:
            candidates.extend(_dedupe_root_candidates(image_root, raw_path))
            candidates.append(os.path.join(image_root, raw_path))
        candidates.append(os.path.join(txt_dir, raw_path))

    if suffix and os.path.splitext(raw_path)[1] == "":
        for candidate in list(candidates):
            candidates.append(candidate + suffix)

    for candidate in _join_tail_candidates(image_root, raw_path, source_image_root):
        candidates.append(candidate)
        if suffix and os.path.splitext(candidate)[1] == "":
            candidates.append(candidate + suffix)

    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm
    return os.path.normpath(candidates[0])


def parse_det_line(line, image_root, txt_dir, suffix=".png", source_image_root=""):
    parts = line.strip().split()
    if len(parts) == 0:
        return None, np.zeros((0, 5), dtype=np.float32)

    image_path = _resolve_image_path(parts[0], image_root, txt_dir, suffix, source_image_root=source_image_root)
    boxes = []
    for item in parts[1:]:
        vals = item.split(",")
        if len(vals) < 4:
            continue
        x1, y1, x2, y2 = map(float, vals[:4])
        cls_id = float(vals[4]) if len(vals) >= 5 else 0.0
        boxes.append([x1, y1, x2, y2, cls_id])
    if len(boxes) == 0:
        return image_path, np.zeros((0, 5), dtype=np.float32)
    return image_path, np.array(boxes, dtype=np.float32)


class DetTxtDataset(data.Dataset):
    def __init__(self, txt_path, input_size=512, image_root="", train=True, suffix=".png", source_image_root=""):
        self.txt_path = txt_path
        self.input_size = int(input_size)
        self.image_root = image_root
        self.source_image_root = source_image_root
        self.train = train
        self.suffix = suffix
        self.samples = []
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        txt_dir = os.path.dirname(os.path.abspath(txt_path))
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                image_path, boxes = parse_det_line(line, image_root, txt_dir, suffix=suffix, source_image_root=source_image_root)
                if image_path is None:
                    continue
                self.samples.append((image_path, boxes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, boxes_xyxy = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        original_size = (image.height, image.width)
        original_boxes_xyxy = boxes_xyxy.copy().astype(np.float32)
        image, boxes_xyxy = self._resize_with_letterbox(image, boxes_xyxy, self.input_size)

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes_xyxy) > 0:
                w = self.input_size
                x1 = boxes_xyxy[:, 0].copy()
                x2 = boxes_xyxy[:, 2].copy()
                boxes_xyxy[:, 0] = w - x2
                boxes_xyxy[:, 2] = w - x1

        image_t = self.transform(image)
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)
        meta = {
            "image_id": build_image_id(image_path),
            "image_path": image_path,
            "original_shape": original_size,
            "gt_boxes_xyxy": original_boxes_xyxy,
        }
        return image_t, boxes_cxcywh, meta

    @staticmethod
    def _xyxy_to_cxcywh(boxes_xyxy):
        if len(boxes_xyxy) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = boxes_xyxy.copy()
        boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
        boxes[:, 0:2] = boxes[:, 0:2] + boxes[:, 2:4] / 2.0
        return boxes.astype(np.float32)

    @staticmethod
    def _resize_with_letterbox(image, boxes_xyxy, input_size):
        iw, ih = image.size
        w = input_size
        h = input_size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(resized, (dx, dy))

        if len(boxes_xyxy) == 0:
            return new_image, boxes_xyxy.astype(np.float32)

        boxes = boxes_xyxy.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(bw > 1, bh > 1)]
        return new_image, boxes.astype(np.float32)


def det_dataset_collate(batch):
    images = []
    boxes = []
    metas = []
    for img, box, meta in batch:
        images.append(img)
        boxes.append(torch.from_numpy(_ensure_box_array(box)).float())
        metas.append(meta)
    images = torch.stack(images, dim=0)
    return images, boxes, metas


def _ensure_box_array(box):
    box = np.asarray(box, dtype=np.float32)
    if box.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if box.ndim == 1:
        box = box.reshape(-1, 5)
    return box.astype(np.float32, copy=False)
