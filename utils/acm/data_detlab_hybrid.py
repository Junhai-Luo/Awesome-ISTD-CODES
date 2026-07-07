import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image

from utils.utils import cvtColor, preprocess_input
from yoloX.utils.dataloader import YoloDataset as _YoloxYoloDataset


STAGE_ORDER = [
    "D0_old_full",
    "D1_yolox_preprocess",
    "D2_yolox_box_parse",
    "D3_yolox_single_aug",
    "D4_yolox_mosaic_sample",
    "D5_yolox_mosaic_merge",
    "D6_yolox_mixup",
    "D7_yolox_full",
]

STAGE_ALIASES = {
    "d0": "D0_old_full",
    "old": "D0_old_full",
    "old-full": "D0_old_full",
    "old_full": "D0_old_full",
    "d1": "D1_yolox_preprocess",
    "preprocess": "D1_yolox_preprocess",
    "yolox-preprocess": "D1_yolox_preprocess",
    "d2": "D2_yolox_box_parse",
    "box-parse": "D2_yolox_box_parse",
    "box_parse": "D2_yolox_box_parse",
    "yolox-box-parse": "D2_yolox_box_parse",
    "d3": "D3_yolox_single_aug",
    "single-aug": "D3_yolox_single_aug",
    "single_aug": "D3_yolox_single_aug",
    "d4": "D4_yolox_mosaic_sample",
    "mosaic-sample": "D4_yolox_mosaic_sample",
    "mosaic_sample": "D4_yolox_mosaic_sample",
    "d5": "D5_yolox_mosaic_merge",
    "mosaic-merge": "D5_yolox_mosaic_merge",
    "mosaic_merge": "D5_yolox_mosaic_merge",
    "d6": "D6_yolox_mixup",
    "mixup": "D6_yolox_mixup",
    "yolox-mixup": "D6_yolox_mixup",
    "d7": "D7_yolox_full",
    "yolox": "D7_yolox_full",
    "yolox-full": "D7_yolox_full",
    "yolox_full": "D7_yolox_full",
}


def normalize_stage(stage):
    text = str(stage or "D0_old_full").strip()
    if not text:
        return "D0_old_full"
    lower = text.lower()
    if lower in STAGE_ALIASES:
        return STAGE_ALIASES[lower]
    for item in STAGE_ORDER:
        if lower == item.lower():
            return item
    raise ValueError("Unknown ACM hybrid dataset stage '%s'. Available: %s" % (stage, ", ".join(STAGE_ORDER)))


def stage_flags(stage):
    stage = normalize_stage(stage)
    idx = STAGE_ORDER.index(stage)
    return {
        "stage": stage,
        "yolox_preprocess": idx >= 1,
        "yolox_box_parse": idx >= 2,
        "yolox_single_aug": idx >= 3,
        "yolox_mosaic_sample": idx >= 4,
        "yolox_mosaic_merge": idx >= 5,
        "yolox_mixup": idx >= 6,
        "yolox_full": idx >= 7,
    }


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


def _join_tail_candidates(image_root, raw_path):
    if not image_root:
        return []

    image_root = os.path.normpath(image_root)
    raw_parts = _normalize_parts(raw_path)
    candidates = []

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


def _resolve_image_path(raw_path, image_root, txt_dir):
    raw_path = raw_path.strip()
    candidates = []

    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        if image_root:
            candidates.extend(_dedupe_root_candidates(image_root, raw_path))
            candidates.append(os.path.join(image_root, raw_path))
        candidates.append(os.path.join(txt_dir, raw_path))

    candidates.extend(_join_tail_candidates(image_root, raw_path))

    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm

    return os.path.normpath(candidates[0]) if candidates else os.path.normpath(raw_path)


def _parse_detlab_line(line, image_root, txt_dir, cast_int=False):
    parts = line.strip().split()
    if len(parts) == 0:
        return None, np.zeros((0, 5), dtype=np.float32)

    image_path = _resolve_image_path(parts[0], image_root, txt_dir)
    boxes = []
    for item in parts[1:]:
        vals = item.split(",")
        if len(vals) < 4:
            continue
        if cast_int:
            x1, y1, x2, y2 = [float(int(float(v))) for v in vals[:4]]
            cls_id = float(int(float(vals[4]))) if len(vals) >= 5 else 0.0
        else:
            x1, y1, x2, y2 = map(float, vals[:4])
            cls_id = float(vals[4]) if len(vals) >= 5 else 0.0
        boxes.append([x1, y1, x2, y2, cls_id])
    if len(boxes) == 0:
        return image_path, np.zeros((0, 5), dtype=np.float32)
    return image_path, np.array(boxes, dtype=np.float32)


class DetlabHybridTxtDetDataset(Data.Dataset):
    """Old DETLAB ACM Dataset with staged replacements toward yoloX YoloDataset."""

    def __init__(
        self,
        txt_path,
        input_size=512,
        image_root="",
        train=True,
        mosaic=False,
        mixup=False,
        mosaic_prob=0.5,
        mixup_prob=0.5,
        epoch_length=100,
        special_aug_ratio=0.7,
        num_classes=1,
        stage="D0_old_full",
    ):
        self.txt_path = txt_path
        self.input_size = int(input_size)
        self.image_root = image_root
        self.train = train
        self.mosaic = bool(mosaic)
        self.mixup = bool(mixup)
        self.mosaic_prob = float(mosaic_prob)
        self.mixup_prob = float(mixup_prob)
        self.epoch_length = int(epoch_length)
        self.special_aug_ratio = float(special_aug_ratio)
        self.num_classes = int(num_classes)
        self.epoch_now = -1
        self.flags = stage_flags(stage)
        self.stage = self.flags["stage"]
        self.samples = []

        txt_dir = os.path.dirname(os.path.abspath(txt_path))
        with open(txt_path, "r", encoding="utf-8") as f:
            annotation_lines = [line for line in f.readlines() if line.strip()]

        if self.flags["yolox_full"]:
            if image_root:
                os.environ["YOLOX_DATASET_IMG_PATH"] = image_root
            self.yolox_dataset = _YoloxYoloDataset(
                annotation_lines,
                [self.input_size, self.input_size],
                self.num_classes,
                epoch_length=self.epoch_length,
                mosaic=self.mosaic,
                mixup=self.mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
                train=self.train,
                special_aug_ratio=self.special_aug_ratio,
            )
        else:
            self.yolox_dataset = None
            for line in annotation_lines:
                image_path, boxes = _parse_detlab_line(
                    line,
                    image_root,
                    txt_dir,
                    cast_int=self.flags["yolox_box_parse"],
                )
                if image_path is None:
                    continue
                self.samples.append((image_path, boxes))

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        print(
            "[ACM Hybrid Dataset] stage=%s preprocess=%s box_parse=%s single_aug=%s mosaic_sample=%s mosaic_merge=%s mixup=%s yolox_full=%s"
            % (
                self.stage,
                int(self.flags["yolox_preprocess"]),
                int(self.flags["yolox_box_parse"]),
                int(self.flags["yolox_single_aug"]),
                int(self.flags["yolox_mosaic_sample"]),
                int(self.flags["yolox_mosaic_merge"]),
                int(self.flags["yolox_mixup"]),
                int(self.flags["yolox_full"]),
            )
        )

    def __len__(self):
        if self.yolox_dataset is not None:
            return len(self.yolox_dataset)
        return len(self.samples)

    def __getitem__(self, index):
        if self.yolox_dataset is not None:
            self.yolox_dataset.epoch_now = self.epoch_now
            return self.yolox_dataset[index]

        index = index % len(self.samples)
        use_mosaic = (
            self.train
            and self.mosaic
            and len(self.samples) >= 4
            and random.random() < self.mosaic_prob
            and self.epoch_now < self.epoch_length * self.special_aug_ratio
        )

        if use_mosaic:
            indices = random.sample(range(len(self.samples)), 3)
            indices.append(index)
            random.shuffle(indices)
            image, boxes_xyxy = self._mosaic(indices)
            if self.mixup and random.random() < self.mixup_prob:
                mix_index = random.randrange(len(self.samples))
                image_2, boxes_2 = self._load_resized_sample(mix_index, random_aug=self.train)
                if self.flags["yolox_mixup"]:
                    image, boxes_xyxy = self._yolox_mixup(image, boxes_xyxy, image_2, boxes_2)
                else:
                    image, boxes_xyxy = self._mixup(image, boxes_xyxy, image_2, boxes_2)
        else:
            image, boxes_xyxy = self._load_resized_sample(index, random_aug=self.train)

        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)
        if self.flags["yolox_preprocess"]:
            image_t = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
            return image_t, boxes_cxcywh

        image_t = self.transform(image)
        return image_t, boxes_cxcywh

    def _load_raw_sample(self, index):
        image_path, boxes_xyxy = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return image, boxes_xyxy.copy()

    @staticmethod
    def _rand(a=0.0, b=1.0):
        return np.random.rand() * (b - a) + a

    @staticmethod
    def _hsv_augment(image_data, hue=.1, sat=0.7, val=0.4):
        image_data = np.array(image_data, dtype=np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_img, sat_img, val_img = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue_img, lut_hue), cv2.LUT(sat_img, lut_sat), cv2.LUT(val_img, lut_val)))
        return cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    def _load_resized_sample(self, index, random_aug=False):
        image, boxes_xyxy = self._load_raw_sample(index)
        if not random_aug:
            return self._resize_with_letterbox(image, boxes_xyxy, self.input_size)
        if self.flags["yolox_single_aug"]:
            return self._yolox_random_resize_sample(image, boxes_xyxy)
        return self._random_resize_sample(image, boxes_xyxy)

    def _random_resize_sample(self, image, boxes_xyxy, jitter=.3):
        iw, ih = image.size
        w = self.input_size
        h = self.input_size

        new_ar = iw / ih * self._rand(1 - jitter, 1 + jitter) / self._rand(1 - jitter, 1 + jitter)
        scale = self._rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        nw = max(1, nw)
        nh = max(1, nh)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self._rand(0, w - nw))
        dy = int(self._rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self._rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = self._hsv_augment(np.array(image, dtype=np.uint8))

        if len(boxes_xyxy) == 0:
            return image_data, boxes_xyxy.astype(np.float32)

        boxes = boxes_xyxy.copy()
        np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
        if flip:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, boxes.astype(np.float32)

    def _yolox_random_resize_sample(self, image, boxes_xyxy, jitter=.3, hue=.1, sat=0.7, val=0.4):
        image = cvtColor(image)
        iw, ih = image.size
        h = self.input_size
        w = self.input_size

        new_ar = iw / ih * self._rand(1 - jitter, 1 + jitter) / self._rand(1 - jitter, 1 + jitter)
        scale = self._rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        nw = max(1, nw)
        nh = max(1, nh)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self._rand(0, w - nw))
        dy = int(self._rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self._rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_img, sat_img, val_img = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue_img, lut_hue), cv2.LUT(sat_img, lut_sat), cv2.LUT(val_img, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        boxes = boxes_xyxy.copy()
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
            if flip:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > w] = w
            boxes[:, 3][boxes[:, 3] > h] = h
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, boxes.astype(np.float32)

    def _mosaic(self, indices):
        if self.flags["yolox_mosaic_sample"]:
            return self._yolox_mosaic(indices)
        return self._old_mosaic(indices)

    def _old_mosaic(self, indices):
        h = self.input_size
        w = self.input_size
        min_offset_x = random.uniform(0.3, 0.7)
        min_offset_y = random.uniform(0.3, 0.7)
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        image_datas = []
        box_datas = []
        for mosaic_index, sample_index in enumerate(indices):
            image, boxes = self._load_raw_sample(sample_index)
            iw, ih = image.size

            if random.random() < 0.5 and len(boxes) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = boxes.copy()
                boxes[:, [0, 2]] = iw - boxes[:, [2, 0]]

            new_ar = iw / ih * random.uniform(0.7, 1.3) / random.uniform(0.7, 1.3)
            scale = random.uniform(0.4, 1.0)
            image_data, boxes = self._resize_mosaic_piece(
                image,
                boxes,
                iw,
                ih,
                w,
                h,
                new_ar,
                scale,
                cutx,
                cuty,
                mosaic_index,
                shuffle_boxes=False,
            )
            image_datas.append(image_data)
            box_datas.append(boxes)

        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = self._hsv_augment(new_image)
        boxes = self._merge_mosaic_boxes(box_datas, cutx, cuty)
        return new_image, boxes

    def _yolox_mosaic(self, indices, jitter=.3, hue=.1, sat=0.7, val=0.4):
        h = self.input_size
        w = self.input_size
        min_offset_x = self._rand(0.3, 0.7)
        min_offset_y = self._rand(0.3, 0.7)
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        image_datas = []
        box_datas = []
        for mosaic_index, sample_index in enumerate(indices):
            image, boxes = self._load_raw_sample(sample_index)
            image = cvtColor(image)
            iw, ih = image.size

            flip = self._rand() < .5
            if flip and len(boxes) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = boxes.copy()
                boxes[:, [0, 2]] = iw - boxes[:, [2, 0]]

            new_ar = iw / ih * self._rand(1 - jitter, 1 + jitter) / self._rand(1 - jitter, 1 + jitter)
            scale = self._rand(.4, 1)
            image_data, boxes = self._resize_mosaic_piece(
                image,
                boxes,
                iw,
                ih,
                w,
                h,
                new_ar,
                scale,
                cutx,
                cuty,
                mosaic_index,
                shuffle_boxes=True,
            )
            image_datas.append(image_data)
            box_datas.append(boxes)

        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_img, sat_img, val_img = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        new_image = cv2.merge((cv2.LUT(hue_img, lut_hue), cv2.LUT(sat_img, lut_sat), cv2.LUT(val_img, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        if self.flags["yolox_mosaic_merge"]:
            boxes = self._yolox_merge_bboxes(box_datas, cutx, cuty)
        else:
            boxes = self._merge_mosaic_boxes(box_datas, cutx, cuty)
        return new_image, boxes

    @staticmethod
    def _resize_mosaic_piece(image, boxes, iw, ih, w, h, new_ar, scale, cutx, cuty, mosaic_index, shuffle_boxes=False):
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        nw = max(1, nw)
        nh = max(1, nh)
        image = image.resize((nw, nh), Image.BICUBIC)

        if mosaic_index == 0:
            dx = cutx - nw
            dy = cuty - nh
        elif mosaic_index == 1:
            dx = cutx - nw
            dy = cuty
        elif mosaic_index == 2:
            dx = cutx
            dy = cuty
        else:
            dx = cutx
            dy = cuty - nh

        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, dtype=np.uint8)

        if len(boxes) == 0:
            return image_data, np.zeros((0, 5), dtype=np.float32)

        boxes = boxes.copy()
        if shuffle_boxes:
            np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(bw > 1, bh > 1)]
        return image_data, boxes.astype(np.float32)

    @staticmethod
    def _merge_mosaic_boxes(box_datas, cutx, cuty):
        merged = []
        for i, boxes in enumerate(box_datas):
            for box in boxes:
                x1, y1, x2, y2, cls_id = box.tolist()

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                elif i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                elif i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                else:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                merged.append([x1, y1, x2, y2, cls_id])

        if len(merged) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(merged, dtype=np.float32)

    @staticmethod
    def _yolox_merge_bboxes(box_datas, cutx, cuty):
        merge_bbox = []
        for i in range(len(box_datas)):
            for box in box_datas[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)

        if len(merge_bbox) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(merge_bbox, dtype=np.float32)

    @staticmethod
    def _mixup(image_1, boxes_1, image_2, boxes_2):
        image = np.array(image_1, dtype=np.float32) * 0.5 + np.array(image_2, dtype=np.float32) * 0.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        if len(boxes_1) == 0:
            boxes = boxes_2
        elif len(boxes_2) == 0:
            boxes = boxes_1
        else:
            boxes = np.concatenate([boxes_1, boxes_2], axis=0)
        return image, boxes.astype(np.float32)

    @staticmethod
    def _yolox_mixup(image_1, boxes_1, image_2, boxes_2):
        image = np.array(image_1, dtype=np.float32) * 0.5 + np.array(image_2, dtype=np.float32) * 0.5
        if len(boxes_1) == 0:
            boxes = boxes_2
        elif len(boxes_2) == 0:
            boxes = boxes_1
        else:
            boxes = np.concatenate([boxes_1, boxes_2], axis=0)
        return image, boxes.astype(np.float32)

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
    for img, box in batch:
        images.append(img)
        box = np.asarray(box, dtype=np.float32)
        if box.size == 0:
            box = np.zeros((0, 5), dtype=np.float32)
        boxes.append(torch.from_numpy(box).float())

    if images and torch.is_tensor(images[0]):
        images = torch.stack(images, dim=0)
    else:
        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    return images, boxes


DetlabTxtDetDataset = DetlabHybridTxtDetDataset
