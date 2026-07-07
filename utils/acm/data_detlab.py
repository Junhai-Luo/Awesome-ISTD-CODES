import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image


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

    return os.path.normpath(candidates[0])


def _box_coord_type():
    value = (
        os.environ.get("DETLAB_BOX_COORD_TYPE")
        or os.environ.get("DATA_DETLAB_BOX_COORD_TYPE")
        or os.environ.get("ACM_BOX_COORD_TYPE")
        or os.environ.get("ALC_BOX_COORD_TYPE")
        or os.environ.get("UIU_BOX_COORD_TYPE")
        or "float"
    )
    value = str(value).strip().lower()
    return "int" if value in ("int", "integer") else "float"


def _box_geometry_dtype():
    value = (
        os.environ.get("DETLAB_BOX_GEOMETRY_DTYPE")
        or os.environ.get("DATA_DETLAB_BOX_GEOMETRY_DTYPE")
        or os.environ.get("ACM_BOX_GEOMETRY_DTYPE")
        or os.environ.get("ALC_BOX_GEOMETRY_DTYPE")
        or os.environ.get("UIU_BOX_GEOMETRY_DTYPE")
        or "float"
    )
    value = str(value).strip().lower()
    return "int" if value in ("int", "integer") else "float"


def _parse_detlab_line(line, image_root, txt_dir):
    parts = line.strip().split()
    if len(parts) == 0:
        return None, np.zeros((0, 5), dtype=np.float32)

    image_path = _resolve_image_path(parts[0], image_root, txt_dir)
    coord_type = _box_coord_type()
    boxes = []
    for item in parts[1:]:
        vals = item.split(',')
        if len(vals) < 4:
            continue
        if coord_type == "int":
            x1, y1, x2, y2 = [float(int(float(v))) for v in vals[:4]]
            cls_id = float(int(float(vals[4]))) if len(vals) >= 5 else 0.0
        else:
            x1, y1, x2, y2 = map(float, vals[:4])
            cls_id = float(vals[4]) if len(vals) >= 5 else 0.0
        boxes.append([x1, y1, x2, y2, cls_id])
    if len(boxes) == 0:
        return image_path, np.zeros((0, 5), dtype=np.float32)
    box_dtype = np.int64 if _box_geometry_dtype() == "int" else np.float32
    return image_path, np.array(boxes, dtype=box_dtype)


class DetlabTxtDetDataset(Data.Dataset):
    def __init__(
        self,
        txt_path,
        input_size=512,
        image_root='',
        train=True,
        mosaic=False,
        mixup=False,
        mosaic_prob=0.5,
        mixup_prob=0.5,
        epoch_length=100,
        special_aug_ratio=0.7,
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
        self.epoch_now = -1
        self.samples = []

        txt_dir = os.path.dirname(os.path.abspath(txt_path))
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                image_path, boxes = _parse_detlab_line(line, image_root, txt_dir)
                if image_path is None:
                    continue
                self.samples.append((image_path, boxes))

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
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
                image, boxes_xyxy = self._mixup(image, boxes_xyxy, image_2, boxes_2)
        else:
            image, boxes_xyxy = self._load_resized_sample(index, random_aug=self.train)

        image_t = self.transform(image)
        boxes_cxcywh = self._xyxy_to_cxcywh(boxes_xyxy)
        return image_t, boxes_cxcywh

    def _load_raw_sample(self, index):
        image_path, boxes_xyxy = self.samples[index]
        image = Image.open(image_path).convert('RGB')
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
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
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

    def _flip_left_right(self, image, boxes_xyxy):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if len(boxes_xyxy) > 0:
            boxes_xyxy = boxes_xyxy.copy()
            w = self.input_size
            x1 = boxes_xyxy[:, 0].copy()
            x2 = boxes_xyxy[:, 2].copy()
            boxes_xyxy[:, 0] = w - x2
            boxes_xyxy[:, 2] = w - x1
        return image, boxes_xyxy

    def _mosaic(self, indices):
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

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_datas.append(np.array(new_image, dtype=np.uint8))

            if len(boxes) == 0:
                box_datas.append(np.zeros((0, 5), dtype=np.float32))
                continue

            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > w] = w
            boxes[:, 3][boxes[:, 3] > h] = h
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(bw > 1, bh > 1)]
            box_datas.append(boxes.astype(np.float32))

        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = self._hsv_augment(new_image)
        boxes = self._merge_mosaic_boxes(box_datas, cutx, cuty)
        return new_image, boxes

    def _merge_mosaic_boxes(self, box_datas, cutx, cuty):
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
    def _xyxy_to_cxcywh(boxes_xyxy):
        if len(boxes_xyxy) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = boxes_xyxy.astype(np.float32, copy=True)
        boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
        boxes[:, 0:2] = boxes[:, 0:2] + boxes[:, 2:4] / 2.0
        return boxes

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
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
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
        boxes.append(torch.from_numpy(box).float())
    images = torch.stack(images, dim=0)
    return images, boxes
