import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def _get_dataset_root_prefix():
    return str(os.environ.get("DATASET_ROOT_PREFIX", "")).strip()


def _normalize_parts(path_value):
    path_value = str(path_value).replace("\\", "/")
    return [part for part in path_value.split("/") if part and part not in (".",)]


def _dedupe_root_candidates(root_prefix, raw_path):
    if not root_prefix:
        return []

    root_prefix = os.path.normpath(root_prefix)
    root_parts = _normalize_parts(root_prefix)
    raw_parts = _normalize_parts(raw_path)
    candidates = []

    max_overlap = min(len(root_parts), len(raw_parts))
    for overlap in range(max_overlap, 0, -1):
        if root_parts[-overlap:] != raw_parts[:overlap]:
            continue
        tail = raw_parts[overlap:]
        candidates.append(os.path.join(root_prefix, *tail) if tail else root_prefix)
        break
    return candidates


def _join_tail_candidates(root_prefix, raw_path):
    if not root_prefix:
        return []

    root_prefix = os.path.normpath(root_prefix)
    raw_parts = _normalize_parts(raw_path)
    candidates = []
    if raw_parts:
        for keep in range(1, min(len(raw_parts), 6) + 1):
            candidates.append(os.path.join(root_prefix, *raw_parts[-keep:]))

    ordered = []
    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return ordered


def _numeric_frame_path_variants(path_value):
    frame_dir = os.path.dirname(path_value)
    base_name = os.path.basename(path_value)
    stem, ext = os.path.splitext(base_name)
    if not stem.isdigit() or not ext:
        return []

    frame_id = int(stem)
    widths = [len(stem), 3, 4, 5]
    variants = []
    seen = set()
    plain_variant = os.path.join(frame_dir, f"{frame_id}{ext}")
    variants.append(os.path.normpath(plain_variant))
    seen.add(os.path.normpath(plain_variant))
    for width in widths:
        variant = os.path.join(frame_dir, f"{frame_id:0{width}d}{ext}")
        norm = os.path.normpath(variant)
        if norm in seen:
            continue
        seen.add(norm)
        variants.append(norm)
    return variants


def _resolve_sequence_path(raw_path, txt_dir=""):
    raw_path = str(raw_path).strip()
    root_prefix = _get_dataset_root_prefix()
    candidates = []

    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        if root_prefix:
            candidates.extend(_dedupe_root_candidates(root_prefix, raw_path))
            candidates.append(os.path.join(root_prefix, raw_path))
            candidates.extend(_join_tail_candidates(root_prefix, raw_path))
        if txt_dir:
            candidates.append(os.path.join(txt_dir, raw_path))
        candidates.append(raw_path)

    ordered = []
    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
        if os.path.exists(norm):
            return norm
        for variant in _numeric_frame_path_variants(norm):
            if variant in seen:
                continue
            seen.add(variant)
            ordered.append(variant)
            if os.path.exists(variant):
                return variant

    searched = "\n  ".join(ordered[:8])
    raise FileNotFoundError(
        "Sequence image not found: %s\nDATASET_ROOT_PREFIX=%s\nSearched:\n  %s"
        % (raw_path, root_prefix or "(empty)", searched)
    )


def _format_history_frame(file_name, frame_id):
    frame_dir = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    stem, ext = os.path.splitext(base_name)
    width = len(stem)
    cur = int(stem)

    zero_path_padded = os.path.join(frame_dir, f"{0:0{width}d}{ext}")
    one_path_padded = os.path.join(frame_dir, f"{1:0{width}d}{ext}")
    zero_path_plain = os.path.join(frame_dir, f"0{ext}")
    one_path_plain = os.path.join(frame_dir, f"1{ext}")
    if os.path.exists(zero_path_padded) or os.path.exists(zero_path_plain):
        min_frame_id = 0
    elif os.path.exists(one_path_padded) or os.path.exists(one_path_plain):
        min_frame_id = 1
    else:
        min_frame_id = 0 if cur == 0 else 1

    idx = max(frame_id, min_frame_id)
    candidates = [
        os.path.join(frame_dir, f"{idx:0{width}d}{ext}"),
        os.path.join(frame_dir, f"{idx}{ext}"),
        os.path.join(frame_dir, f"{idx:03d}{ext}"),
        os.path.join(frame_dir, f"{idx:04d}{ext}"),
    ]
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path

    return candidates[0]


def _history_frame_paths(line, num_frame=5):
    frame_dir = os.path.dirname(line)
    file_name = os.path.basename(line)
    stem, ext = os.path.splitext(file_name)
    index = int(stem)
    width = len(stem)

    zero_path_padded = os.path.join(frame_dir, f"{0:0{width}d}{ext}")
    one_path_padded = os.path.join(frame_dir, f"{1:0{width}d}{ext}")
    zero_path_plain = os.path.join(frame_dir, f"0{ext}")
    one_path_plain = os.path.join(frame_dir, f"1{ext}")
    if os.path.exists(zero_path_padded) or os.path.exists(zero_path_plain):
        min_frame_id = 0
    elif os.path.exists(one_path_padded) or os.path.exists(one_path_plain):
        min_frame_id = 1
    else:
        min_frame_id = 0 if index == 0 else 1

    out = []
    for i in range(index - num_frame + 1, index + 1):
        idx = max(i, min_frame_id)
        out.append(_format_history_frame(line, idx))
    return out


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    return image.convert("RGB")


def preprocess(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def _sequence_box_fix_enabled():
    value = os.environ.get("SEQUENCE_BOX_MODE") or os.environ.get("SEQUENCE_DATASET_BOX_MODE") or "float-copy"
    value = str(value).strip().lower()
    return value in ("fixed", "float", "float-copy", "copy", "detlab")


def augmentation(images, boxes, h, w, hue=.1, sat=0.7, val=0.4):
    filp = rand() < .5
    if filp:
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype("uint8")).convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for i in range(len(boxes)):
            boxes[i][[0, 2]] = w - boxes[i][[2, 0]]

    images = np.array(images, np.uint8)
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    for i in range(len(images)):
        hue_c, sat_c, val_c = cv2.split(cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV))
        dtype = images[i].dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        images[i] = cv2.merge((cv2.LUT(hue_c, lut_hue), cv2.LUT(sat_c, lut_sat), cv2.LUT(val_c, lut_val)))
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2RGB)

    return np.array(images, dtype=np.float32), np.array(boxes, dtype=np.float32)


class seqDataset(Dataset):
    def __init__(self, dataset_path, image_size, num_frame=5, type="train"):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        self.txt_path = dataset_path
        self.aug = type == "train"
        self.txt_dir = os.path.dirname(os.path.abspath(self.txt_path))
        self.fix_box_dtype = _sequence_box_fix_enabled()

        with open(self.txt_path) as f:
            data_lines = [line for line in f.readlines() if line.strip()]
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip("\n").split()
                self.img_idx.append(_resolve_sequence_path(line[0], self.txt_dir))
                if self.fix_box_dtype:
                    boxes = [np.array(list(map(float, box.split(","))), dtype=np.float32) for box in line[1:]]
                    self.anno_idx.append(np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32))
                else:
                    self.anno_idx.append(np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]]))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.transpose(preprocess(images), (3, 0, 1, 2))
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)
        return images, box

    def get_data(self, index):
        image_data = []
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]
        image_id = int(os.path.splitext(os.path.basename(file_name))[0])
        label_data = self.anno_idx[index].copy() if self.fix_box_dtype else self.anno_idx[index]
        for id in range(0, self.num_frame):
            img = Image.open(_format_history_frame(file_name, image_id - id))
            img = cvtColor(img)
            iw, ih = img.size

            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            img = img.resize((nw, nh), _BICUBIC)
            new_img = Image.new("RGB", (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            image_data.append(np.array(new_img, np.float32))

            if len(label_data) > 0 and id == 0:
                np.random.shuffle(label_data)
                label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
                label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy

                label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
                label_data[:, 2][label_data[:, 2] > w] = w
                label_data[:, 3][label_data[:, 3] > h] = h
                box_w = label_data[:, 2] - label_data[:, 0]
                box_h = label_data[:, 3] - label_data[:, 1]
                label_data = label_data[np.logical_and(box_w > 1, box_h > 1)]

        image_data = np.array(image_data[::-1])
        label_data = np.array(label_data, dtype=np.float32)
        if self.aug is True:
            pass
        return image_data, label_data


def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes


if __name__ == "__main__":
    train_dataset = seqDataset("/home/coco_val_IRDST.txt", 512, 5, "test")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=dataset_collate)
    t = time.time()
    for index, batch in enumerate(train_dataloader):
        images, targets = batch[0], batch[1]
        print(index)
    print(time.time() - t)
