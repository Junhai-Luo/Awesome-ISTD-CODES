import json
import os
import os.path as ops
import time
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from nets.uiu.det_loss import YOLOLoss
from nets.uiu.detection import UIUNETDet, UIUNETSaliencyDet
from utils.acm.data_detlab import DetlabTxtDetDataset, _resolve_image_path, det_dataset_collate
from utils.callbacks import LossHistory
from utils.coco_compat import ensure_coco_dataset_compat
from utils.uiu.det_bbox import decode_outputs, non_max_suppression


def build_dataset_components():
    print("[UIU Train] Dataset: utils/acm/data_detlab.py")
    return "detlab", DetlabTxtDetDataset, det_dataset_collate


def parse_args():
    parser = ArgumentParser(description="UIUNet single-frame detector training")
    parser.add_argument("--init-model-path", type=str, default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--train-txt-path", type=str, default=os.environ.get("TRAIN_ANNOTATION_PATH", ""))
    parser.add_argument("--val-txt-path", type=str, default=os.environ.get("VAL_ANNOTATION_PATH", ""))
    parser.add_argument("--image-root", type=str, default=os.environ.get("UIU_IMAGE_ROOT", ""))
    parser.add_argument("--base-size", type=int, default=int(os.environ.get("UIU_BASE_SIZE", 512)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("UIU_BATCH_SIZE", 8)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("UIU_EPOCHS", 300)))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("INIT_LR", 0.001)))
    parser.add_argument("--min-learning-rate", type=float, default=float(os.environ.get("MIN_LR", 1e-6)))
    parser.add_argument("--optimizer-type", type=str, default=os.environ.get("OPTIMIZER_TYPE", "adam"), choices=["adagrad", "adam", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", 0.0)))
    parser.add_argument("--momentum", type=float, default=float(os.environ.get("MOMENTUM", 0.9)))

    parser.add_argument("--fuse-mode", type=str, default=os.environ.get("UIU_FUSE_MODE", "AsymBi"), choices=["AsymBi"])
    parser.add_argument("--det-mode", type=str, default=os.environ.get("UIU_DET_MODE", "feature"), choices=["feature", "saliency"])
    parser.add_argument("--num-classes", type=int, default=int(os.environ.get("UIU_NUM_CLASSES", 1)))
    parser.add_argument("--stride", type=str, default=os.environ.get("UIU_STRIDE", "8"))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("UIU_NUM_WORKERS", 2)))
    parser.add_argument("--save-period", type=int, default=int(os.environ.get("UIU_SAVE_PERIOD", 10)))

    parser.add_argument("--fp16", action="store_true", default=bool(int(os.environ.get("UIU_FP16", "0"))))
    parser.add_argument("--grad-clip-norm", type=float, default=float(os.environ.get("UIU_GRAD_CLIP_NORM", 1.0)))
    parser.add_argument("--skip-invalid-grad", type=int, default=int(os.environ.get("UIU_SKIP_INVALID_GRAD", "1")))

    parser.add_argument("--eval-flag", type=int, default=int(os.environ.get("UIU_EVAL_FLAG", "1")))
    parser.add_argument("--eval-period", type=int, default=int(os.environ.get("UIU_EVAL_PERIOD", "10")))
    parser.add_argument("--eval-json-path", type=str, default=os.environ.get("UIU_EVAL_JSON_PATH", ""))
    parser.add_argument("--eval-image-root", type=str, default=os.environ.get("UIU_EVAL_IMAGE_ROOT", ""))
    parser.add_argument("--eval-confidence", type=float, default=float(os.environ.get("UIU_EVAL_CONFIDENCE", "0.001")))
    parser.add_argument("--eval-nms-iou", type=float, default=float(os.environ.get("UIU_EVAL_NMS_IOU", "0.65")))

    dataset_name = os.environ.get("DATASET_NAME", "dataset")
    network_name = os.environ.get("NETWORK_NAME", "uiunet")
    save_root = os.environ.get("SAVE_ROOT", "logs")
    parser.add_argument("--save-dir", type=str, default=os.path.join(save_root, dataset_name, network_name))
    return parser.parse_args()


def parse_strides(stride_arg):
    text = str(stride_arg).strip()
    if "," in text:
        strides = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        strides = [int(text)] if text else []
    if len(strides) == 0:
        raise ValueError("At least one stride must be provided.")
    return strides


def adjust_learning_rate(optimizer, epoch, total_epochs, init_lr, min_lr):
    cur_lr = pow(1 - float(epoch) / (total_epochs + 1), 0.9) * (init_lr - min_lr) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr
    return cur_lr


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.strides = parse_strides(args.stride)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, dataset_cls, collate_fn = build_dataset_components()

        train_kwargs = dict(txt_path=args.train_txt_path, input_size=args.base_size, image_root=args.image_root, train=True)
        val_kwargs = dict(txt_path=args.val_txt_path, input_size=args.base_size, image_root=args.image_root, train=False)
        if dataset_name == "yolox":
            train_kwargs["num_classes"] = args.num_classes
            val_kwargs["num_classes"] = args.num_classes
        train_set = dataset_cls(**train_kwargs)
        val_set = dataset_cls(**val_kwargs)
        self.train_loader = Data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        self.val_loader = Data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

        model_cls = UIUNETSaliencyDet if args.det_mode == "saliency" else UIUNETDet
        self.net = model_cls(in_ch=3, num_classes=args.num_classes, fuse_mode=args.fuse_mode)
        print("[UIU Train] Model:", model_cls.__name__, "det_mode=%s" % args.det_mode)
        self.net.apply(self.weight_init)
        if args.init_model_path:
            self._load_init_weights(args.init_model_path)
        self.net = self.net.to(self.device)

        self.criterion = YOLOLoss(num_classes=args.num_classes, strides=self.strides)
        if args.optimizer_type == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True,
            )
        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.scaler = GradScaler("cuda", enabled=args.fp16 and self.device.type == "cuda")

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        time_str = "%s_pid%d" % (time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())), os.getpid())
        self.log_dir = os.path.join(self.save_dir, "loss_" + str(time_str))
        self.loss_history = LossHistory(self.log_dir, self.net, input_shape=[args.base_size, args.base_size])
        self.writer = self.loss_history.writer

        self.best_val_loss = float("inf")
        self.eval_enabled = bool(int(args.eval_flag)) and bool(args.eval_json_path) and bool(args.eval_image_root)
        self.maps = [0.0]
        self.epoches = [0]
        with open(os.path.join(self.log_dir, "epoch_map.txt"), "a", encoding="utf-8") as f:
            f.write("0\n")

        self.infer_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_init_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("init model not found: %s" % path)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        current = self.net.state_dict()
        filtered = {}
        skipped_examples = []
        for key, value in state.items():
            matched = False
            for candidate in self._state_key_candidates(key):
                if candidate in current and current[candidate].shape == value.shape:
                    filtered[candidate] = value
                    matched = True
                    break
            if (not matched) and len(skipped_examples) < 10:
                skipped_examples.append(str(key))
        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        print("Loaded init model:", path)
        print("Loaded keys:", len(filtered), "Source keys:", len(state), "Missing keys:", len(missing), "Unexpected keys ignored:", len(unexpected))
        if skipped_examples:
            print("Skipped key examples:", skipped_examples)

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

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

    def _to_device_targets(self, targets):
        return [t.to(self.device) for t in targets]

    def _raise_on_invalid_tensor(self, name, tensor, epoch, batch_idx, stage):
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            detached = tensor.detach()
            finite_mask = torch.isfinite(detached)
            finite_values = detached[finite_mask]
            finite_min = float(finite_values.min().item()) if finite_values.numel() > 0 else float("nan")
            finite_max = float(finite_values.max().item()) if finite_values.numel() > 0 else float("nan")
            raise FloatingPointError(
                "%s %s became invalid at epoch %d batch %d. shape=%s finite_min=%s finite_max=%s"
                % (stage, name, epoch, batch_idx, tuple(detached.shape), finite_min, finite_max)
            )

    def _check_targets(self, targets, epoch, batch_idx, stage):
        for target_idx, target in enumerate(targets):
            self._raise_on_invalid_tensor("targets[%d]" % target_idx, target, epoch, batch_idx, stage)
            if target.ndim != 2 or (target.numel() > 0 and target.shape[1] != 5):
                raise ValueError(
                    "%s targets[%d] must have shape [N, 5] at epoch %d batch %d, got %s"
                    % (stage, target_idx, epoch, batch_idx, tuple(target.shape))
                )
            if target.numel() > 0:
                bad_size_mask = target[:, 2:4].min(dim=1).values <= 0
                if torch.any(bad_size_mask):
                    bad_boxes = target[bad_size_mask]
                    raise ValueError(
                        "%s found non-positive box width/height at epoch %d batch %d target %d: %s"
                        % (stage, epoch, batch_idx, target_idx, bad_boxes[:5].detach().cpu().tolist())
                    )

                cls_ids = target[:, 4]
                cls_ids_long = cls_ids.to(torch.int64)
                if torch.any(cls_ids_long < 0) or torch.any(cls_ids_long >= self.args.num_classes):
                    raise ValueError(
                        "%s found invalid class ids at epoch %d batch %d target %d. Expected [0, %d), got %s"
                        % (stage, epoch, batch_idx, target_idx, self.args.num_classes, cls_ids[:20].detach().cpu().tolist())
                    )

    def _check_outputs(self, outputs, epoch, batch_idx, stage):
        if len(outputs) != len(self.strides):
            raise ValueError(
                "%s model returned %d detection levels at epoch %d batch %d, but strides=%s has %d entries"
                % (stage, len(outputs), epoch, batch_idx, self.strides, len(self.strides))
            )
        for output_idx, output in enumerate(outputs):
            self._raise_on_invalid_tensor("outputs[%d]" % output_idx, output, epoch, batch_idx, stage)

    def _check_loss(self, loss, epoch, batch_idx, stage):
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "%s loss became invalid at epoch %d batch %d: %s"
                % (stage, epoch, batch_idx, float(loss.detach().cpu().item()))
            )

    def _check_grad_norm(self, grad_norm, epoch, batch_idx):
        if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm):
            message = "train gradient norm became invalid at epoch %d batch %d: %s" % (
                epoch,
                batch_idx,
                float(grad_norm.detach().cpu().item()),
            )
            if int(self.args.skip_invalid_grad):
                print("[UIU Train] Skip batch:", message)
                return False
            raise FloatingPointError(message)
        return True

    def _preprocess_image(self, image):
        iw, ih = image.size
        w = int(self.args.base_size)
        h = int(self.args.base_size)
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        resized = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(resized, ((w - nw) // 2, (h - nh) // 2))
        return self.infer_transform(new_image).unsqueeze(0)

    def train_one_epoch(self, epoch):
        self.net.train()
        losses = []
        skipped = 0
        tbar = tqdm(self.train_loader)
        for batch_idx, (images, targets) in enumerate(tbar, start=1):
            images = images.to(self.device)
            self._raise_on_invalid_tensor("images", images, epoch, batch_idx, "train")
            targets = self._to_device_targets(targets)
            self._check_targets(targets, epoch, batch_idx, "train")

            self.optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=self.args.fp16 and self.device.type == "cuda"):
                outputs = self.net(images)
            self._check_outputs(outputs, epoch, batch_idx, "train")
            with autocast(device_type="cuda", enabled=False):
                loss = self.criterion(outputs, targets)
            self._check_loss(loss, epoch, batch_idx, "train")

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                if not self._check_grad_norm(grad_norm, epoch, batch_idx):
                    skipped += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    tbar.set_description(
                        "Epoch:%3d lr:%f train_loss:%f skipped:%d"
                        % (epoch, self.optimizer.param_groups[0]["lr"], np.mean(losses) if losses else 0.0, skipped)
                    )
                    continue
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())
            tbar.set_description(
                "Epoch:%3d lr:%f train_loss:%f skipped:%d"
                % (epoch, self.optimizer.param_groups[0]["lr"], np.mean(losses), skipped)
            )

        lr = adjust_learning_rate(
            self.optimizer,
            epoch,
            self.args.epochs,
            self.args.learning_rate,
            self.args.min_learning_rate,
        )
        if self.writer is not None:
            self.writer.add_scalar("Lr/value", lr, epoch)
        return float(np.mean(losses)) if losses else 0.0

    def validate_one_epoch(self, epoch):
        self.net.eval()
        losses = []
        tbar = tqdm(self.val_loader)
        for batch_idx, (images, targets) in enumerate(tbar, start=1):
            images = images.to(self.device)
            self._raise_on_invalid_tensor("images", images, epoch, batch_idx, "val")
            targets = self._to_device_targets(targets)
            self._check_targets(targets, epoch, batch_idx, "val")
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=self.args.fp16 and self.device.type == "cuda"):
                    outputs = self.net(images)
                self._check_outputs(outputs, epoch, batch_idx, "val")
                with autocast(device_type="cuda", enabled=False):
                    loss = self.criterion(outputs, targets)
                self._check_loss(loss, epoch, batch_idx, "val")
            losses.append(loss.item())
            tbar.set_description("Epoch:%3d val_loss:%f" % (epoch, np.mean(losses)))
        return float(np.mean(losses)) if losses else 0.0

    def save_weights(self, epoch, train_loss, val_loss):
        state_dict = self.net.state_dict()
        save_dir = self.log_dir

        if (epoch % max(1, int(self.args.save_period)) == 0) or (epoch == self.args.epochs):
            name = "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch, train_loss, val_loss)
            torch.save(state_dict, os.path.join(save_dir, name))

        if len(self.loss_history.val_loss) <= 1 or val_loss <= min(self.loss_history.val_loss):
            print("Save best model to best_epoch_weights.pth")
            torch.save(state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

    def _write_map_curve(self):
        plt.figure()
        plt.plot(self.epoches, self.maps, "red", linewidth=2, label="train map")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Map 0.5")
        plt.title("A Map Curve")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
        plt.cla()
        plt.close("all")

    def evaluate_coco_map(self, epoch):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except Exception as e:
            print("[UIU Eval] pycocotools unavailable, skip eval:", str(e))
            return None

        coco = ensure_coco_dataset_compat(COCO(self.args.eval_json_path))
        image_ids = coco.getImgIds()
        clsid2catid = coco.getCatIds()
        det_results = []
        eval_json_dir = ops.dirname(ops.abspath(self.args.eval_json_path))

        self.net.eval()
        for image_id in tqdm(image_ids, desc="UIU Eval", leave=False):
            image_info = coco.loadImgs(image_id)[0]
            file_name = image_info.get("file_name", "")
            if not file_name:
                continue
            image_path = _resolve_image_path(file_name, self.args.eval_image_root, eval_json_dir)
            if not ops.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            image_shape = (image.height, image.width)
            image_data = self._preprocess_image(image).to(self.device)

            with torch.no_grad():
                outputs = self.net(image_data)
                decoded = decode_outputs(outputs, (self.args.base_size, self.args.base_size))
                detections = non_max_suppression(
                    decoded,
                    num_classes=self.args.num_classes,
                    input_shape=(self.args.base_size, self.args.base_size),
                    image_shape=image_shape,
                    conf_thres=self.args.eval_confidence,
                    nms_thres=self.args.eval_nms_iou,
                    letterbox_image=True,
                )[0]

            if detections is None:
                continue

            for det in detections:
                y1, x1, y2, x2, obj_conf, cls_conf, cls_id = det.tolist()
                cls_idx = int(cls_id)
                if cls_idx < 0:
                    continue
                if cls_idx < len(clsid2catid):
                    category_id = int(clsid2catid[cls_idx])
                else:
                    category_id = int(cls_idx + 1)
                det_results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(obj_conf * cls_conf),
                    }
                )

        if len(det_results) == 0:
            metrics = {
                "AP@[0.50:0.95]": 0.0,
                "AP@0.50": 0.0,
                "AP@0.75": 0.0,
                "AR@1": 0.0,
                "AR@10": 0.0,
                "AR@100": 0.0,
            }
        else:
            tmp_pred_json = ops.join(self.log_dir, "_tmp_eval_results.json")
            try:
                with open(tmp_pred_json, "w", encoding="utf-8") as f:
                    json.dump(det_results, f, ensure_ascii=False)

                coco_dt = coco.loadRes(tmp_pred_json)
                coco_eval = COCOeval(coco, coco_dt, "bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                metrics = {
                    "AP@[0.50:0.95]": float(coco_eval.stats[0]),
                    "AP@0.50": float(coco_eval.stats[1]),
                    "AP@0.75": float(coco_eval.stats[2]),
                    "AR@1": float(coco_eval.stats[6]),
                    "AR@10": float(coco_eval.stats[7]),
                    "AR@100": float(coco_eval.stats[8]),
                }
            finally:
                if ops.exists(tmp_pred_json):
                    os.remove(tmp_pred_json)

        metrics_path = os.path.join(self.log_dir, "epoch_%03d_metrics.json" % epoch)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        map50 = float(metrics.get("AP@0.50", 0.0))
        self.maps.append(map50)
        self.epoches.append(epoch)
        with open(os.path.join(self.log_dir, "epoch_map.txt"), "a", encoding="utf-8") as f:
            f.write(str(map50))
            f.write("\n")
        self._write_map_curve()

        if self.writer is not None:
            self.writer.add_scalar("Eval/mAP50", map50, epoch)
            self.writer.add_scalar("Eval/mAP5095", float(metrics.get("AP@[0.50:0.95]", 0.0)), epoch)
        return metrics


def main():
    args = parse_args()
    if not args.train_txt_path or not args.val_txt_path:
        raise ValueError("UIUNet training requires train/val txt paths. Please set TRAIN_ANNOTATION_PATH and VAL_ANNOTATION_PATH.")

    trainer = Trainer(args)
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_one_epoch(epoch)
        val_loss = trainer.validate_one_epoch(epoch)

        trainer.loss_history.append_loss(epoch, train_loss, val_loss)
        if trainer.eval_enabled and (epoch == 1 or epoch % max(1, int(args.eval_period)) == 0):
            trainer.evaluate_coco_map(epoch)

        print("Epoch:%d/%d" % (epoch, args.epochs))
        print("Total Loss: %.3f || Val Loss: %.3f" % (train_loss, val_loss))
        trainer.save_weights(epoch, train_loss, val_loss)

    if trainer.loss_history.writer is not None:
        trainer.loss_history.writer.close()


if __name__ == "__main__":
    main()
