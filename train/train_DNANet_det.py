import json
import os
import os.path as ops
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from nets.dna.det_loss import YOLOLoss
from nets.dna.model_DNANet_det import DNANetDet, DNANetSaliencyDet
from nets.yolo_training import ModelEMA
from utils.callbacks import LossHistory
from utils.coco_compat import ensure_coco_dataset_compat
from utils.dna.det_bbox import decode_outputs, non_max_suppression
from utils.dna.det_dataset import (
    _resolve_image_path,
    DetTxtDataset,
    det_dataset_collate,
)


def parse_args():
    parser = ArgumentParser(description="DNANet bbox detector training")
    parser.add_argument("--init-model-path", type=str, default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--train-txt-path", type=str, default=os.environ.get("TRAIN_ANNOTATION_PATH", ""))
    parser.add_argument("--val-txt-path", type=str, default=os.environ.get("VAL_ANNOTATION_PATH", ""))
    parser.add_argument("--image-root", type=str, default=os.environ.get("DNA_IMAGE_ROOT", ""))
    parser.add_argument("--source-image-root", type=str, default=os.environ.get("DNA_SOURCE_IMAGE_ROOT", ""))

    parser.add_argument("--eval-json-path", type=str, default=os.environ.get("DNA_EVAL_JSON_PATH", ""))
    parser.add_argument("--eval-image-root", type=str, default=os.environ.get("DNA_EVAL_IMAGE_ROOT", ""))
    parser.add_argument("--base-size", type=int, default=int(os.environ.get("DNA_BASE_SIZE", 512)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DNA_BATCH_SIZE", 8)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("DNA_EPOCHS", 100)))
    parser.add_argument("--warm-up-epochs", type=int, default=int(os.environ.get("DNA_WARM_UP_EPOCHS", 0)))

    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("INIT_LR", 0.01)))
    parser.add_argument("--min-learning-rate", type=float, default=float(os.environ.get("MIN_LR", 1e-6)))
    parser.add_argument("--optimizer-type", type=str, default=os.environ.get("OPTIMIZER_TYPE", "adagrad"), choices=["adagrad", "adam", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", 1e-4)))
    parser.add_argument("--momentum", type=float, default=float(os.environ.get("MOMENTUM", 0.9)))

    parser.add_argument("--num-classes", type=int, default=int(os.environ.get("DNA_NUM_CLASSES", 1)))
    parser.add_argument("--stride", type=int, default=int(os.environ.get("DNA_STRIDE", 8)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("DNA_NUM_WORKERS", 2)))
    parser.add_argument("--save-period", type=int, default=int(os.environ.get("DNA_SAVE_PERIOD", 10)))
    parser.add_argument("--grad-clip-norm", type=float, default=float(os.environ.get("DNA_GRAD_CLIP_NORM", "1.0")))
    parser.add_argument("--skip-invalid-grad", type=int, default=int(os.environ.get("DNA_SKIP_INVALID_GRAD", "1")))

    parser.add_argument("--eval-flag", type=int, default=int(os.environ.get("DNA_EVAL_FLAG", "1")))
    parser.add_argument("--eval-period", type=int, default=int(os.environ.get("DNA_EVAL_PERIOD", "10")))
    parser.add_argument("--eval-confidence", type=float, default=float(os.environ.get("DNA_EVAL_CONFIDENCE", "0.001")))
    parser.add_argument("--eval-nms-iou", type=float, default=float(os.environ.get("DNA_EVAL_NMS_IOU", "0.65")))

    parser.add_argument("--channel-size", type=str, default=os.environ.get("DNA_CHANNEL_SIZE", "three"))
    parser.add_argument("--backbone", type=str, default=os.environ.get("DNA_BACKBONE", "resnet_18"))
    parser.add_argument(
        "--det-mode",
        type=str,
        default=os.environ.get("DNA_DET_MODE", "feature"),
        choices=["feature", "saliency"],
    )
    parser.add_argument("--suffix", type=str, default=os.environ.get("DNA_SUFFIX", ".png"))
    parser.add_argument("--mosaic", type=int, default=int(os.environ.get("DNA_MOSAIC", "1")))
    parser.add_argument("--mosaic-prob", type=float, default=float(os.environ.get("DNA_MOSAIC_PROB", "0.5")))
    parser.add_argument("--mixup", type=int, default=int(os.environ.get("DNA_MIXUP", "1")))
    parser.add_argument("--mixup-prob", type=float, default=float(os.environ.get("DNA_MIXUP_PROB", "0.5")))
    parser.add_argument("--special-aug-ratio", type=float, default=float(os.environ.get("DNA_SPECIAL_AUG_RATIO", "0.7")))
    parser.add_argument("--ema", type=int, default=int(os.environ.get("DNA_EMA", "0")))
    parser.add_argument("--freeze-train", type=int, default=int(os.environ.get("DNA_FREEZE_TRAIN", "0")))
    parser.add_argument("--freeze-epochs", type=int, default=int(os.environ.get("DNA_FREEZE_EPOCHS", "50")))
    parser.add_argument("--lr-decay-type", type=str, default=os.environ.get("DNA_LR_DECAY_TYPE", "cos"), choices=["cos", "step"])

    dataset_name = os.environ.get("DATASET_NAME", "dataset")
    network_name = os.environ.get("NETWORK_NAME", "dnanet_det")
    save_root = os.environ.get("SAVE_ROOT", "logs")
    parser.add_argument("--save-dir", type=str, default=os.path.join(save_root, dataset_name, network_name))
    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, total_epochs, init_lr, warm_up_epochs, min_lr):
    if epoch <= warm_up_epochs and warm_up_epochs > 0:
        lr = init_lr * float(epoch) / float(max(warm_up_epochs, 1))
    else:
        progress = float(epoch - warm_up_epochs) / float(max(total_epochs - warm_up_epochs, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        lr = min_lr + (init_lr - min_lr) * cosine

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def scaled_fit_lrs(batch_size, optimizer_type, init_lr, min_lr):
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == "adam" else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    return init_lr_fit, min_lr_fit


def xywh_iou(box_a, box_b):
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2 = ax1 + max(aw, 0.0)
    ay2 = ay1 + max(ah, 0.0)
    bx2 = bx1 + max(bw, 0.0)
    by2 = by1 + max(bh, 0.0)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return 0.0 if union <= 0.0 else inter / union


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset_cls = DetTxtDataset
        collate_fn = det_dataset_collate
        train_kwargs = {
            "txt_path": args.train_txt_path,
            "input_size": args.base_size,
            "image_root": args.image_root,
            "train": True,
            "suffix": args.suffix,
            "source_image_root": args.source_image_root,
        }
        val_kwargs = {
            "txt_path": args.val_txt_path,
            "input_size": args.base_size,
            "image_root": args.image_root,
            "train": False,
            "suffix": args.suffix,
            "source_image_root": args.source_image_root,
        }
        print("[DNANet Train] Dataset: utils/dna/det_dataset.DetTxtDataset")

        train_set = dataset_cls(**train_kwargs)
        val_set = dataset_cls(**val_kwargs)
        self.train_loader = data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.val_loader = data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        model_cls = DNANetSaliencyDet if args.det_mode == "saliency" else DNANetDet
        self.net = model_cls(
            input_channels=3,
            num_classes=args.num_classes,
            channel_size=args.channel_size,
            backbone=args.backbone,
        )
        self.net.apply(self.weight_init)
        if args.init_model_path:
            self._load_init_weights(args.init_model_path)
        self.net = self.net.to(self.device)

        self.strides = [args.stride]
        self.criterion = YOLOLoss(num_classes=args.num_classes, strides=self.strides)
        self.freeze_train = bool(int(args.freeze_train))
        self.unfreeze_done = False
        if self.freeze_train and int(args.freeze_epochs) > 0:
            self._set_backbone_trainable(False)
            print("[DNANet Train] Freeze backbone for first %d epochs." % int(args.freeze_epochs))

        self.fit_learning_rate, self.fit_min_learning_rate = self._resolve_fit_lr(args.batch_size)
        self.optimizer = self._build_optimizer(self.fit_learning_rate)
        self.lr_scheduler_func = None
        self.ema = ModelEMA(self.net) if bool(int(args.ema)) else None
        if self.ema is not None:
            self.ema.updates = len(self.train_loader) * 0
            print("[DNANet Train] EMA enabled.")

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        time_str = "%s_pid%d" % (__import__("time").strftime("%Y_%m_%d_%H_%M_%S", __import__("time").localtime()), os.getpid())
        self.log_dir = os.path.join(self.save_dir, "loss_" + str(time_str))
        self.loss_history = LossHistory(self.log_dir, self.net, input_shape=[args.base_size, args.base_size])
        self.writer = self.loss_history.writer

        self.best_val_loss = float("inf")
        self.eval_enabled = bool(int(args.eval_flag)) and bool(args.eval_json_path) and bool(args.eval_image_root or args.image_root)
        self.maps = [0.0]
        self.epoches = [0]
        if self.eval_enabled:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), "a", encoding="utf-8") as f:
                f.write("0\n")
        else:
            print("[DNANet Eval] Disabled because eval_json_path or eval_image_root/image_root is empty.")

        self.infer_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        print(
            "DNANet effective lr:",
            "init_lr=%g" % self.fit_learning_rate,
            "min_lr=%g" % self.fit_min_learning_rate,
            "(configured init_lr=%g min_lr=%g)" % (args.learning_rate, args.min_learning_rate),
            "grad_clip_norm=%g" % args.grad_clip_norm,
            "skip_invalid_grad=%d" % int(args.skip_invalid_grad),
            "ema=%d" % int(args.ema),
        )

    def _load_init_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("init model not found: %s" % path)
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
        print(
            "Loaded keys:",
            len(filtered),
            "Source keys:",
            len(state),
            "Skipped source keys:",
            max(0, len(state) - len(filtered)),
            "Missing model keys:",
            len(missing),
            "Unexpected keys ignored:",
            len(unexpected),
        )
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
            nn.init.constant_(m.bias, 0)

    def _set_backbone_trainable(self, trainable):
        backbone = getattr(self.net, "backbone", None)
        if backbone is None:
            return
        for param in backbone.parameters():
            param.requires_grad = trainable

    def _resolve_fit_lr(self, batch_size):
        if self.args.det_mode == "saliency":
            return scaled_fit_lrs(
                batch_size,
                self.args.optimizer_type,
                self.args.learning_rate,
                self.args.min_learning_rate,
            )
        return float(self.args.learning_rate), float(self.args.min_learning_rate)

    def _build_optimizer(self, lr):
        if self.args.optimizer_type == "adam":
            return torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        if self.args.optimizer_type == "sgd":
            return torch.optim.SGD(
                self.net.parameters(),
                lr=lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=True,
            )
        return torch.optim.Adagrad(self.net.parameters(), lr=lr, weight_decay=self.args.weight_decay)

    def _eval_net(self):
        if self.ema is not None:
            return self.ema.ema
        return self.net

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
            if target.numel() == 0:
                continue
            if target.ndim != 2 or (target.numel() > 0 and target.shape[1] != 5):
                raise ValueError(
                    "%s targets[%d] must have shape [N, 5] at epoch %d batch %d, got %s"
                    % (stage, target_idx, epoch, batch_idx, tuple(target.shape))
                )
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

    def _loss_is_valid(self, loss, epoch, batch_idx, stage):
        if bool(torch.isfinite(loss).item()):
            return True
        message = "%s loss became invalid at epoch %d batch %d: %s" % (
            stage,
            epoch,
            batch_idx,
            float(loss.detach().cpu().item()),
        )
        if stage == "train" and int(self.args.skip_invalid_grad):
            print("[DNANet Train] Skip batch:", message)
            return False
        raise FloatingPointError(message)

    def _grad_norm_is_valid(self, grad_norm, epoch, batch_idx):
        grad_norm_invalid = False
        grad_norm_value = grad_norm
        if torch.is_tensor(grad_norm):
            grad_norm_invalid = not bool(torch.isfinite(grad_norm).item())
            grad_norm_value = float(grad_norm.detach().cpu().item())
        else:
            grad_norm_invalid = not np.isfinite(float(grad_norm))
            grad_norm_value = float(grad_norm)
        if grad_norm_invalid:
            message = "train gradient norm became invalid at epoch %d batch %d: %s" % (
                epoch,
                batch_idx,
                grad_norm_value,
            )
            if int(self.args.skip_invalid_grad):
                print("[DNANet Train] Skip batch:", message)
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
        if self.freeze_train and (epoch > int(self.args.freeze_epochs)) and not self.unfreeze_done:
            self._set_backbone_trainable(True)
            self.unfreeze_done = True
            if self.ema is not None:
                self.ema.updates = len(self.train_loader) * (epoch - 1)
            print("[DNANet Train] Unfreeze backbone at epoch %d." % epoch)

        self.net.train()
        if hasattr(self.train_loader.dataset, "epoch_now"):
            self.train_loader.dataset.epoch_now = epoch - 1
        losses = []
        losses_iou = []
        losses_obj = []
        losses_cls = []
        num_fg_list = []
        skipped = 0
        tbar = tqdm(self.train_loader)
        for batch_idx, (images, targets, _) in enumerate(tbar, start=1):
            images = images.to(self.device)
            self._raise_on_invalid_tensor("images", images, epoch, batch_idx, "train")
            targets = self._to_device_targets(targets)
            self._check_targets(targets, epoch, batch_idx, "train")

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.net(images)
            self._check_outputs(outputs, epoch, batch_idx, "train")
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict["loss"]
            if not self._loss_is_valid(loss, epoch, batch_idx, "train"):
                skipped += 1
                tbar.set_description(
                    "Epoch:%3d lr:%f train_loss:%f iou:%f obj:%f cls:%f skipped:%d"
                    % (
                        epoch,
                        self.optimizer.param_groups[0]["lr"],
                        np.mean(losses) if losses else 0.0,
                        np.mean(losses_iou) if losses_iou else 0.0,
                        np.mean(losses_obj) if losses_obj else 0.0,
                        np.mean(losses_cls) if losses_cls else 0.0,
                        skipped,
                    )
                )
                continue

            loss.backward()
            if self.args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
                if not self._grad_norm_is_valid(grad_norm, epoch, batch_idx):
                    skipped += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    tbar.set_description(
                        "Epoch:%3d lr:%f train_loss:%f iou:%f obj:%f cls:%f skipped:%d"
                        % (
                            epoch,
                            self.optimizer.param_groups[0]["lr"],
                            np.mean(losses) if losses else 0.0,
                            np.mean(losses_iou) if losses_iou else 0.0,
                            np.mean(losses_obj) if losses_obj else 0.0,
                            np.mean(losses_cls) if losses_cls else 0.0,
                            skipped,
                        )
                    )
                    continue
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.net)

            losses.append(loss.item())
            losses_iou.append(loss_dict["loss_iou"].item())
            losses_obj.append(loss_dict["loss_obj"].item())
            losses_cls.append(loss_dict["loss_cls"].item())
            num_fg_list.append(loss_dict["num_fg"].item())
            tbar.set_description(
                "Epoch:%3d lr:%f train_loss:%f iou:%f obj:%f cls:%f skipped:%d"
                % (
                    epoch,
                    self.optimizer.param_groups[0]["lr"],
                    np.mean(losses),
                    np.mean(losses_iou),
                    np.mean(losses_obj),
                    np.mean(losses_cls),
                    skipped,
                )
            )

        if self.lr_scheduler_func is None:
            lr = adjust_learning_rate(
                self.optimizer,
                epoch,
                self.args.epochs,
                self.fit_learning_rate,
                self.args.warm_up_epochs,
                self.fit_min_learning_rate,
            )
        else:
            lr = self.optimizer.param_groups[0]["lr"]
        if self.writer is not None:
            self.writer.add_scalar("Lr/value", lr, epoch)
        return float(np.mean(losses)) if losses else 0.0

    def validate_one_epoch(self, epoch):
        eval_net = self._eval_net()
        eval_net.eval()
        losses = []
        tbar = tqdm(self.val_loader)
        for batch_idx, (images, targets, _) in enumerate(tbar, start=1):
            images = images.to(self.device)
            self._raise_on_invalid_tensor("images", images, epoch, batch_idx, "val")
            targets = self._to_device_targets(targets)
            self._check_targets(targets, epoch, batch_idx, "val")
            with torch.no_grad():
                outputs = eval_net(images)
                self._check_outputs(outputs, epoch, batch_idx, "val")
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["loss"]
                self._loss_is_valid(loss, epoch, batch_idx, "val")
            losses.append(loss.item())
            tbar.set_description("Epoch:%3d val_loss:%f" % (epoch, np.mean(losses)))
        return float(np.mean(losses)) if losses else 0.0

    def save_weights(self, epoch, train_loss, val_loss):
        state_dict = self._eval_net().state_dict()
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

    def evaluate_map(self, epoch):
        if not self.eval_enabled:
            return None

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except Exception as e:
            print("[DNANet Eval] pycocotools unavailable, skip eval:", str(e))
            return None

        coco = ensure_coco_dataset_compat(COCO(self.args.eval_json_path))
        image_ids = coco.getImgIds()
        clsid2catid = coco.getCatIds()
        det_results = []
        num_found_images = 0
        num_missing_images = 0
        missing_examples = []
        score_sum = 0.0
        score_count = 0
        score_max = 0.0
        num_gt_annotations = 0
        images_with_detections = 0
        images_with_iou_10 = 0
        images_with_iou_30 = 0
        images_with_iou_50 = 0
        pred_best_iou_sum = 0.0
        pred_best_iou_count = 0
        pred_best_iou_max = 0.0
        diagnostic_examples = []

        gt_by_image = {}
        for image_id in image_ids:
            anns = []
            for ann in coco.imgToAnns.get(image_id, []):
                if int(ann.get("iscrowd", 0)) != 0:
                    continue
                bbox = ann.get("bbox", None)
                if not bbox or len(bbox) < 4:
                    continue
                anns.append(
                    {
                        "category_id": int(ann.get("category_id", -1)),
                        "bbox": [float(v) for v in bbox[:4]],
                    }
                )
            gt_by_image[int(image_id)] = anns
            num_gt_annotations += len(anns)

        eval_root = self.args.eval_image_root or self.args.image_root
        eval_json_dir = ops.dirname(ops.abspath(self.args.eval_json_path))

        eval_net = self._eval_net()
        eval_net.eval()
        for image_id in tqdm(image_ids, desc="DNANet Eval", leave=False):
            image_info = coco.loadImgs(image_id)[0]
            file_name = str(image_info.get("file_name", "")).strip()
            if not file_name:
                continue

            image_path = _resolve_image_path(
                file_name,
                eval_root,
                eval_json_dir,
                suffix="",
                source_image_root=self.args.source_image_root,
            )
            if not ops.exists(image_path):
                num_missing_images += 1
                if len(missing_examples) < 5:
                    missing_examples.append({"file_name": file_name, "resolved_path": image_path})
                continue

            num_found_images += 1
            pil_image = Image.open(image_path).convert("RGB")
            image_shape = (pil_image.height, pil_image.width)
            image_data = self._preprocess_image(pil_image).to(self.device)

            with torch.no_grad():
                outputs = eval_net(image_data)
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

            images_with_detections += 1
            image_best_iou = 0.0
            for det in detections:
                y1, x1, y2, x2, obj_conf, cls_conf, cls_id = det.tolist()
                cls_idx = int(cls_id)
                if cls_idx < 0:
                    continue
                if cls_idx < len(clsid2catid):
                    category_id = int(clsid2catid[cls_idx])
                else:
                    category_id = int(cls_idx + 1)
                pred_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                gt_candidates = [
                    ann
                    for ann in gt_by_image.get(int(image_id), [])
                    if ann["category_id"] == category_id
                ]
                best_iou = 0.0
                for ann in gt_candidates:
                    best_iou = max(best_iou, xywh_iou(pred_bbox, ann["bbox"]))
                image_best_iou = max(image_best_iou, best_iou)
                pred_best_iou_sum += best_iou
                pred_best_iou_count += 1
                pred_best_iou_max = max(pred_best_iou_max, best_iou)

                det_results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": pred_bbox,
                        "score": float(obj_conf * cls_conf),
                    }
                )
                score = float(obj_conf * cls_conf)
                score_sum += score
                score_count += 1
                score_max = max(score_max, score)
                diagnostic_examples.append(
                    {
                        "image_id": int(image_id),
                        "file_name": file_name,
                        "category_id": int(category_id),
                        "score": float(score),
                        "best_iou": float(best_iou),
                        "bbox": pred_bbox,
                        "num_same_category_gt": int(len(gt_candidates)),
                    }
                )
                diagnostic_examples = sorted(
                    diagnostic_examples,
                    key=lambda item: item["score"],
                    reverse=True,
                )[:20]

            if image_best_iou >= 0.1:
                images_with_iou_10 += 1
            if image_best_iou >= 0.3:
                images_with_iou_30 += 1
            if image_best_iou >= 0.5:
                images_with_iou_50 += 1

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
            tmp_pred_json = os.path.join(self.log_dir, "_tmp_eval_results.json")
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
                if os.path.exists(tmp_pred_json):
                    os.remove(tmp_pred_json)

        metrics_path = os.path.join(self.log_dir, "epoch_%03d_metrics.json" % epoch)
        metrics.update(
            {
                "num_eval_images": int(len(image_ids)),
                "num_found_images": int(num_found_images),
                "num_missing_images": int(num_missing_images),
                "num_gt_annotations": int(num_gt_annotations),
                "num_detections": int(len(det_results)),
                "num_images_with_detections": int(images_with_detections),
                "num_images_with_best_iou_ge_0.10": int(images_with_iou_10),
                "num_images_with_best_iou_ge_0.30": int(images_with_iou_30),
                "num_images_with_best_iou_ge_0.50": int(images_with_iou_50),
                "max_pred_best_iou": float(pred_best_iou_max),
                "mean_pred_best_iou": float(pred_best_iou_sum / max(pred_best_iou_count, 1)),
                "max_score": float(score_max),
                "mean_score": float(score_sum / max(score_count, 1)),
                "eval_confidence": float(self.args.eval_confidence),
                "eval_nms_iou": float(self.args.eval_nms_iou),
                "det_mode": str(self.args.det_mode),
                "base_size": int(self.args.base_size),
                "stride": int(self.args.stride),
            }
        )
        if missing_examples:
            metrics["missing_image_examples"] = missing_examples
        if diagnostic_examples:
            metrics["top_score_diagnostic_examples"] = diagnostic_examples
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(
            "[DNANet Eval]",
            "images=%d found=%d missing=%d gt=%d detections=%d max_score=%.6f mean_score=%.6f max_best_iou=%.4f iou50_images=%d conf=%.6f nms=%.3f"
            % (
                len(image_ids),
                num_found_images,
                num_missing_images,
                num_gt_annotations,
                len(det_results),
                score_max,
                score_sum / max(score_count, 1),
                pred_best_iou_max,
                images_with_iou_50,
                self.args.eval_confidence,
                self.args.eval_nms_iou,
            ),
        )
        if missing_examples:
            print("[DNANet Eval] missing image examples:", missing_examples)

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
        raise ValueError("DNANet detection training requires train/val txt paths.")
    if not os.path.exists(args.train_txt_path):
        raise FileNotFoundError("Train txt not found: %s" % args.train_txt_path)
    if not os.path.exists(args.val_txt_path):
        raise FileNotFoundError("Val txt not found: %s" % args.val_txt_path)

    trainer = Trainer(args)
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_one_epoch(epoch)
        val_loss = trainer.validate_one_epoch(epoch)
        trainer.loss_history.append_loss(epoch, train_loss, val_loss)
        if trainer.eval_enabled and (epoch == 1 or epoch % max(1, int(args.eval_period)) == 0):
            trainer.evaluate_map(epoch)
        print("Epoch:%d/%d" % (epoch, args.epochs))
        print("Total Loss: %.3f || Val Loss: %.3f" % (train_loss, val_loss))
        trainer.save_weights(epoch, train_loss, val_loss)

    if trainer.loss_history.writer is not None:
        trainer.loss_history.writer.close()


if __name__ == "__main__":
    main()
