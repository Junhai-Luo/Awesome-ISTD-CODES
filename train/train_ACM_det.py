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
from tqdm import tqdm

from nets.acm.det_loss import YOLOLoss
from nets.acm.detection import ASKCResNetFPNDet, ASKCResUNetDet, ASKCResUNetSaliencyDet
from nets.yolo_training import ModelEMA
from utils.acm.data_detlab import DetlabTxtDetDataset, det_dataset_collate
from utils.acm.data_detlab import _resolve_image_path
from utils.acm.det_bbox import decode_outputs, non_max_suppression
from utils.acm.lr_scheduler import adjust_learning_rate
from utils.callbacks import LossHistory
from utils.coco_compat import ensure_coco_dataset_compat


def build_dataset_components():
    print("[ACM Train] Dataset: utils/acm/data_detlab.py")
    return DetlabTxtDetDataset, det_dataset_collate


def parse_args():
    parser = ArgumentParser(description="ACM single-frame detector training")
    parser.add_argument("--init-model-path", type=str, default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--train-txt-path", type=str, default=os.environ.get("TRAIN_ANNOTATION_PATH", ""))
    parser.add_argument("--val-txt-path", type=str, default=os.environ.get("VAL_ANNOTATION_PATH", ""))
    parser.add_argument("--image-root", type=str, default=os.environ.get("ACM_IMAGE_ROOT", ""))
    parser.add_argument("--base-size", type=int, default=int(os.environ.get("ACM_BASE_SIZE", 512)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("ACM_BATCH_SIZE", 8)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("ACM_EPOCHS", 100)))
    parser.add_argument("--run-epochs", type=int, default=int(os.environ.get("ACM_RUN_EPOCHS", "0") or "0"))
    parser.add_argument("--warm-up-epochs", type=int, default=int(os.environ.get("ACM_WARM_UP_EPOCHS", 0)))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("INIT_LR", 0.01)))
    parser.add_argument("--min-learning-rate", type=float, default=float(os.environ.get("MIN_LR", 1e-6)))
    parser.add_argument("--optimizer-type", type=str, default=os.environ.get("OPTIMIZER_TYPE", "adagrad"), choices=["adagrad", "adam", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", 1e-4)))
    parser.add_argument("--momentum", type=float, default=float(os.environ.get("MOMENTUM", 0.9)))
    parser.add_argument("--backbone-mode", type=str, default=os.environ.get("ACM_BACKBONE_MODE", "FPN"), choices=["FPN", "UNet"])
    parser.add_argument("--det-mode", type=str, default=os.environ.get("ACM_DET_MODE", "feature"), choices=["feature", "saliency"])
    parser.add_argument("--fuse-mode", type=str, default=os.environ.get("ACM_FUSE_MODE", "AsymBi"), choices=["BiLocal", "AsymBi", "BiGlobal"])
    parser.add_argument("--blocks-per-layer", type=int, default=int(os.environ.get("ACM_BLOCKS_PER_LAYER", 4)))
    parser.add_argument("--num-classes", type=int, default=int(os.environ.get("ACM_NUM_CLASSES", 1)))
    parser.add_argument("--stride", type=int, default=int(os.environ.get("ACM_STRIDE", 8)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("ACM_NUM_WORKERS", 2)))
    parser.add_argument("--save-period", type=int, default=int(os.environ.get("ACM_SAVE_EVERY", 10)))
    parser.add_argument("--mosaic", type=int, default=int(os.environ.get("ACM_MOSAIC", "1")))
    parser.add_argument("--mosaic-prob", type=float, default=float(os.environ.get("ACM_MOSAIC_PROB", "0.5")))
    parser.add_argument("--mixup", type=int, default=int(os.environ.get("ACM_MIXUP", "1")))
    parser.add_argument("--mixup-prob", type=float, default=float(os.environ.get("ACM_MIXUP_PROB", "0.5")))
    parser.add_argument("--special-aug-ratio", type=float, default=float(os.environ.get("ACM_SPECIAL_AUG_RATIO", "0.7")))
    parser.add_argument("--ema", type=int, default=int(os.environ.get("ACM_EMA", "0")))
    parser.add_argument("--freeze-train", type=int, default=int(os.environ.get("ACM_FREEZE_TRAIN", "0")))
    parser.add_argument("--freeze-epochs", type=int, default=int(os.environ.get("ACM_FREEZE_EPOCHS", "50")))
    parser.add_argument("--lr-decay-type", type=str, default=os.environ.get("ACM_LR_DECAY_TYPE", "cos"), choices=["cos", "step"])
    parser.add_argument("--grad-clip-norm", type=float, default=float(os.environ.get("ACM_GRAD_CLIP_NORM", "0") or "0"))
    parser.add_argument("--skip-invalid-grad", type=int, default=int(os.environ.get("ACM_SKIP_INVALID_GRAD", "0") or "0"))

    parser.add_argument("--eval-flag", type=int, default=int(os.environ.get("ACM_EVAL_FLAG", "1")))
    parser.add_argument("--eval-period", type=int, default=int(os.environ.get("ACM_EVAL_PERIOD", "10")))
    parser.add_argument("--eval-json-path", type=str, default=os.environ.get("ACM_EVAL_JSON_PATH", ""))
    parser.add_argument("--eval-image-root", type=str, default=os.environ.get("ACM_EVAL_IMAGE_ROOT", ""))
    parser.add_argument("--eval-confidence", type=float, default=float(os.environ.get("ACM_EVAL_CONFIDENCE", "0.001")))
    parser.add_argument("--eval-nms-iou", type=float, default=float(os.environ.get("ACM_EVAL_NMS_IOU", "0.65")))

    dataset_name = os.environ.get("DATASET_NAME", "dataset")
    network_name = os.environ.get("NETWORK_NAME", "acm")
    save_root = os.environ.get("SAVE_ROOT", "logs")
    parser.add_argument("--save-dir", type=str, default=os.path.join(save_root, dataset_name, network_name))
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_cls, collate_fn = build_dataset_components()

        train_kwargs = dict(
            txt_path=args.train_txt_path,
            input_size=args.base_size,
            image_root=args.image_root,
            train=True,
            mosaic=bool(int(args.mosaic)),
            mixup=bool(int(args.mixup)),
            mosaic_prob=args.mosaic_prob,
            mixup_prob=args.mixup_prob,
            epoch_length=args.epochs,
            special_aug_ratio=args.special_aug_ratio,
        )
        val_kwargs = dict(
            txt_path=args.val_txt_path,
            input_size=args.base_size,
            image_root=args.image_root,
            train=False,
        )
        train_set = dataset_cls(**train_kwargs)
        val_set = dataset_cls(**val_kwargs)
        self.train_loader = Data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.val_loader = Data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        layer_blocks = [args.blocks_per_layer] * 3
        channels = [8, 16, 32, 64]
        if args.det_mode == "saliency":
            if args.backbone_mode != "UNet":
                raise ValueError("ACM saliency detection requires --backbone-mode UNet.")
            self.net = ASKCResUNetSaliencyDet(layer_blocks, channels, args.fuse_mode, args.num_classes)
            print("[ACM Train] Model: DETLAB ASKCResUNetSaliencyDet.")
        elif args.backbone_mode == "FPN":
            self.net = ASKCResNetFPNDet(layer_blocks, channels, args.fuse_mode, args.num_classes)
        else:
            self.net = ASKCResUNetDet(layer_blocks, channels, args.fuse_mode, args.num_classes)

        self.net.apply(self.weight_init)
        if args.init_model_path:
            self._load_init_weights(args.init_model_path)
        self.net = self.net.to(self.device)

        print("[ACM Train] Loss: nets/acm/det_loss.py")
        self.criterion = YOLOLoss(num_classes=args.num_classes, strides=[args.stride])
        self.freeze_train = bool(int(args.freeze_train))
        self.unfreeze_done = False
        if self.freeze_train and int(args.freeze_epochs) > 0:
            self._set_backbone_trainable(False)
            print("[ACM Train] Freeze backbone for first %d epochs." % int(args.freeze_epochs))

        self.init_lr_fit, self.min_lr_fit = self._resolve_fit_lr(args.batch_size)
        self.optimizer = self._build_optimizer(self.init_lr_fit)
        self.lr_scheduler_func = None
        self.ema = ModelEMA(self.net) if bool(int(args.ema)) else None
        if self.ema is not None:
            self.ema.updates = len(self.train_loader) * 0
            print("[ACM Train] EMA enabled.")

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
        if self.eval_enabled:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), "a") as f:
                f.write("0\n")
        else:
            print("[ACM Eval] Disabled because eval_json_path or eval_image_root is empty.")

        self.infer_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_init_weights(self, path):
        if not os.path.exists(path):
            print("Init model not found, train ACM from scratch:", path)
            return
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = self._normalize_init_state_dict(state)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        print("Loaded init model:", path)
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

    def _normalize_init_state_dict(self, state):
        if not isinstance(state, dict):
            return state

        normalized = {}
        for key, value in state.items():
            new_key = str(key)
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            replacements = {
                "head.cls_pred.": "head.cls_preds.",
                "head.reg_pred.": "head.reg_preds.",
                "head.obj_pred.": "head.obj_preds.",
            }
            for old, new in replacements.items():
                if new_key.startswith(old):
                    new_key = new + new_key[len(old) :]
                    break
            normalized[new_key] = value
        return normalized

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)

    def _set_backbone_trainable(self, trainable):
        backbone = getattr(self.net, "backbone", None)
        if backbone is None:
            return
        for param in backbone.parameters():
            param.requires_grad = trainable

    def _resolve_fit_lr(self, batch_size):
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

    @staticmethod
    def _outputs_are_finite(outputs):
        return all(torch.isfinite(output).all().item() for output in outputs)

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
            print("[ACM Train] Unfreeze backbone at epoch %d." % epoch)

        self.net.train()
        if hasattr(self.train_loader.dataset, "epoch_now"):
            self.train_loader.dataset.epoch_now = epoch - 1
        losses = []
        skipped = 0
        tbar = tqdm(self.train_loader)
        for batch_idx, (images, targets) in enumerate(tbar):
            images = images.to(self.device)
            targets = self._to_device_targets(targets)

            outputs = self.net(images)
            if not self._outputs_are_finite(outputs):
                message = "[ACM Train] non-finite model outputs at epoch %d batch %d" % (epoch, batch_idx)
                if bool(int(self.args.skip_invalid_grad)):
                    skipped += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    tbar.set_description("Epoch:%3d lr:%f train_loss:%f skipped:%d" % (epoch, self.optimizer.param_groups[0]["lr"], np.mean(losses) if losses else 0.0, skipped))
                    continue
                raise FloatingPointError(message)
            loss = self.criterion(outputs, targets)
            if not torch.isfinite(loss).item():
                message = "[ACM Train] non-finite loss at epoch %d batch %d: %s" % (epoch, batch_idx, str(loss.detach().item()))
                if bool(int(self.args.skip_invalid_grad)):
                    skipped += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    tbar.set_description("Epoch:%3d lr:%f train_loss:%f skipped:%d" % (epoch, self.optimizer.param_groups[0]["lr"], np.mean(losses) if losses else 0.0, skipped))
                    continue
                raise FloatingPointError(message)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(self.args.grad_clip_norm) > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), float(self.args.grad_clip_norm))
                if not torch.isfinite(grad_norm).item():
                    message = "[ACM Train] non-finite grad norm at epoch %d batch %d: %s" % (epoch, batch_idx, str(float(grad_norm.detach().cpu())))
                    if bool(int(self.args.skip_invalid_grad)):
                        skipped += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        tbar.set_description("Epoch:%3d lr:%f train_loss:%f skipped:%d" % (epoch, self.optimizer.param_groups[0]["lr"], np.mean(losses) if losses else 0.0, skipped))
                        continue
                    raise FloatingPointError(message)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.net)

            losses.append(loss.item())
            tbar.set_description("Epoch:%3d lr:%f train_loss:%f skipped:%d" % (epoch, self.optimizer.param_groups[0]["lr"], np.mean(losses), skipped))

        if self.lr_scheduler_func is None:
            adjust_learning_rate(
                self.optimizer,
                epoch,
                self.args.epochs,
                self.args.learning_rate,
                self.args.warm_up_epochs,
                self.args.min_learning_rate,
            )
        if self.writer is not None:
            self.writer.add_scalar("Lr/value", self.optimizer.param_groups[0]["lr"], epoch)
        return float(np.mean(losses))

    def validate_one_epoch(self, epoch):
        eval_net = self._eval_net()
        eval_net.eval()
        losses = []
        skipped = 0
        tbar = tqdm(self.val_loader)
        for batch_idx, (images, targets) in enumerate(tbar):
            images = images.to(self.device)
            targets = self._to_device_targets(targets)
            with torch.no_grad():
                outputs = eval_net(images)
                if not self._outputs_are_finite(outputs):
                    skipped += 1
                    tbar.set_description("Epoch:%3d val_loss:%f skipped:%d" % (epoch, np.mean(losses) if losses else 0.0, skipped))
                    continue
                loss = self.criterion(outputs, targets)
                if not torch.isfinite(loss).item():
                    skipped += 1
                    tbar.set_description("Epoch:%3d val_loss:%f skipped:%d" % (epoch, np.mean(losses) if losses else 0.0, skipped))
                    continue
            losses.append(loss.item())
            tbar.set_description("Epoch:%3d val_loss:%f skipped:%d" % (epoch, np.mean(losses), skipped))
        return float(np.mean(losses)) if losses else float("inf")

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

    def evaluate_coco_map(self, epoch):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except Exception as e:
            print("[ACM Eval] pycocotools unavailable, skip eval:", str(e))
            return None

        coco = ensure_coco_dataset_compat(COCO(self.args.eval_json_path))
        image_ids = coco.getImgIds()
        clsid2catid = coco.getCatIds()
        det_results = []
        num_found_images = 0
        num_missing_images = 0
        score_sum = 0.0
        score_count = 0
        score_max = 0.0
        eval_json_dir = ops.dirname(ops.abspath(self.args.eval_json_path))

        eval_net = self._eval_net()
        eval_net.eval()
        for image_id in tqdm(image_ids, desc="ACM Eval", leave=False):
            image_info = coco.loadImgs(image_id)[0]
            file_name = image_info.get("file_name", "")
            if not file_name:
                continue
            image_path = _resolve_image_path(file_name, self.args.eval_image_root, eval_json_dir)
            if not ops.exists(image_path):
                num_missing_images += 1
                continue
            num_found_images += 1

            image = Image.open(image_path).convert("RGB")
            image_shape = (image.height, image.width)
            image_data = self._preprocess_image(image).to(self.device)

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
                score = float(obj_conf * cls_conf)
                score_sum += score
                score_count += 1
                score_max = max(score_max, score)

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
        metrics.update(
            {
                "num_eval_images": int(len(image_ids)),
                "num_found_images": int(num_found_images),
                "num_missing_images": int(num_missing_images),
                "num_detections": int(len(det_results)),
                "max_score": float(score_max),
                "mean_score": float(score_sum / max(score_count, 1)),
                "eval_confidence": float(self.args.eval_confidence),
                "eval_nms_iou": float(self.args.eval_nms_iou),
            }
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        map50 = float(metrics.get("AP@0.50", 0.0))
        self.maps.append(map50)
        self.epoches.append(epoch)
        with open(os.path.join(self.log_dir, "epoch_map.txt"), "a") as f:
            f.write(str(map50))
            f.write("\n")
        self._write_map_curve()

        if self.writer is not None:
            self.writer.add_scalar("Eval/mAP50", map50, epoch)
            self.writer.add_scalar("Eval/mAP5095", float(metrics.get("AP@[0.50:0.95]", 0.0)), epoch)

        print(
            "[ACM Eval] images=%d found=%d missing=%d detections=%d max_score=%.6f mean_score=%.6f conf=%.6f nms=%.3f"
            % (
                int(len(image_ids)),
                int(num_found_images),
                int(num_missing_images),
                int(len(det_results)),
                float(score_max),
                float(score_sum / max(score_count, 1)),
                float(self.args.eval_confidence),
                float(self.args.eval_nms_iou),
            )
        )
        print("[ACM Eval] Epoch %d => mAP50=%.6f, mAP50:95=%.6f" % (epoch, map50, float(metrics.get("AP@[0.50:0.95]", 0.0))))
        return metrics


def main():
    args = parse_args()
    if not args.train_txt_path or not args.val_txt_path:
        raise ValueError("ACM training requires train/val txt paths. Please set TRAIN_ANNOTATION_PATH and VAL_ANNOTATION_PATH.")

    trainer = Trainer(args)
    run_epochs = int(args.run_epochs) if int(args.run_epochs) > 0 else int(args.epochs)
    if run_epochs != int(args.epochs):
        print("[ACM Train] Run epochs limited to %d while schedule/augmentation epoch_length stays %d." % (run_epochs, int(args.epochs)))

    for epoch in range(1, run_epochs + 1):
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
