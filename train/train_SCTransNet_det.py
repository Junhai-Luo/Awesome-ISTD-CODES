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
from nets.sctransnet.Config import get_SCTrans_config
from nets.sctransnet.SCTransNetDet import SCTransNetDet
from utils.acm.data_detlab import DetlabTxtDetDataset, det_dataset_collate
from utils.acm.det_bbox import decode_outputs, non_max_suppression
from utils.acm.lr_scheduler import adjust_learning_rate
from utils.callbacks import LossHistory
from utils.coco_compat import ensure_coco_dataset_compat
from utils.dna.det_dataset import _resolve_image_path


def build_dataset_components():
    print("[SCTransNet Train] Dataset: utils/acm/data_detlab.py")
    return "detlab", DetlabTxtDetDataset, det_dataset_collate


def parse_args():
    parser = ArgumentParser(description="SCTransNet single-frame detector training")
    parser.add_argument("--init-model-path", type=str, default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--train-txt-path", type=str, default=os.environ.get("TRAIN_ANNOTATION_PATH", ""))
    parser.add_argument("--val-txt-path", type=str, default=os.environ.get("VAL_ANNOTATION_PATH", ""))
    parser.add_argument("--image-root", type=str, default=os.environ.get("SCTRANS_IMAGE_ROOT", ""))
    parser.add_argument("--base-size", type=int, default=int(os.environ.get("SCTRANS_BASE_SIZE", 512)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SCTRANS_BATCH_SIZE", 4)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SCTRANS_EPOCHS", 100)))
    parser.add_argument("--warm-up-epochs", type=int, default=int(os.environ.get("SCTRANS_WARM_UP_EPOCHS", 0)))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("INIT_LR", 0.01)))
    parser.add_argument("--min-learning-rate", type=float, default=float(os.environ.get("MIN_LR", 1e-6)))
    parser.add_argument("--optimizer-type", type=str, default=os.environ.get("OPTIMIZER_TYPE", "adagrad"), choices=["adagrad", "adam", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", 1e-4)))
    parser.add_argument("--momentum", type=float, default=float(os.environ.get("MOMENTUM", 0.9)))
    parser.add_argument("--num-classes", type=int, default=int(os.environ.get("SCTRANS_NUM_CLASSES", 1)))
    parser.add_argument("--stride", type=int, default=int(os.environ.get("SCTRANS_STRIDE", 8)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SCTRANS_NUM_WORKERS", 2)))
    parser.add_argument("--save-period", type=int, default=int(os.environ.get("SCTRANS_SAVE_EVERY", 10)))
    parser.add_argument("--mosaic", type=int, default=int(os.environ.get("SCTRANS_MOSAIC", "1")))
    parser.add_argument("--mosaic-prob", type=float, default=float(os.environ.get("SCTRANS_MOSAIC_PROB", "0.5")))
    parser.add_argument("--mixup", type=int, default=int(os.environ.get("SCTRANS_MIXUP", "1")))
    parser.add_argument("--mixup-prob", type=float, default=float(os.environ.get("SCTRANS_MIXUP_PROB", "0.5")))
    parser.add_argument("--special-aug-ratio", type=float, default=float(os.environ.get("SCTRANS_SPECIAL_AUG_RATIO", "0.7")))

    parser.add_argument("--eval-flag", type=int, default=int(os.environ.get("SCTRANS_EVAL_FLAG", "1")))
    parser.add_argument("--eval-period", type=int, default=int(os.environ.get("SCTRANS_EVAL_PERIOD", "10")))
    parser.add_argument("--eval-json-path", type=str, default=os.environ.get("SCTRANS_EVAL_JSON_PATH", ""))
    parser.add_argument("--eval-image-root", type=str, default=os.environ.get("SCTRANS_EVAL_IMAGE_ROOT", ""))
    parser.add_argument("--eval-confidence", type=float, default=float(os.environ.get("SCTRANS_EVAL_CONFIDENCE", "0.001")))
    parser.add_argument("--eval-nms-iou", type=float, default=float(os.environ.get("SCTRANS_EVAL_NMS_IOU", "0.65")))

    dataset_name = os.environ.get("DATASET_NAME", "dataset")
    network_name = os.environ.get("NETWORK_NAME", "sctransnet")
    save_root = os.environ.get("SAVE_ROOT", "logs")
    parser.add_argument("--save-dir", type=str, default=os.path.join(save_root, dataset_name, network_name))
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_name, dataset_cls, collate_fn = build_dataset_components()

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
        val_kwargs = dict(txt_path=args.val_txt_path, input_size=args.base_size, image_root=args.image_root, train=False)
        if dataset_name == "yolox":
            train_kwargs["num_classes"] = args.num_classes
            val_kwargs["num_classes"] = args.num_classes
        train_set = dataset_cls(**train_kwargs)
        val_set = dataset_cls(**val_kwargs)
        self.train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        self.val_loader = Data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        self.net = SCTransNetDet(
            get_SCTrans_config(),
            n_channels=3,
            num_classes=args.num_classes,
            img_size=args.base_size,
        )
        print("[SCTransNet Train] Model: SCTransNetDet")

        self.net.apply(self.weight_init)
        if args.init_model_path:
            self._load_init_weights(args.init_model_path)
        self.net = self.net.to(self.device)

        self.criterion = YOLOLoss(num_classes=args.num_classes, strides=[args.stride])
        if args.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        else:
            self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        time_str = "%s_pid%d" % (time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())), os.getpid())
        self.log_dir = os.path.join(self.save_dir, "loss_" + str(time_str))
        self.loss_history = LossHistory(self.log_dir, self.net, input_shape=[args.base_size, args.base_size])
        self.writer = self.loss_history.writer
        self.best_val_loss = float("inf")
        self.eval_enabled = bool(int(args.eval_flag)) and bool(args.eval_json_path)
        self.maps = [0.0]
        self.epoches = [0]
        with open(os.path.join(self.log_dir, "epoch_map.txt"), "a", encoding="utf-8") as f:
            f.write("0\n")
        if not self.eval_enabled:
            print("[SCTransNet Eval] Disabled because eval_json_path is empty.")

        self.infer_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def _load_init_weights(self, path):
        if not os.path.exists(path):
            print("[SCTransNet Train] init model not found, train from scratch:", path)
            return
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        current = self.net.state_dict()
        filtered = {k: v for k, v in state.items() if (k in current and current[k].shape == v.shape)}
        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        skipped = len(state) - len(filtered) if isinstance(state, dict) else 0
        print("Loaded init model:", path)
        print("Matched keys:", len(filtered), "Missing keys:", len(missing), "Unexpected keys:", len(unexpected), "Skipped mismatched:", skipped)

    def weight_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0, 0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)

    def _to_device_targets(self, targets):
        return [t.to(self.device) for t in targets]

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
        tbar = tqdm(self.train_loader)
        for images, targets in tbar:
            images = images.to(self.device)
            targets = self._to_device_targets(targets)
            outputs = self.net(images)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            tbar.set_description("Epoch:%3d lr:%f train_loss:%f" % (epoch, self.optimizer.param_groups[0]["lr"], np.mean(losses)))
        adjust_learning_rate(self.optimizer, epoch, self.args.epochs, self.args.learning_rate, self.args.warm_up_epochs, self.args.min_learning_rate)
        if self.writer is not None:
            self.writer.add_scalar("Lr/value", self.optimizer.param_groups[0]["lr"], epoch)
        return float(np.mean(losses))

    def validate_one_epoch(self, epoch):
        self.net.eval()
        losses = []
        tbar = tqdm(self.val_loader)
        for images, targets in tbar:
            images = images.to(self.device)
            targets = self._to_device_targets(targets)
            with torch.no_grad():
                outputs = self.net(images)
                loss = self.criterion(outputs, targets)
            losses.append(loss.item())
            tbar.set_description("Epoch:%3d val_loss:%f" % (epoch, np.mean(losses)))
        return float(np.mean(losses))

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
        plt.title("SCTransNet Map Curve")
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
            print("[SCTransNet Eval] pycocotools unavailable, skip eval:", str(e))
            return None

        coco = ensure_coco_dataset_compat(COCO(self.args.eval_json_path))
        image_ids = coco.getImgIds()
        clsid2catid = coco.getCatIds()
        det_results = []

        eval_root = self.args.eval_image_root or self.args.image_root
        eval_json_dir = ops.dirname(ops.abspath(self.args.eval_json_path))

        self.net.eval()
        for image_id in tqdm(image_ids, desc="SCTransNet Eval", leave=False):
            image_info = coco.loadImgs(image_id)[0]
            file_name = str(image_info.get("file_name", "")).strip()
            if not file_name:
                continue

            image_path = _resolve_image_path(
                file_name,
                eval_root,
                eval_json_dir,
                suffix="",
                source_image_root="",
            )
            if not ops.exists(image_path):
                continue

            pil_image = Image.open(image_path).convert("RGB")
            image_shape = (pil_image.height, pil_image.width)
            image_data = self._preprocess_image(pil_image).to(self.device)

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
        raise ValueError("SCTransNet training requires train/val txt paths. Please set TRAIN_ANNOTATION_PATH and VAL_ANNOTATION_PATH.")
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
