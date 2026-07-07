from email.mime import image
import json
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import decode_outputs, non_max_suppression
from .utils_map import get_coco_map, get_map
from .dataloader_for_sequence import _history_frame_paths, _resolve_sequence_path

# from utils import cvtColor, preprocess_input, resize_image
# from utils_bbox import decode_outputs, non_max_suppression
# from utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir) if SummaryWriter is not None else None
        if self.writer is not None:
            try:
                dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
                self.writer.add_graph(model, dummy_input)
            except Exception:
                pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        if self.writer is not None:
            self.writer.add_scalar('loss', loss, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1,
            eval_json_path="", eval_image_root=""):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines

        self.log_dir            = log_dir
        self.cuda               = cuda
        if (not map_out_path) or os.path.normpath(map_out_path) == os.path.normpath(".temp_map_out"):
            map_out_path = os.path.join(self.log_dir, ".temp_map_out")
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        self.eval_json_path     = str(eval_json_path or "").strip()
        self.eval_image_root    = str(eval_image_root or "").strip()
        self.eval_records       = self._load_eval_records_from_json(self.eval_json_path, self.eval_image_root)
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
     
    def get_history_imgs(self, line):
        return _history_frame_paths(line)

    @staticmethod
    def _normalize_parts(path_value):
        path_value = str(path_value).replace("\\", "/")
        return [part for part in path_value.split("/") if part and part not in (".",)]

    @classmethod
    def _resolve_eval_image_path(cls, file_name, image_root, json_dir):
        file_name = str(file_name).strip()
        candidates = []
        if os.path.isabs(file_name):
            candidates.append(file_name)
        else:
            if image_root:
                root_parts = cls._normalize_parts(image_root)
                file_parts = cls._normalize_parts(file_name)
                max_overlap = min(len(root_parts), len(file_parts))
                for overlap in range(max_overlap, 0, -1):
                    if root_parts[-overlap:] == file_parts[:overlap]:
                        tail = file_parts[overlap:]
                        candidates.append(os.path.join(image_root, *tail) if tail else image_root)
                        break
                candidates.append(os.path.join(image_root, file_name))
                for keep in range(1, min(len(file_parts), 6) + 1):
                    candidates.append(os.path.join(image_root, *file_parts[-keep:]))
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

    def _category_name(self, category_id, catid_to_name):
        name = catid_to_name.get(category_id, "")
        if name in self.class_names:
            return name
        if len(self.class_names) == 1:
            return self.class_names[0]
        return name

    def _load_eval_records_from_json(self, json_path, image_root):
        if not json_path:
            return []
        if not os.path.exists(json_path):
            print("[EvalCallback] eval_json_path not found, fallback to val txt:", json_path)
            return []

        with open(json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        catid_to_name = {int(cat.get("id")): str(cat.get("name", "")) for cat in categories if "id" in cat}
        anns_by_image = {}
        for ann in annotations:
            image_id = ann.get("image_id")
            if image_id is None or ann.get("iscrowd", 0):
                continue
            anns_by_image.setdefault(image_id, []).append(ann)

        json_dir = os.path.dirname(os.path.abspath(json_path))
        records = []
        for image_info in images:
            image_id = image_info.get("id")
            file_name = image_info.get("file_name", "")
            if image_id is None or not file_name:
                continue
            image_path = self._resolve_eval_image_path(file_name, image_root, json_dir)
            boxes = []
            for ann in anns_by_image.get(image_id, []):
                bbox = ann.get("bbox", [])
                if len(bbox) < 4:
                    continue
                left, top, width, height = [float(v) for v in bbox[:4]]
                if width <= 0 or height <= 0:
                    continue
                class_name = self._category_name(int(ann.get("category_id", 1)), catid_to_name)
                if class_name not in self.class_names:
                    continue
                boxes.append((class_name, left, top, left + width, top + height))
            records.append({"image_id": str(image_id), "image_path": image_path, "boxes": boxes})

        if records:
            print("[EvalCallback] Use COCO eval json:", json_path, "| images:", len(records))
        return records


    def get_map_txt(self, image_id, images, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(images[0])[0:2])
        #---------------------------------------------------------#
        #   鍦ㄨ繖閲屽皢鍥惧儚杞崲鎴怰GB鍥惧儚锛岄槻姝㈢伆搴﹀浘鍦ㄩ娴嬫椂鎶ラ敊銆?
        #   浠ｇ爜浠呬粎鏀寔RGB鍥惧儚鐨勯娴嬶紝鎵€鏈夊叾瀹冪被鍨嬬殑鍥惧儚閮戒細杞寲鎴怰GB
        #---------------------------------------------------------#
        images       = [cvtColor(image) for image in images]
        #---------------------------------------------------------#
        #   缁欏浘鍍忓鍔犵伆鏉★紝瀹炵幇涓嶅け鐪熺殑resize
        #   涔熷彲浠ョ洿鎺esize杩涜璇嗗埆
        #---------------------------------------------------------#
        image_data  = [resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image) for image in images]
        #---------------------------------------------------------#
        #   娣诲姞涓奲atch_size缁村害
        #---------------------------------------------------------#
        image_data = [np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)) for image in image_data]
        # (3, 640, 640) -> (3, 16, 640, 640)
        image_data = np.stack(image_data, axis=1)


        image_data  = np.expand_dims(image_data, 0)


        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   灏嗗浘鍍忚緭鍏ョ綉缁滃綋涓繘琛岄娴嬶紒
            #---------------------------------------------------------#
            outputs = self.net(images) 
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   灏嗛娴嬫杩涜鍫嗗彔锛岀劧鍚庤繘琛岄潪鏋佸ぇ鎶戝埗
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        if self.eval_flag and (epoch == 1 or epoch % self.period == 0):
            self.net = model_eval
            if os.path.exists(self.map_out_path):
                shutil.rmtree(self.map_out_path)
            os.makedirs(os.path.join(self.map_out_path, "ground-truth"), exist_ok=True)
            os.makedirs(os.path.join(self.map_out_path, "detection-results"), exist_ok=True)

            print("Get map.")
            if self.eval_records:
                for record in tqdm(self.eval_records):
                    image_id = record["image_id"]
                    images = self.get_history_imgs(record["image_path"])
                    images = [Image.open(item) for item in images]

                    self.get_map_txt(image_id, images, self.class_names, self.map_out_path)

                    with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                        for obj_name, left, top, right, bottom in record["boxes"]:
                            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            else:
                for annotation_line in tqdm(self.val_lines):
                    line = annotation_line.split()
                    resolved_path = _resolve_sequence_path(line[0])
                    path_parts = [part for part in resolved_path.replace("\\", "/").split("/") if part]
                    image_id = "-".join(path_parts[-2:]).split('.')[0]

                    images = self.get_history_imgs(resolved_path)
                    images = [Image.open(item) for item in images]
                    gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

                    self.get_map_txt(image_id, images, self.class_names, self.map_out_path)

                    with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                        for box in gt_boxes:
                            left, top, right, bottom, obj = box
                            obj_name = self.class_names[obj]
                            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            metrics = {}
            try:
                coco_stats = get_coco_map(class_names=self.class_names, path=self.map_out_path)
                temp_map = float(coco_stats[1])
                metrics = {
                    "AP@[0.50:0.95]": float(coco_stats[0]),
                    "AP@0.50": float(coco_stats[1]),
                    "AP@0.75": float(coco_stats[2]),
                    "AP_small": float(coco_stats[3]),
                    "AP_medium": float(coco_stats[4]),
                    "AP_large": float(coco_stats[5]),
                    "AR@1": float(coco_stats[6]),
                    "AR@10": float(coco_stats[7]),
                    "AR@100": float(coco_stats[8]),
                    "AR_small": float(coco_stats[9]),
                    "AR_medium": float(coco_stats[10]),
                    "AR_large": float(coco_stats[11]),
                }
            except Exception:
                temp_map = float(get_map(self.MINOVERLAP, False, path=self.map_out_path))
                metrics = {
                    "mAP": temp_map,
                    "MINOVERLAP": float(self.MINOVERLAP),
                }

            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), "a") as f:
                f.write(str(temp_map))
                f.write("\n")

            metrics_path = os.path.join(self.log_dir, "epoch_%03d_metrics.json" % epoch)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            plt.figure()
            plt.plot(self.epoches, self.maps, "red", linewidth=2, label="train map")
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Map %s" % str(self.MINOVERLAP))
            plt.title("A Map Curve")
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)







# def get_history_imgs(line):
#     dir_path = line.replace(line.split('/')[-1],'')
#     file_type = line.split('.')[-1]
#     index = int(line.split('/')[-1][:-4])
#     image_id    = "-".join(line.split("/")[6:8]).split('.')[0]
#     print(image_id)
    
#     return [os.path.join(dir_path,  "%d.%s" % (max(id, 0),file_type)) for id in range(index - 4, index + 1)]


# if __name__ == "__main__":
#     with open('coco_val.txt', encoding='utf-8') as f:
#         val_lines   = f.readlines()
#     for annotation_line in val_lines:
#         line        = annotation_line.split()
#         images = get_history_imgs(line[0])
#         # for item in images:
#         #     print(item)
#         # break



