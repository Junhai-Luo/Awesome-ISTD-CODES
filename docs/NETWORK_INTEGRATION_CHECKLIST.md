# MIRSTD-Lab 统一训练/预测与新网络接入规范

这份文档用于维护 MIRSTD-Lab 当前的统一实验体系，重点覆盖：

- `run_train_experiment.py` 训练入口
- `run_predict_experiment_all.py` 预测/评估入口
- `configs/*.json` 配置文件
- `data_txt/` 与 `data_json/` 数据文件
- `nets/`、`train/`、`infer/`、`utils/` 中新增网络时的代码位置

目标是让新增网络、数据集和评估流程都能纳入同一套管理方式，而不是每个网络维护一套临时脚本。

## 1. 当前统一入口

训练入口：

```bash
python run_train_experiment.py --dataset IRDST --network sstnet
python run_train_experiment.py --dataset ITSDT_15K --network tridos --dry-run
```

预测/评估入口：

```bash
python run_predict_experiment_all.py --dataset IRDST --network sstnet --weight-policy loss
python run_predict_experiment_all.py --dataset DAUB --network alcnet --output-mode eval_only
```

常用优先级从高到低：

1. 命令行参数，例如 `--dataset`、`--network`、`--model-path`
2. `configs/train_experiment_config.json` 或 `configs/predict_experiment_config.json`
3. `configs/experiment_presets.json`

新增功能优先接到这两个入口，除非是一次性排查脚本。

## 2. 当前数据集配置

数据集统一在 `configs/experiment_presets.json -> dataset_presets` 注册。

当前数据集：

- `DAUB`
- `IRDST`
- `ITSDT_15K`

每个数据集至少需要：

```json
"DATASET_NAME": {
  "train_script": "train/train_sequence_det.py",
  "default_train_txt": "data_txt/DATASET_train.txt",
  "default_val_txt": "data_txt/DATASET_val.txt",
  "default_predict": {
    "json_path": "data_json/DATASET_test.json",
    "dataset_img_path": "datasets/DATASET"
  }
}
```

说明：

- `default_train_txt` 用于训练。
- `default_val_txt` 用于训练期 `val_loss`。
- `default_predict.json_path` 用于训练期 mAP 和预测期 COCO 评估。
- `default_predict.dataset_img_path` 是图片根目录。

注意：训练 loss 通常走 `txt`，mAP 评估走 `json`。例如 ITSDT：

```text
train loss: data_txt/ITSDT_15k_train.txt
val loss:   data_txt/ITSDT_15k_val.txt
mAP eval:   data_json/ITSDT_15k_test.json
image root: datasets/ITSDT-15k
```

## 3. 数据文件规范

### 3.1 `data_txt/`

txt 每行格式：

```text
image_path x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
```

示例：

```text
datasets/IRDST/images/71/336.bmp 100,120,108,128,0
datasets/ITSDT-15k/images/1/0.bmp
datasets/ITSDT-15k/images/1/4.bmp 432,471,453,479,0
```

规范：

- 路径使用 `/`。
- 不建议写死服务器绝对路径。
- 路径大小写要和磁盘一致，例如 `images` 不要写成 `Images`。
- 序列数据的帧号要和磁盘一致，例如 ITSDT 当前是 `0.bmp`、`1.bmp`，不是 `001.bmp`。
- 空标注图片保留图片路径即可。

### 3.2 `data_json/`

json 使用 COCO 格式：

```json
{
  "info": {},
  "licenses": [],
  "images": [],
  "annotations": [],
  "categories": []
}
```

如果历史 json 缺少 `info` 或 `licenses`，评估入口应使用 `utils/coco_compat.py -> ensure_coco_dataset_compat()` 做兼容。

### 3.3 图片根路径

全局根路径在：

```json
"global": {
  "dataset_root_prefix": "/project/IDIP/QXY"
}
```

本地 txt 可以写：

```text
datasets/IRDST/...
```

训练/预测时由 `dataset_root_prefix` 或 `dataset_img_path` 补齐为服务器绝对路径。

## 4. 当前网络命名

统一可运行网络名以 `configs/experiment_presets.json -> network_pretrained_map` 为准。

当前保留的 canonical 名称：

- `sstnet`
- `tridos`
- `slowfastnet`
- `slowfastnet_9520`
- `acm_fpn`
- `acm_unet`
- `acm_unet_saliency`
- `dnanet`
- `dnanet_saliency`
- `uiunet`
- `uiunet_saliency`
- `alcnet`
- `alcnet_saliency`
- `sctransnet`

新增网络不要只加别名。推荐先定义一个唯一 canonical 名称，再按需在 `network_alias` 里加短别名。

## 5. 配置文件职责

### 5.1 `configs/experiment_presets.json`

这是最重要的配置文件，管理稳定默认值。

新增网络时通常要改：

- `network_pretrained_map`
- `network_alias`
- `network_train_defaults`
- `network_presets`
- `network_train_scripts`

字段说明：

```json
"network_pretrained_map": {
  "newnet": "model_data/newnet_pretrained.pth"
}
```

绑定默认预训练权重。没有预训练就填空字符串。

```json
"network_alias": {
  "newnet": "newnet"
}
```

把命令行名称映射到 canonical 名称。

```json
"network_train_defaults": {
  "newnet": {
    "optimizer_type": "sgd",
    "init_lr": 0.01,
    "min_lr_ratio": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005
  }
}
```

绑定默认优化器和学习率。不要让新网络回退到 `sstnet` 的默认值。

```json
"network_presets": {
  "newnet": {
    "base_size": 512,
    "batch_size": 4,
    "epochs": 100,
    "stride": 8,
    "num_classes": 1
  }
}
```

保存网络专属默认参数。

```json
"network_train_scripts": {
  "newnet": "train/train_NewNet_det.py"
}
```

如果新网络不走序列默认训练脚本，需要绑定专属训练脚本。

### 5.2 `configs/train_experiment_config.json`

这是训练覆盖配置，适合放本次实验要临时调整的值。

常用字段：

- `dataset`
- `network`
- `train_model_path`
- `train_output_root`
- `optimizer_type`
- `init_lr`
- `min_lr`
- `momentum`
- `weight_decay`
- `eval_period`
- `eval_confidence`
- `eval_nms_iou`
- `dataset_root_prefix`
- `train_txt_path`
- `val_txt_path`
- `letterbox_train`
- `box_precision`

默认 box 精度：

```text
--box-precision high
```

含义：

- 单帧 DETLAB dataset：box geometry 使用 `float`
- 序列 dataset：使用 `float-copy`

截断式消融：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --box-precision truncate
```

含义：

- 单帧 geometry 变为 int 截断
- 序列 dataset 变为 legacy 截断行为

### 5.3 `configs/predict_experiment_config.json`

这是预测/评估覆盖配置。

常用字段在 `predict` 下：

- `json_path`
- `dataset_img_path`
- `model_path`
- `classes_path`
- `output_dir`
- `predict_output_root`
- `input_size`
- `num_classes`
- `confidence`
- `nms_iou`
- `output_mode`
- `letterbox`
- `run_eval`
- `save_failed_images`

`output_mode` 可选：

- `eval_only`
- `vis_only`
- `all`

默认建议保持：

```json
"output_mode": "eval_only",
"letterbox": true,
"run_eval": true
```

## 6. 训练入口扩展点

核心文件：

- `run_train_experiment.py`

新增网络时至少检查：

1. 是否在 `configs/experiment_presets.json` 注册。
2. `resolve_pretrained()` 能否找到默认权重。
3. `resolve_train_hparams()` 能否找到默认训练超参。
4. `resolve_train_script()` 是否能找到训练脚本。
5. 是否需要新增 `is_xxx_network()` 分支。
6. 是否需要下发专属环境变量。

已有网络族分支：

- sequence 系列：`sstnet`、`tridos`、`slowfastnet`、`slowfastnet_9520`
- ACM 系列：`acm_fpn`、`acm_unet`、`acm_unet_saliency`
- DNA 系列：`dnanet`、`dnanet_saliency`
- UIU 系列：`uiunet`、`uiunet_saliency`
- ALC 系列：`alcnet`、`alcnet_saliency`

如果新增网络可以复用某个已有训练脚本，优先放入该族分支；如果参数差异大，再新增独立训练脚本和 `is_newnet_network()`。

## 7. 预测入口扩展点

核心文件：

- `run_predict_experiment_all.py`
- `infer/predict_from_coco_json.py`

新增网络时至少检查：

1. `run_predict_experiment_all.py` 是否识别该网络。
2. 是否需要在预测命令中追加专属参数。
3. `infer/predict_from_coco_json.py` 是否能实例化模型。
4. 是否有对应 decode / nms / resize 后处理。
5. COCO 输出是否为标准格式：

```json
{
  "image_id": 1,
  "category_id": 1,
  "bbox": [x, y, w, h],
  "score": 0.9
}
```

预测输出目录默认形如：

```text
result/predict/<dataset>/<network>/<run_tag>/<timestamp>/
```

自动选权重逻辑依赖训练目录：

```text
logs/<dataset>/<network>/loss_*/best_epoch_weights.pth
logs/<dataset>/<network>/loss_*/epXXX-lossX.XXX-val_lossX.XXX.pth
```

所以新增网络不要自定义完全不同的保存结构。

## 8. 代码目录职责

新增网络时推荐按以下位置放文件。

### 8.1 `nets/`

模型结构：

```text
nets/newnet/
  model.py
  detection.py
  det_loss.py
  det_bbox.py
```

如果是单帧检测网络，建议参考：

- `nets/acm/`
- `nets/alc/`
- `nets/uiu/`
- `nets/dna/`

如果是序列网络，参考：

- `nets/sstnet.py`
- `nets/slowfastnet_9520.py`

统一构建入口：

- `nets/factory.py`

如果训练脚本或预测脚本通过 `build_network()` 创建模型，要在这里注册。

### 8.2 `train/`

训练脚本：

```text
train/train_NewNet_det.py
```

优先要求：

- 从环境变量读取路径和超参。
- 不硬编码数据集路径。
- 使用 `LossHistory` 保存 loss。
- 保存 `best_epoch_weights.pth` 和 `last_epoch_weights.pth`。
- 训练期 mAP 使用 COCO json。
- 支持 `eval_period`。
- 首轮评估一次，便于排查评估链路。

现有参考：

- `train/train_sequence_det.py`
- `train/train_ACM_det.py`
- `train/train_ALCNet_det.py`
- `train/train_DNANet_det.py`
- `train/train_UIUNet_det.py`

### 8.3 `utils/`

数据读取和工具：

```text
utils/newnet/
  dataset.py
  bbox.py
```

如果网络可以使用通用单帧 DETLAB txt dataset，优先复用：

- `utils/acm/data_detlab.py`

该 dataset 当前支持：

- 高精度 float box，默认
- int 截断消融，需显式开启
- `mosaic`
- `mixup`
- `letterbox_train`
- 路径根目录解析


### 8.4 `infer/`

统一预测脚本：

- `infer/predict_from_coco_json.py`

新增网络时应在这里新增 predictor 分支，而不是另写一套评估脚本。

### 8.5 `configs/`

稳定默认值放：

- `configs/experiment_presets.json`

一次实验覆盖值放：

- `configs/train_experiment_config.json`
- `configs/predict_experiment_config.json`

不要把临时实验参数写死到训练脚本常量里。

### 8.6 `data_txt/` 与 `data_json/`

数据索引：

```text
data_txt/<DATASET>_train.txt
data_txt/<DATASET>_val.txt
data_json/<DATASET>_train.json
data_json/<DATASET>_test.json
```

命名需要和 `dataset_presets` 对齐。

## 9. 新增网络标准流程

### Step 1: 放模型代码

把模型结构放到 `nets/<network>/`，并确认 forward 输出能被 loss 和预测 decode 使用。

### Step 2: 决定训练脚本

三种选择：

1. 序列网络：复用 `train/train_sequence_det.py`
2. 单帧检测网络：参考 `train/train_ACM_det.py` / `train/train_ALCNet_det.py`
3. 特殊网络：新增 `train/train_<Network>_det.py`

### Step 3: 注册配置

在 `configs/experiment_presets.json` 加：

- `network_pretrained_map`
- `network_alias`
- `network_train_defaults`
- `network_presets`
- `network_train_scripts`

### Step 4: 接入训练入口

如果是新网络族，修改 `run_train_experiment.py`：

- 新增 `is_newnet_network()`
- 在 `resolve_train_script()` 中绑定训练脚本
- 在 `run_train()` 中读取 `network_presets`
- 下发 `NEWNET_*` 环境变量
- 打印关键 resolved config

### Step 5: 接入预测入口

修改 `run_predict_experiment_all.py`：

- 新增网络识别函数
- 按网络类型追加预测参数
- 保持 `--json_path`、`--dataset_img_path`、`--model_path`、`--input_size`、`--confidence`、`--nms_iou` 统一

修改 `infer/predict_from_coco_json.py`：

- 新增 predictor 或复用已有 predictor
- 加载模型
- decode 输出
- nms
- 还原到原图坐标
- 写 COCO det json

### Step 6: dry-run

训练：

```bash
python run_train_experiment.py --dataset IRDST --network newnet --dry-run
```

预测：

```bash
python run_predict_experiment_all.py --dataset IRDST --network newnet --dry-run
```

检查输出中至少包含：

- dataset
- network
- train script
- train txt / val txt
- eval json / image root
- model path
- input size
- batch size
- optimizer / lr
- output dir

### Step 7: 跑第 1 个 epoch

确认：

- 能读图
- 能读标注
- loss 非 NaN
- 能保存权重
- 首轮评估不崩
- `epoch_001_metrics.json` 正常保存

### Step 8: 用预测脚本复评

训练期评估和预测期评估要能对齐：

```bash
python run_predict_experiment_all.py \
  --dataset IRDST \
  --network newnet \
  --weight-policy loss \
  --confidence 0.001 \
  --nms-iou 0.65 \
  --output-mode eval_only
```

## 10. 默认精度与消融开关

当前默认是高精度：

```text
Box precision: high (single-frame geometry=float, sequence=float-copy)
```

适用：

- `ACM`
- `ALC`
- `UIU`
- 序列 `SST/TRIDOS/SlowFast`

显式切换截断式：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --box-precision truncate
```

旧兼容参数仍可用，但不建议新实验优先使用：

- `--box-geometry-dtype`
- `--sequence-box-mode`

文档和论文记录中建议统一写：

```text
box_precision=high
box_precision=truncate
```

## 11. Dataset 选择

单帧网络固定使用 DETLAB dataset：

```text
utils/acm/data_detlab.py
```

适用网络：

- `ACM`
- `ALC`
- `UIU`

项目中不再保留 Dataset 实现切换入口。需要做 box 几何精度消融时，只使用统一的 `--box-precision` 参数：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --box-precision high
python run_train_experiment.py --dataset IRDST --network alcnet --box-precision truncate
```

说明：

- `high` 是默认值，保留 float 几何精度。
- `truncate` 只模拟 box 几何截断，不切换 Dataset 实现。
- 新增单帧网络默认也应复用 `utils/acm/data_detlab.py`，除非网络输入格式确实无法兼容。

## 12. 训练与评估数据口径

训练期有两条数据链：

```text
train/val loss -> txt
mAP eval       -> json
```

预测期：

```text
predict/eval -> json + dataset_img_path
```

因此新增数据集时必须同时检查：

- txt 能训练
- json 能评估
- txt 路径和 json `file_name` 指向同一批图片
- `dataset_img_path` 拼接后能找到图

## 13. 训练输出规范

训练输出根目录：

```text
logs/<dataset>/<network>/
```

单次训练子目录：

```text
logs/<dataset>/<network>/loss_YYYY_MM_DD_HH_MM_SS_pidXXXX/
```

必须尽量保持：

- `best_epoch_weights.pth`
- `last_epoch_weights.pth`
- `epXXX-lossX.XXX-val_lossX.XXX.pth`
- `epoch_loss.txt`
- `epoch_val_loss.txt`
- `epoch_loss.png`
- `epoch_map.txt`
- `epoch_map.png`
- `epoch_XXX_metrics.json`

`run_predict_experiment_all.py --weight-policy loss` 依赖这些权重命名和目录结构。

## 14. 预测输出规范

预测输出根目录：

```text
result/predict/<dataset>/<network>/<run_tag>/<timestamp>/
```

建议保存：

- `eval_results.json`
- COCO summary
- failed prediction image list
- 可选可视化图片

PR 曲线相关导出由工具脚本读取预测输出，不建议训练脚本重复生成一套格式。

## 15. 常见问题

### 15.1 COCO `KeyError: 'info'`

原因：gt json 缺少 `info`。

解决：使用：

```python
from utils.coco_compat import ensure_coco_dataset_compat
```

并在 `coco.loadRes()` 前兼容。

### 15.2 训练能读图，评估读不到图

通常是 txt loader 和 json eval 使用了不同路径解析。

检查：

- `default_train_txt`
- `default_val_txt`
- `default_predict.json_path`
- `default_predict.dataset_img_path`
- `dataset_root_prefix`

### 15.3 指标异常高或异常低

优先排查：

- 训练期 eval json 是否和预测期一致
- `letterbox` 是否一致
- `confidence` / `nms_iou` 是否一致
- box 是否被 int 截断
- dataset 是否仍为当前统一的 DETLAB dataset
- 权重选择策略是否一致

### 15.4 新网络没跑到预期训练脚本

检查：

- `network_alias`
- `network_train_scripts`
- `resolve_train_script()`
- `run_train_experiment.py --dry-run`

### 15.5 服务器还在跑旧文件

当前正式流程不应打印历史备份、临时消融脚本或已删除目录路径。

如果服务器日志和本地 dry-run 不一致，至少检查：

- `configs/experiment_presets.json`
- `train/train_ACM_det.py`
- `train/train_DNANet_det.py`
- `utils/acm/data_detlab.py`
- `nets/acm/det_loss.py`
- `run_train_experiment.py`

## 16. 新增网络合入前检查表

- [ ] canonical 网络名已确定
- [ ] `network_pretrained_map` 已注册
- [ ] `network_alias` 已注册
- [ ] `network_train_defaults` 已注册
- [ ] `network_presets` 已注册
- [ ] 如需专属训练脚本，`network_train_scripts` 已注册
- [ ] `run_train_experiment.py --dry-run` 输出正确
- [ ] `run_predict_experiment_all.py --dry-run` 输出正确
- [ ] 训练脚本不硬编码数据路径
- [ ] 训练脚本支持 `MODEL_PATH`
- [ ] 训练脚本支持 eval json / image root
- [ ] 预测脚本支持该网络
- [ ] COCO det json 格式正确
- [ ] 首轮训练评估通过
- [ ] 预测脚本复评通过
- [ ] 输出目录符合 `logs/<dataset>/<network>/...`
- [ ] 没有引入新的临时别名、备份文件或消融目录依赖

## 17. 推荐维护原则

- 稳定默认值放 `experiment_presets.json`。
- 临时实验值放 train/predict config 或命令行。
- 新网络先接训练入口，再接预测入口，最后接可视化/PR 工具。
- 数据路径统一从 `dataset_presets` 和 `dataset_root_prefix` 解析。
- 不在 txt/json 中混用大小写、补零与非补零帧名。
- 默认保持高精度 box，截断式只作为显式消融。
- 每次新增网络都跑一次 `--dry-run` 和第 1 epoch。

## 18. Mask / Saliency 网络改检测监督

输出 mask / saliency map 的网络接入检测监督时，请参考：

```text
docs/MASK_TO_DETECTION_GUIDE.md
```
