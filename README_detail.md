# MIRSTD-Lab 使用说明

如果是第一次接手项目，建议先阅读根目录的 [README.md](README.md) 了解项目总览、目录职责和文档导航；本文件更偏详细使用手册。

本项目已经整理为统一的训练入口和预测/评估入口：

- 训练：`run_train_experiment.py`
- 预测/评估：`run_predict_experiment_all.py`
- 稳定默认配置：`configs/experiment_presets.json`
- 训练覆盖配置：`configs/train_experiment_config.json`
- 预测覆盖配置：`configs/predict_experiment_config.json`

建议所有正式实验都优先通过这两个入口运行，避免每个网络维护一套临时命令。

## 1. 环境安装

推荐使用项目中的 `environment.yml` 创建 conda 环境：

```bash
conda env create -f environment.yml
conda activate detlab_py39
```

如果环境已经存在，需要按文件更新：

```bash
conda env update -n detlab_py39 -f environment.yml --prune
```

快速验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from pycocotools.coco import COCO; import cv2; print('ok')"
```

`environment.yml` 当前固定 PyTorch 2.4.1 + CUDA 12.4，对应服务器需要有足够新的 NVIDIA 驱动。若服务器已有可用环境，可以把该文件作为依赖版本参考，不必强制重建。

## 2. 项目路径与图片存储

代码项目目录示例：

```text
/project/IDIP/TWW/DETLAB
```

数据集根目录通常放在项目外层：

```text
/project/IDIP/TWW/datasets
```

因此配置里用：

```json
"global": {
  "dataset_root_prefix": "/project/IDIP/TWW"
}
```

项目中的 txt 索引通常写相对路径：

```text
datasets/IRDST/images/71/336.bmp 100,120,108,128,0
datasets/ITSDT-15k/images/1/0.bmp
```

运行时会通过 `dataset_root_prefix` 或 `dataset_img_path` 补齐为：

```text
/project/IDIP/TWW/datasets/IRDST/...
/project/IDIP/TWW/datasets/ITSDT-15k/...
```

注意：

- txt/json 里的路径大小写要和磁盘一致，例如 `images` 不要写成 `Images`。
- ITSDT 当前帧名按磁盘实际使用 `0.bmp`、`1.bmp`，不是 `001.bmp`。
- 不建议在 txt 里写死服务器绝对路径，优先让配置统一拼接。

## 3. 数据文件

训练 loss 使用 txt：

```text
data_txt/<DATASET>_train.txt
data_txt/<DATASET>_val.txt
```

训练期 mAP 和预测评估使用 COCO json：

```text
data_json/<DATASET>_test.json
```

当前数据集配置在 `configs/experiment_presets.json -> dataset_presets`：

```text
DAUB
IRDST
ITSDT_15K
```

例如 `ITSDT_15K`：

```text
train loss: data_txt/ITSDT_15k_train.txt
val loss:   data_txt/ITSDT_15k_val.txt
mAP eval:   data_json/ITSDT_15k_test.json
image root: datasets/ITSDT-15k
```

## 4. 当前网络名

统一入口中可用的 canonical 网络名：

```text
sstnet
tridos
slowfastnet
slowfastnet_9520
acm_fpn
acm_unet
acm_unet_saliency
dnanet
dnanet_saliency
uiunet
uiunet_saliency
alcnet
alcnet_saliency
sctransnet
dqaligner
dqaligner_saliency
```

新增网络时应同步更新 `configs/experiment_presets.json` 和必要的训练/预测分支。详细接入规范见：

```text
docs/NETWORK_INTEGRATION_CHECKLIST.md
```

## 5. 训练入口

基本命令：

```bash
python run_train_experiment.py --dataset IRDST --network sstnet
python run_train_experiment.py --dataset ITSDT_15K --network tridos
python run_train_experiment.py --dataset DAUB --network alcnet
python run_train_experiment.py --dataset IRDST --network sctransnet
```

先检查解析结果，不真正训练：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --dry-run
```

常用参数：

```text
--config                  训练配置文件，默认 configs/train_experiment_config.json
--presets                 总预设文件，默认 configs/experiment_presets.json
--dataset                 数据集名，例如 DAUB / IRDST / ITSDT_15K
--network                 网络名，例如 sstnet / tridos / alcnet
--model-path              指定初始化权重；覆盖默认预训练权重
--optimizer               覆盖优化器：sgd / adam / adagrad
--init-lr                 覆盖初始学习率
--min-lr                  覆盖最小学习率
--momentum                覆盖 momentum
--weight-decay            覆盖 weight decay
--epochs                  覆盖训练轮数
--eval-period             覆盖训练期评估周期
--dataset-root-prefix     覆盖数据根前缀
--letterbox-train         单帧 DETLAB dataset 使用 letterbox 训练
--box-coord-type          DETLAB txt 坐标解析类型：float / int
--box-precision           box 几何精度：high / truncate
--dry-run                 只打印解析结果
```

默认 box 精度是高精度：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --box-precision high
```

截断式消融：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --box-precision truncate
```

截断式是sst等项目使用的旧reszie方式，我们怀疑是对将resize后的目标坐标进行了失误截断导致在IRDST上的检测结果较低，故而我们进行了精度修复，通过 `--box-precision` 控制高精度或截断式。

## 6. 预测与评估入口

基本命令：

```bash
python run_predict_experiment_all.py --dataset IRDST --network sstnet
python run_predict_experiment_all.py --dataset IRDST --network tridos --weight-policy loss
python run_predict_experiment_all.py --dataset DAUB --network alcnet --output-mode eval_only
python run_predict_experiment_all.py --dataset IRDST --network sctransnet --weight-policy loss
```

指定权重：

```bash
python run_predict_experiment_all.py \
  --dataset IRDST \
  --network alcnet \
  --model-path logs/IRDST/alcnet/loss_xxx/best_epoch_weights.pth
```

指定 GT json：

```bash
python run_predict_experiment_all.py \
  --dataset IRDST \
  --network alcnet \
  --json-path data_json/IRDST_test.json
```

一键跑最新权重：

```bash
python run_predict_experiment_all.py \
  --all-latest \
  --datasets DAUB IRDST \
  --networks sstnet tridos alcnet \
  --weight-policy loss \
  --confidence 0.001 \
  --nms-iou 0.65
```

常用参数：

```text
--config                  预测配置文件，默认 configs/predict_experiment_config.json
--presets                 总预设文件，默认 configs/experiment_presets.json
--dataset                 数据集名
--network                 网络名
--model-path              显式指定权重，优先级最高
--json-path / --gt-json   指定 COCO GT json
--dataset-img-path        指定图片根目录
--dataset-root-prefix     覆盖数据根前缀
--train-output-root       训练输出根目录，默认 logs
--weight-policy           自动选权重策略：ap50 / ap50:95 / loss / last
--all-latest              批量跑所有有权重的数据集/网络组合
--networks                配合 --all-latest 指定网络列表
--datasets                配合 --all-latest 指定数据集列表
--output-mode             all / vis_only / eval_only
--confidence              检测置信度阈值
--nms-iou                 NMS IoU 阈值
--vis-confidence          可视化阈值
--vis-max-boxes           单图最大绘制框数，0 表示不限制
--save-failed-images      保存评估失败图片列表
--failed-iou              判断 failed image 的 IoU 阈值
--batch-size              序列网络预测 batch size
--run-tag                 自定义输出分组名
--strict                  all-latest 模式下缺权重时报错
--dry-run                 只打印解析命令
```

`--weight-policy` 说明：

```text
loss      使用 best_epoch_weights.pth，通常对应最低 val_loss
last      使用 last_epoch_weights.pth 或最新 epoch 权重
ap50      使用训练期评估 AP@0.50 最好的 epoch
ap50:95   使用训练期评估 AP@[0.50:0.95] 最好的 epoch
```

## 7. 配置文件说明

### 7.1 `configs/experiment_presets.json`

稳定默认配置，建议长期维护。

主要字段：

```text
dataset_presets          数据集 txt/json/image_root 配置
network_pretrained_map   网络默认预训练权重
network_alias            网络别名到 canonical 名称的映射
network_train_defaults   网络默认优化器、学习率、weight decay
network_presets          网络专属参数，例如 batch_size/base_size/stride
network_train_scripts    网络到训练脚本的映射
global.dataset_root_prefix 数据根前缀
```

新增数据集主要改 `dataset_presets`。

新增网络通常要改：

```text
network_pretrained_map
network_alias
network_train_defaults
network_presets
network_train_scripts
```

### 7.2 `configs/train_experiment_config.json`

训练覆盖配置。命令行参数优先级高于该文件。

常用字段：

```text
dataset
network
train_model_path
train_output_root
optimizer_type
init_lr
min_lr
momentum
weight_decay
eval_period
eval_confidence
eval_nms_iou
eval_json_path
eval_image_root
train_txt_path
val_txt_path
dataset_root_prefix
```

网络专属字段示例：

```text
acm_base_size / acm_batch_size / acm_epochs
alc_base_size / alc_batch_size / alc_epochs
dna_base_size / dna_batch_size_by_dataset
uiu_batch_size / uiu_fp16
```

### 7.3 `configs/predict_experiment_config.json`

预测覆盖配置。命令行参数优先级高于该文件。

核心字段在 `predict` 下：

```text
json_path
dataset_img_path
model_path
classes_path
output_dir
predict_output_root
input_size
confidence
nms_iou
vis_confidence
vis_max_boxes
num_frame
letterbox
output_mode
run_eval
save_failed_images
failed_iou
```

默认建议：

```json
"output_mode": "eval_only",
"letterbox": true,
"run_eval": true,
"save_failed_images": true
```

## 8. 结果保存

训练输出默认在：

```text
logs/<dataset>/<network>/loss_YYYY_MM_DD_HH_MM_SS_pidXXXX/
```

常见文件：

```text
best_epoch_weights.pth
last_epoch_weights.pth
epXXX-lossX.XXX-val_lossX.XXX.pth
epoch_loss.txt
epoch_val_loss.txt
epoch_loss.png
epoch_map.txt
epoch_map.png
epoch_XXX_metrics.json
```

预测输出默认在：

```text
result/predict/<dataset>/<network>/<run_tag>/<timestamp_pid>/
```

`run_tag` 默认由权重策略、confidence、nms 生成，例如：

```text
wp-loss_conf0p001_nms0p65
```

常见文件：

```text
eval_results.json
failed_prediction_images_iou0p50.txt
可视化图片，取决于 output_mode
COCO 评估输出，取决于 run_eval
```

PR/ROC 相关统计文件通常由 `tools/` 下的导出脚本基于预测目录生成，不建议训练脚本重复保存另一套格式。

## 9. 推荐工作流

1. 先 dry-run：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --dry-run
```

2. 正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 python run_train_experiment.py --dataset IRDST --network alcnet
```

3. 用预测脚本复评：

```bash
python run_predict_experiment_all.py \
  --dataset IRDST \
  --network alcnet \
  --weight-policy loss \
  --confidence 0.001 \
  --nms-iou 0.65 \
  --output-mode eval_only
```

4. 如果训练期评估和预测期评估差异很大，优先核对：

```text
GT json 是否一致
dataset_img_path 是否一致
input_size 是否一致
letterbox 是否一致
confidence / nms_iou 是否一致
weight-policy 是否选到了同一权重
box_precision 是否一致
```

## 10. 常见注意事项

- 训练 loss 用 txt，mAP 评估用 json，两者需要对应同一批数据。
- 单帧网络 `ACM/ALC/UIU` 固定使用 `utils/acm/data_detlab.py`。
- 默认 box 几何是高精度 float；需要截断式实验时显式加 `--box-precision truncate`。
- 序列网络 `sstnet/tridos/slowfastnet` 使用 5 帧输入，显存占用高于单帧网络。
- `run_predict_experiment_all.py --all-latest` 会根据训练输出目录自动找权重。
- 上传服务器时要同步 `configs/`、`run_train_experiment.py`、`run_predict_experiment_all.py`、对应 `train/`、`infer/`、`utils/` 和 `nets/` 文件。
## 11. 结果汇总与可视化

预测完成后，可以使用 `tools/visualize_predict_metrics.py` 对 `result/predict/` 下的多个数据集、多个网络结果做统一汇总。

常用命令：

```bash
python tools/visualize_predict_metrics.py \
  --root result/predict \
  --select latest \
  --run-tag auto \
  --weight-policy loss \
  --confidence 0.001 \
  --nms-iou 0.65
```

如果只想快速生成指标排名图，不重新计算 PR/ROC 曲线：

```bash
python tools/visualize_predict_metrics.py \
  --root result/predict \
  --select latest \
  --run-tag auto \
  --weight-policy loss \
  --confidence 0.001 \
  --nms-iou 0.65 \
  --no-pr \
  --no-roc
```

输出目录默认位于：

```text
result/predict/_summary/<timestamp>_<select>_iou0p5_<run_tag>/
```

当前总览图不再输出旧版的密集图：

```text
latest_metrics_bars.png
latest_metrics_heatmap.png
```

而是按指标分别输出更清晰的“数据集分面 + 网络横向排名图”：

```text
map50_95_ranked_by_dataset.png / .pdf / .csv
map50_ranked_by_dataset.png / .pdf / .csv
map75_ranked_by_dataset.png / .pdf / .csv
precision_ranked_by_dataset.png / .pdf / .csv
recall_ranked_by_dataset.png / .pdf / .csv
f1_ranked_by_dataset.png / .pdf / .csv
```

这些图的特点：

- 每个数据集单独一个分区，避免网络数量多时挤在一起。
- 每个分区内部按当前指标从高到低排序。
- 横轴统一显示百分比，柱子上直接标出数值。
- 同时保存 `.csv`，方便复制到论文表格或进一步画图。

仍会保留基础汇总表：

```text
all_runs_metrics.csv
latest_metrics.csv
run_info.json
```

如果没有使用 `--no-pr / --no-roc`，并且预测目录里存在 `eval_results.json`，脚本还会继续生成 PR、ROC/FROC 相关曲线和 CSV。
