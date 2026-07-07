# MIRSTD-Lab Tools 使用说明

`tools/` 目录用于放置训练/预测主流程之外的辅助脚本。当前正式工具包括：

- `compare_dataset_outputs.py`
- `export_eval_pr_stats.py`
- `export_feature_maps.py`
- `visualize_predict_metrics.py`
- `profile_networks.py`

这些工具默认围绕统一目录结构工作：

```text
data_txt/
data_json/
configs/
logs/
result/predict/
result/
```

建议先通过 `--help` 查看参数：

```bash
python tools/export_eval_pr_stats.py --help
python tools/export_feature_maps.py --help
python tools/visualize_predict_metrics.py --help
python tools/profile_networks.py --help
python tools/compare_dataset_outputs.py --help
```

## 1. `compare_dataset_outputs.py`

用途：

- 对比 DETLAB dataset 在高精度 box 和截断式 box 下的输出差异。
- 用于复查 `--box-precision high` 与 `--box-precision truncate` 对输入图像、box 几何的影响。
- 当前项目统一使用 DETLAB dataset，本工具只比较高精度 box 与截断式 box。

常用命令：

```bash
python tools/compare_dataset_outputs.py \
  --train-txt data_txt/IRDST_train.txt \
  --image-root /project/IDIP/QXY/datasets/IRDST \
  --input-size 512 \
  --mode letterbox \
  --indices 0,1,2,10,100,1000 \
  --out-dir result/dataset_compare
```

训练增强模式对比：

```bash
python tools/compare_dataset_outputs.py \
  --train-txt data_txt/IRDST_train.txt \
  --image-root /project/IDIP/QXY/datasets/IRDST \
  --mode mosaic \
  --seed 2026
```

主要参数：

```text
--train-txt             训练 txt 索引
--image-root            图片根目录
--input-size            输入尺寸，默认 512
--indices               指定样本序号
--mode                  letterbox / random-single / mosaic
--box-coord-type        原始坐标解析类型：float / int
--out-dir               输出目录
```

输出：

```text
result/dataset_compare/dataset_precision_compare_summary.csv
result/dataset_compare/dataset_precision_compare_details.json
```

注意：

- `letterbox` 模式最适合复查几何精度问题。
- `mosaic` 模式涉及随机增强，建议固定 `--seed`。

## 2. `export_eval_pr_stats.py`

用途：

- 从预测输出目录中的 `eval_results.json` 读取检测结果。
- 结合 COCO GT json 计算 PR 曲线点。
- 导出画 PR 图所需的 `pr_curve.json`。
- 另存 TP/FP/FN 和 score median 等统计信息。

处理单个预测目录：

```bash
python tools/export_eval_pr_stats.py \
  --run-dir result/predict/IRDST/alcnet/wp-loss_conf0p001_nms0p65/202605xx_xxxxxx_pidxxxx \
  --gt-json data_json/IRDST_test.json
```

处理单个 eval json：

```bash
python tools/export_eval_pr_stats.py \
  --eval-json result/predict/IRDST/alcnet/.../eval_results.json \
  --gt-json data_json/IRDST_test.json
```

一键处理最新预测结果：

```bash
python tools/export_eval_pr_stats.py \
  --all-latest \
  --datasets DAUB IRDST ITSDT_15K \
  --networks sstnet tridos alcnet \
  --weight-policy loss \
  --confidence 0.001 \
  --nms-iou 0.65
```

常用参数：

```text
--predict-root          预测根目录，默认 result/predict
--presets               数据集预设，用于推断 GT json
--eval-json             直接处理一个 eval_results.json
--run-dir               直接处理一个预测 run 目录
--gt-json / --gt_json   指定 GT COCO json
--gt-map                数据集到 GT json 的映射，例如 DAUB=data_json/DAUB_test.json
--dataset / --network   指定单个数据集/网络
--datasets / --networks 批量指定
--weight-policy         run-tag 中的权重策略标签
--confidence            run-tag 中的 confidence
--nms-iou               run-tag 中的 nms
--run-tag               精确指定 run-tag
--all                   处理所有匹配 eval_results.json
--all-latest            每个 dataset/network 处理最新匹配结果
--latest-per-run-tag    每个 run-tag 保留最新结果
--iou                   TP 匹配 IoU，默认 0.5
--class-agnostic        匹配时忽略 category_id
--summary-csv           指定全局 summary csv 路径
--dry-run               只显示将处理哪些文件
```

典型输出在预测 run 目录下：

```text
pr_curve.json
pr_stats_iou0p50.json
pr_stats_summary_iou0p50.csv
```

说明：

- 画 PR 曲线只需要 `precision / recall / score_threshold` 等曲线点。
- TP/FP/FN、TP score median、FP score median 属于统计信息，单独保存。

## 3. `visualize_predict_metrics.py`

用途：

- 汇总 `result/predict/` 下多个网络/数据集的预测结果。
- 绘制 AP/AR 指标柱状图或折线图。
- 可选绘制 PR 曲线、ROC/FROC 曲线。

基础命令：

```bash
python tools/visualize_predict_metrics.py \
  --root result/predict \
  --select latest
```

按 run-tag 过滤：

```bash
python tools/visualize_predict_metrics.py \
  --root result/predict \
  --run-tag auto \
  --weight-policy loss \
  --confidence 0.001 \
  --nms-iou 0.65
```

只画 PR，不画 ROC：

```bash
python tools/visualize_predict_metrics.py \
  --root result/predict \
  --no-roc
```

常用参数：

```text
--root                  预测结果根目录，默认 result/predict
--output-dir            汇总图输出目录；为空时自动生成
--config                预测配置，用于 auto run-tag 默认值
--select                latest / ap50 / ap50:95 / best-ap50 / best-map
--metrics               指定要画的指标，逗号分隔
--presets               数据集预设，用于 ROC GT 查找
--roc-iou               ROC/FROC 匹配 IoU
--max-fppi              FROC 图最大 FPPI
--confidence            输出目录标签
--nms-iou               输出目录标签
--weight-policy         auto run-tag 的权重策略
--run-tag               过滤预测 run；auto 会生成 wp-*_conf*_nms*
--tag                   输出目录额外标签
--no-roc                不生成 ROC/FROC
--no-pr                 不生成 PR
--curve-points          曲线采样点数量
--quiet                 减少日志
```

常见输出：

```text
result/_summary/<timestamp>/
```

其中默认包含：

```text
all_runs_metrics.csv
latest_metrics.csv
run_info.json
```

总览图现在按指标分别输出“数据集分面 + 网络横向排名图”，不再输出旧版密集的 `latest_metrics_bars.png` 和 `latest_metrics_heatmap.png`。默认指标会生成：

```text
map50_95_ranked_by_dataset.png / .pdf / .csv
map50_ranked_by_dataset.png / .pdf / .csv
map75_ranked_by_dataset.png / .pdf / .csv
precision_ranked_by_dataset.png / .pdf / .csv
recall_ranked_by_dataset.png / .pdf / .csv
f1_ranked_by_dataset.png / .pdf / .csv
```

这些排名图每个数据集单独一个分区，每个分区内部按当前指标排序，适合网络数量较多时快速比较性能。

如果没有指定 `--no-pr` / `--no-roc`，并且预测目录中存在 `eval_results.json`，还会额外输出 PR、ROC/FROC 图和曲线数据。

## 4. `profile_networks.py`

用途：

- 统计网络参数量。
- 可选统计 MACs/FLOPs。
- 可选测试 FPS。
- 适合比较不同网络在相同输入尺寸下的复杂度。

全部网络 profile：

```bash
python tools/profile_networks.py \
  --networks all \
  --datasets IRDST \
  --input-size 512
```

只跑指定网络：

```bash
python tools/profile_networks.py \
  --networks sstnet tridos alcnet \
  --datasets IRDST \
  --batch-size 1 \
  --repeat 50
```

只统计参数和 FLOPs，不测 FPS：

```bash
python tools/profile_networks.py \
  --networks alcnet dnanet_saliency \
  --datasets IRDST \
  --no-fps
```

CPU 环境或没有 CUDA 时：

```bash
python tools/profile_networks.py \
  --networks alcnet \
  --datasets IRDST \
  --cpu \
  --no-fps
```

常用参数：

```text
--presets               网络/数据集预设
--predict-config        预测配置，用于 input size 默认值
--output                输出 csv，默认 result/profile/profile_networks.csv
--networks              网络列表，或 all
--datasets              数据集列表，或 all
--input-size            覆盖输入尺寸
--num-frame             序列网络帧数，默认 5
--batch-size            profile batch size
--num-classes           类别数
--warmup                FPS warmup 次数
--repeat                FPS repeat 次数
--cpu                   强制 CPU
--no-flops              不统计 FLOPs/MACs
--no-fps                不测试 FPS
```

输出：

```text
result/profile/profile_networks.csv
```

注意：

- FLOPs/MACs 依赖 `thop`。
- FPS 结果和 GPU、batch size、输入尺寸、当前负载强相关。
- 序列网络使用 `--num-frame` 控制时间窗口，默认与训练/预测一致为 5。

## 5. 推荐使用顺序

典型实验完成后：

1. 先用 `run_predict_experiment_all.py` 生成预测评估结果。
2. 用 `export_eval_pr_stats.py` 生成 PR 曲线与 TP/FP/FN 统计。
3. 用 `visualize_predict_metrics.py` 汇总多个网络/数据集的图。
4. 用 `profile_networks.py` 补充模型复杂度和速度。
5. 如怀疑 box 精度或增强问题，用 `compare_dataset_outputs.py` 单独排查。

## 6. 环境依赖

这些工具依赖项目主环境，推荐使用：

```bash
conda env create -f environment.yml
conda activate detlab_py39
```

轻量查看 `--help` 通常不需要完整训练数据，但实际运行会依赖：

- `torch`
- `torchvision`
- `numpy`
- `opencv-python`
- `pycocotools`
- `matplotlib`
- `thop`，仅 `profile_networks.py` 的 FLOPs/MACs 需要

## 7. 维护约定

- 新增工具必须支持 `--help`。
- 输入输出路径要默认落在项目统一目录下。
- 不要在工具里硬编码服务器绝对路径。
- 如果工具读取预测结果，默认使用 `result/predict`。
- 如果工具读取数据集配置，默认使用 `configs/experiment_presets.json`。
- 如果工具生成汇总文件，默认写入 `result/` 下的子目录。

## 8. `export_feature_maps.py`

用途：

- 在一次预测式 forward 中导出中间层特征图。
- 默认 `--layers auto` 会优先抓检测头上一层附近的特征，例如 `head.reg_convs.0`、`head.cls_convs.0`、`head.reg_conv`、`head.cls_conv`、`det_head_s8.reg_conv`、`downsample`。
- 每个层会保存整体响应热力图、叠加原图的 overlay，以及 top-k 通道特征图。

查看某个网络可 hook 的层名：

```bash
python tools/export_feature_maps.py \
  --network alcnet \
  --model-path logs/IRDST/alcnet/xxx/best_epoch_weights.pth \
  --list-layers
```

直接用图片导出自动层：

```bash
python tools/export_feature_maps.py \
  --network alcnet \
  --model-path logs/IRDST/alcnet/xxx/best_epoch_weights.pth \
  --image-path /project/IDIP/QXY/datasets/IRDST/images/xxx.bmp \
  --input-size 512 \
  --output-dir result/feature_maps
```

从 COCO json 中选择样本导出：

```bash
python tools/export_feature_maps.py \
  --network tridos \
  --model-path logs/IRDST/tridos/xxx/best_epoch_weights.pth \
  --json-path data_json/IRDST_test.json \
  --dataset-img-path /project/IDIP/QXY/datasets/IRDST \
  --index 0 \
  --input-size 512 \
  --num-frame 5
```

从 COCO json 中批量导出：

```bash
python tools/export_feature_maps.py \
  --network tridos \
  --model-path logs/IRDST/tridos/xxx/best_epoch_weights.pth \
  --json-path data_json/IRDST_test.json \
  --dataset-img-path /project/IDIP/QXY/datasets/IRDST \
  --indices 0,1,2,10-20 \
  --layers head.stems.0 \
  --max-images 16
```

从图片列表批量导出：

```bash
python tools/export_feature_maps.py \
  --network alcnet \
  --model-path logs/IRDST/alcnet/xxx/best_epoch_weights.pth \
  --image-list data_txt/IRDST_val.txt \
  --layers head.reg_conv head.cls_conv \
  --max-images 16
```

指定层导出：

```bash
python tools/export_feature_maps.py \
  --network dnanet_saliency \
  --model-path logs/IRDST/dna_saliency/xxx/best_epoch_weights.pth \
  --image-path /project/IDIP/QXY/datasets/IRDST/images/xxx.bmp \
  --layers head.reg_convs head.cls_convs \
  --topk 16
```

典型输出：

```text
result/feature_maps/<network>/<image_stem>_<timestamp>/
  input_resized.png
  feature_summary.json
  <layer_name>/
    heatmap.png
    overlay.png
    channel_rank01_cxxxx.png

result/feature_maps/<network>/feature_batch_summary_<timestamp>.json
```

注意：

- 序列网络会复用项目中的历史帧查找逻辑，默认 `--num-frame 5`。
- 默认使用 `--letterbox`，与当前预测/训练默认输入几何保持一致；可用 `--no-letterbox` 关闭。
- `--model-path` 可以不传，但那只会导出随机初始化模型的特征，通常只用于检查流程。
