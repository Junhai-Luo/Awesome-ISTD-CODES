# MIRSTD-Lab 项目概念说明

**MIRSTD-Lab**（Multi-network Infrared Small Target Detection Lab）是一个面向红外小目标检测的统一实验框架。

红外小目标领域中，很多经典网络原本并不是标准目标检测器，一些方法为掩码输出的分割式网络。MIRSTD-Lab 的目标就是把这些差异收束到统一检测框架中：所有网络最终都以 bbox detector 的形式训练、预测和评估。

这个项目的核心工作不是单纯收集多个网络，而是把原本来源不同、监督形式不同、输出形式不同的红外小目标方法，整理到同一个检测实验体系中，使它们可以在相同数据划分、相同输入尺寸、相同预测脚本、相同 COCO 检测指标下进行对比。

项目目前收录包括：

- 将 `SSTNet / Tridos` 等检测式或序列式网络纳入统一训练与预测入口。
- 将 `ACM / ALCNet / DNANet / UIUNet / SCTransNet / DQAligner` 等原本更接近分割、显著性或 mask 输出的网络重新架构为检测网络。
- 为所有网络统一使用 bbox 检测监督、统一输出 `[reg, obj, cls]` 检测头格式。
- 使用同一套 COCO bbox 评估流程计算 `mAP50:95 / mAP50 / mAP75 / Precision / Recall / F1 / AR`。
- 提供数据输入几何、box 精度、PR 曲线、复杂度和特征图热力图等分析工具。


## 主要工作一：分割式网络检测化

项目中一个重要工作，是把原本偏分割/显著性检测的网络改造成 bbox 检测网络。例如：

```text
ACM
ALCNet
DNANet
UIUNet
SCTransNet
DQAligner
```

这些网络的原始形式通常是：

```text
image / sequence -> backbone / encoder-decoder -> mask or saliency map
```

在本项目中，它们被改造成：

```text
image / sequence -> 原网络特征提取部分 -> detection head -> [reg, obj, cls]
```

也就是说，原网络不再直接输出 mask 作为最终结果，而是作为检测特征提取器，后接 YOLOX 风格的单尺度检测头，输出 bbox 回归、目标置信度和类别预测。

对于分割式或显著性网络，本项目主要保留两种重新架构方式。

### 方式 A：Feature 检测化

Feature 模式直接取原网络中较深层或融合后的 feature map，接入检测头。

概念形式：

```text
Backbone / Encoder-Decoder feature    -> detection head    -> bbox predictions
```

特点：

- 更像“把原网络当作特征提取器”。
- 检测头直接学习从深层特征到 bbox 的映射。
- 适合原网络内部已有较稳定的多层特征融合结构。

项目中的例子：

```text
acm_fpn
acm_unet
alcnet
dnanet
uiunet
dqaligner
```

### 方式 B：Saliency 检测化

Saliency 模式更强调原网络的显著性响应或 mask-like 输出路径。它通常取原网络接近 saliency/mask 输出前后的特征，再接检测头。

概念形式：

```text
Saliency / mask-oriented feature
    -> detection head
    -> bbox predictions
```

特点：

- 保留原分割/显著性网络对小目标响应区域的建模方式。
- 不直接把 mask 当最终输出，而是把 saliency 分支作为检测特征来源。
- 适合原始网络本身对红外小目标定位响应较强的结构。

项目中的例子：

```text
acm_unet_saliency
dnanet_saliency
uiunet_saliency
alcnet_saliency
dqaligner_saliency
```

前者是对检测后者更还原原分割网络的输出

## 统一后的检测输出

所有改造后的网络最终都对齐为检测头输出：

```text
[reg, obj, cls]
```
其中：

- `reg`：bbox 回归预测，通常为 4 个通道。
- `obj`：目标存在置信度。
- `cls`：类别预测，本项目多数实验为单类别红外小目标。

预测阶段统一经过：

```text
decode outputs -> confidence filtering -> NMS -> COCO detection json
```

最终统一采用 COCO bbox 检测指标。

核心指标包括：

```text
AP@[0.50:0.95]   即 mAP50:95
AP@0.50          即 mAP50 / AP50
AP@0.75          即 mAP75 / AP75
Precision@0.5
Recall@0.5
F1@0.5
AR@100
```

## 统一数据和输入几何

项目当前围绕三个数据集组织：[DAUB](https://www.scidb.cn/en/detail?dataSetId=720626420933459968),[IRDST](https://xzbai.buaa.edu.cn/datasets.html),[ITSDT_15K](https://www.scidb.cn/en/detail?dataSetId=de971a1898774dc5921b68793817916e&dataSetType=journal).

项目中配置文件参考UESTC-nnlab调整后版本设置，以便训练对齐
特别注明DAUB下载后需调整文件夹命名，也可直接使用UESTC-nnlab调整后版本[DAUB](https://pan.baidu.com/s/1nNTvjgDaEAQU7tqQjPZGrw?from=init&pwd=saew), [IRDST](https://pan.baidu.com/s/1igjIT30uqfCKjLbmsMfoFw?pwd=rrnr#list/path=%2F),[ITSDT-15K](https://drive.google.com/file/d/1nnlXK0QCoFqToOL-7WdRQCZfbGJvHLh2/view)。

训练和评估输入统一到当前配置中的尺寸，默认主要为：

```text
512 x 512
```

项目中特别保留了 box 精度控制：

```bash
--box-precision high
--box-precision truncate
```

我们复现SSTNet / Tridos等网络时,发现其原代码可能在resize图片存在失误，进行了int截断。我们修正数据格式为float即默认使用高精度 box，truncate是旧版int截断形式。

## 权重文件说明

为了避免 GitHub 仓库体积过大，上传代码仓库时不随仓库上传预训练权重、训练权重和预测结果。`model_data/` 中的大权重文件（如 `.pth`、`.pt`、`.tar`、`.ckpt`）会在上传前删除或通过 `.gitignore` 排除；仓库中只保留必要的轻量配置文件，例如类别文件。

权重下载链接：[预训练权重](https://pan.quark.cn/s/2eb95ce26b72?pwd=qsjv
)提取码：qsjv

下载权重后，建议按 `README_detail.md` 和 `configs/experiment_presets.json` 中记录的路径放回 `model_data/`，或在训练/预测时使用 `--model-path` 显式指定权重路径。

## 代码结构和文档导航
主要代码目录：

```text
configs/      数据集、网络、训练、预测配置
data_txt/     训练和验证 txt 索引
data_json/    COCO bbox 评估 json
nets/         网络结构、检测头、loss
train/        各网络训练脚本
infer/        统一预测脚本
utils/        dataset、bbox、callback、COCO 兼容工具
tools/        实验分析工具
docs/         新网络接入和检测化改造说明
```
详细文档：

- [README_detail.md](README_detail.md)：环境、路径、训练、预测、配置、结果保存的详细说明。
- [tools/README.md](tools/README.md)：所有工具脚本的用法。
- [docs/NETWORK_INTEGRATION_CHECKLIST.md](docs/NETWORK_INTEGRATION_CHECKLIST.md)：新增网络/数据集时的工程接入规范。
- [docs/MASK_TO_DETECTION_GUIDE.md](docs/MASK_TO_DETECTION_GUIDE.md)：分割/显著性网络改造成检测监督网络的细节。
