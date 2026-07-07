# Mask / Saliency 网络改检测监督说明

本文说明把原本输出 mask、saliency map 或分割结果的网络接入 MIRSTD-Lab 检测体系时，常见的两种改法、命名方式和注意点。

## 1. 两种接入形式

### 1.1 Feature Detection

代表：

```text
alcnet
acm_unet
uiunet
sctransnet
dqaligner
```

做法：

```text
原始 backbone / encoder-decoder
        ->
取中间 feature map
        ->
接 YOLOX-style detection head
        ->
输出 [reg, obj, cls]
        ->
用 DETLAB YOLOLoss 做 bbox 检测监督
```

特点：

- 不使用原始 mask 输出作为最终监督。
- 不要求网络 forward 返回 saliency map。
- 检测头通常接在 stride=8 的特征层上。
- 更像“借用 mask 网络的特征提取能力”，本质是检测模型。

当前 `sctransnet` 就属于这种形式：

```python
d4 = self.up_decoder4(d5, x4)
return [self.head(d4)]
```

因此它应该叫：

```text
sctransnet
sctransnet_det
```

不建议叫 `sctransnet_saliency`。

### 1.2 Saliency Detection

代表：

```text
alcnet_saliency
acm_unet_saliency
uiunet_saliency
dnanet_saliency
dqaligner_saliency
```

做法：

```text
原始 mask / saliency 网络
        ->
保留 saliency 路径或 saliency-like decoder feature
        ->
基于 saliency 分支接检测头
        ->
输出 [reg, obj, cls]
        ->
仍然用 bbox 检测监督
```

特点：

- 名字里带 `saliency`，表示检测头更依赖原始显著性分支。
- 仍然不是用 mask loss 训练；监督目标依然是 bbox。
- 如果保留原始 mask 输出但训练时不用 mask loss，要明确这是辅助结构，不是分割监督。

## 2. 当前 DETLAB 统一约定

检测训练的统一输出应为：

```text
List[Tensor]
Tensor shape: [B, 5 + num_classes, H / stride, W / stride]
channel order: [reg_x, reg_y, reg_w, reg_h, obj, cls...]
```

检测 loss 默认使用：

```text
nets/acm/det_loss.py 或各网络对应的 DETLAB YOLOLoss
```

数据集默认使用：

```text
utils/acm/data_detlab.py
```

默认 box 几何精度：

```text
--box-precision high
```

截断式 box 只作为显式消融：

```bash
python run_train_experiment.py --dataset IRDST --network alcnet --box-precision truncate
```

## 3. 命名建议

如果只是取中间 feature 接检测头：

```text
xxx
xxx_det
```

如果明确使用 saliency 分支 / saliency decoder / mask-response 路径接检测头：

```text
xxx_saliency
```

不建议只因为原始论文是分割网络就加 `saliency`。是否加 `saliency` 应该看当前 DETLAB 代码里的检测头接法。

## 4. 接入时必须核对的点

1. 模型输出是否是 YOLOLoss 需要的 `[reg, obj, cls]`。
2. 输出 stride 是否和配置一致，通常是 `8`。
3. 训练脚本是否使用当前统一的 DETLAB Dataset。
4. 训练期 eval 和预测期 eval 是否使用同一份 COCO json。
5. 预测脚本是否走单帧检测分支，而不是序列网络分支。
6. decode / NMS 是否和训练输出格式一致。
7. 若加载原始分割预训练权重，检测头新增参数应允许 missing keys。
8. 若没有预训练权重，应允许从头训练，而不是报 FileNotFoundError。

## 5. 常见问题

### 5.1 mAP 长期为 0

优先检查：

- input size 是否训练预测一致。
- stride 是否和输出特征图大小一致。
- 检测头 channel 顺序是否是 `[reg, obj, cls]`。
- 预测端 decode 是否使用同一套 bbox 解码函数。
- 训练和预测是否用了同一个网络类。

### 5.2 指标异常高或异常低

优先检查：

- Dataset 是否被切换。
- box 是否被 int 截断。
- letterbox 是否训练预测一致。
- eval json 和预测 json 是否一一对应。
- weight-policy 是否选到了预期权重。

### 5.3 feature 版和 saliency 版命名混乱

判断标准不是原始网络来自分割还是检测，而是当前接入方式：

```text
中间 feature 接检测头       -> feature detection
saliency / mask 路径接检测头 -> saliency detection
```

当前 `sctransnet` 是 feature detection。

当前 `dqaligner` 是 sequence feature detection，使用对齐后的时序 feature 接检测头。

当前 `dqaligner_saliency` 是 sequence saliency detection，使用 DQAligner 原始 mask logits 接检测头。
