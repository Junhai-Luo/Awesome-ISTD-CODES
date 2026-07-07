import torch
import torch.nn as nn

from .uiunet import UIUNET, RSU7, RSU6, RSU5, RSU4, RSU4F, _upsample_like
from .fusion import AsymBiChaFuseReduce


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super(ConvBNAct, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k, s, p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UIUDetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UIUDetHead, self).__init__()
        mid_channels = 128
        self.stem = ConvBNAct(in_channels, mid_channels, k=1, s=1, p=0)
        self.cls_conv = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels),
            ConvBNAct(mid_channels, mid_channels),
        )
        self.reg_conv = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels),
            ConvBNAct(mid_channels, mid_channels),
        )
        self.cls_pred = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(mid_channels, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)
        return torch.cat([reg_out, obj_out, cls_out], dim=1)


class UIUSaliencyBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, groups=1, bias=False):
        super(UIUSaliencyBaseConv, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=s,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UIUSaliencyDetHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super(UIUSaliencyDetHead, self).__init__()
        self.cls_convs = nn.Sequential(
            UIUSaliencyBaseConv(in_channels, in_channels, 3, 1),
            UIUSaliencyBaseConv(in_channels, in_channels, 3, 1),
        )
        self.reg_convs = nn.Sequential(
            UIUSaliencyBaseConv(in_channels, in_channels, 3, 1),
            UIUSaliencyBaseConv(in_channels, in_channels, 3, 1),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1, 1, 0)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1, 0)

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return torch.cat([self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)


class UIUNETSaliencyDet(nn.Module):
    """UIU-Net detector that follows the original saliency-output-to-box-head route."""

    def __init__(self, in_ch=3, num_classes=1, fuse_mode="AsymBi"):
        super(UIUNETSaliencyDet, self).__init__()
        if fuse_mode != "AsymBi":
            raise ValueError("UIUNETSaliencyDet currently supports only fuse_mode='AsymBi'.")
        self.backbone = UIUNET(in_ch=in_ch, out_ch=1)
        self.conv = nn.Sequential(
            UIUSaliencyBaseConv(7, 16, 3, 2),
            UIUSaliencyBaseConv(16, 64, 3, 2),
            UIUSaliencyBaseConv(64, 256, 3, 2),
        )
        self.head = UIUSaliencyDetHead(256, num_classes)

    def forward(self, x):
        saliency_outputs = self.backbone(x)
        if not isinstance(saliency_outputs, (list, tuple)):
            saliency_outputs = [saliency_outputs]
        if len(saliency_outputs) < 7:
            saliency_outputs = list(saliency_outputs) + [saliency_outputs[-1]] * (7 - len(saliency_outputs))
        feat = torch.cat(list(saliency_outputs[:7]), dim=1)
        feat = self.conv(feat)
        return [self.head(feat)]


class UIUNETDet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, fuse_mode="AsymBi"):
        super(UIUNETDet, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.fuse5 = self._fuse_layer(512, 512, 512, fuse_mode=fuse_mode)
        self.fuse4 = self._fuse_layer(512, 512, 512, fuse_mode=fuse_mode)
        self.fuse3 = self._fuse_layer(256, 256, 256, fuse_mode=fuse_mode)
        self.fuse2 = self._fuse_layer(128, 128, 128, fuse_mode=fuse_mode)

        self.det_head_s8 = UIUDetHead(256, num_classes)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode="AsymBi"):
        if fuse_mode == "AsymBi":
            return AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        raise ValueError("Unsupported fuse_mode '%s'. Use AsymBi." % fuse_mode)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        fusec51, fusec52 = self.fuse5(hx6up, hx5)
        hx5d = self.stage5d(torch.cat((fusec51, fusec52), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        fusec41, fusec42 = self.fuse4(hx5dup, hx4)
        hx4d = self.stage4d(torch.cat((fusec41, fusec42), 1))
        return [self.det_head_s8(hx4d)]
