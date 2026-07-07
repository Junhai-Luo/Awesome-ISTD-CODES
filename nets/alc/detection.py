import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from .fusion import AsymBiChaFuse


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


class ALCDetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ALCDetHead, self).__init__()
        mid_channels = 64
        self.stem = ConvBNAct(in_channels, mid_channels, k=1, s=1, p=0)
        self.cls_conv = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, k=3, s=1, p=1),
            ConvBNAct(mid_channels, mid_channels, k=3, s=1, p=1),
        )
        self.reg_conv = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, k=3, s=1, p=1),
            ConvBNAct(mid_channels, mid_channels, k=3, s=1, p=1),
        )
        self.cls_pred = nn.Conv2d(mid_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.reg_pred = nn.Conv2d(mid_channels, 4, kernel_size=1, stride=1, padding=0)
        self.obj_pred = nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)
        return torch.cat([reg_out, obj_out, cls_out], dim=1)


class ALCSaliencyBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, groups=1, bias=False):
        super(ALCSaliencyBaseConv, self).__init__()
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


class ALCSaliencyDetHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super(ALCSaliencyDetHead, self).__init__()
        self.cls_convs = nn.Sequential(
            ALCSaliencyBaseConv(in_channels, in_channels, 3, 1),
            ALCSaliencyBaseConv(in_channels, in_channels, 3, 1),
        )
        self.reg_convs = nn.Sequential(
            ALCSaliencyBaseConv(in_channels, in_channels, 3, 1),
            ALCSaliencyBaseConv(in_channels, in_channels, 3, 1),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1, 1, 0)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1, 0)

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return torch.cat([self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)


class ALCNetDet(nn.Module):
    def __init__(self, in_channels=3, layers=None, channels=None, fuse_mode="AsymBi", num_classes=1):
        super(ALCNetDet, self).__init__()
        if layers is None:
            layers = [4, 4, 4]
        if channels is None:
            channels = [8, 16, 32, 64]
        if len(layers) != 3:
            raise ValueError("ALCNetDet currently expects exactly 3 residual stages.")
        if fuse_mode != "AsymBi":
            raise ValueError("ALCNetDet currently supports only fuse_mode='AsymBi'.")

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = channels[1]
        stem_width = int(channels[0])

        self.stem = nn.Sequential(
            self._norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=stem_width, kernel_size=3, stride=2, padding=1, bias=False),
            self._norm_layer(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_width, out_channels=stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            self._norm_layer(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_width, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            self._norm_layer(stem_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(BasicBlock, channels[1], layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, channels[3], layers[2], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.deconv0 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=4, stride=2, padding=1)

        self.fuse23 = AsymBiChaFuse(channels=channels[2])
        self.fuse12 = AsymBiChaFuse(channels=channels[1])
        self.head = ALCDetHead(channels[2], num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        out = self.layer3(c2)

        out = F.interpolate(out, size=(max(1, hei // 16), max(1, wid // 16)), mode="bilinear", align_corners=False)
        out = self.deconv2(out)
        out = F.interpolate(out, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        out = self.fuse23(out, c2)
        out = F.interpolate(out, size=(max(1, hei // 8), max(1, wid // 8)), mode="bilinear", align_corners=False)

        pred = self.head(out)
        return [pred]


class ALCNetSaliencyDet(nn.Module):
    """ALCNet detector that routes a predicted saliency map into a box head."""

    def __init__(self, in_channels=3, layers=None, channels=None, fuse_mode="AsymBi", num_classes=1):
        super(ALCNetSaliencyDet, self).__init__()
        if layers is None:
            layers = [4, 4, 4]
        if channels is None:
            channels = [8, 16, 32, 64]
        if len(layers) != 3:
            raise ValueError("ALCNetSaliencyDet currently expects exactly 3 residual stages.")
        if fuse_mode != "AsymBi":
            raise ValueError("ALCNetSaliencyDet currently supports only fuse_mode='AsymBi'.")

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = channels[1]
        stem_width = int(channels[0])

        self.stem = nn.Sequential(
            self._norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=stem_width, kernel_size=3, stride=2, padding=1, bias=False),
            self._norm_layer(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_width, out_channels=stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            self._norm_layer(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_width, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            self._norm_layer(stem_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(BasicBlock, channels[1], layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, channels[3], layers[2], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.fuse23 = AsymBiChaFuse(channels=channels[2])
        self.saliency_pred = nn.Conv2d(channels[2], 1, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            ALCSaliencyBaseConv(1, 16, 3, 2),
            ALCSaliencyBaseConv(16, 64, 3, 2),
            ALCSaliencyBaseConv(64, 256, 3, 2),
        )
        self.head = ALCSaliencyDetHead(256, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        out = self.layer3(c2)

        out = F.interpolate(out, size=(max(1, hei // 16), max(1, wid // 16)), mode="bilinear", align_corners=False)
        out = self.deconv2(out)
        out = F.interpolate(out, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        out = self.fuse23(out, c2)
        out = F.interpolate(out, size=(max(1, hei // 8), max(1, wid // 8)), mode="bilinear", align_corners=False)

        saliency = torch.sigmoid(self.saliency_pred(out))
        saliency = F.interpolate(saliency, size=(hei, wid), mode="bilinear", align_corners=False)
        feat = self.conv(saliency)
        return [self.head(feat)]
