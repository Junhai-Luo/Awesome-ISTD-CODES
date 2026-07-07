import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import AsymBiChaFuseReduce, BiLocalChaFuseReduce, BiGlobalChaFuseReduce
from .segmentation import ASKCResUNet, ResidualBlock


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super(ConvBNAct, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k, s, p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class ACMDetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ACMDetHead, self).__init__()
        mid_channels = 128
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
        return torch.cat([reg_out, obj_out, cls_out], 1)


class ACMSaliencyBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, groups=1, bias=False):
        super(ACMSaliencyBaseConv, self).__init__()
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


class ACMSaliencyDetHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super(ACMSaliencyDetHead, self).__init__()
        self.cls_convs = nn.Sequential(
            ACMSaliencyBaseConv(in_channels, in_channels, 3, 1),
            ACMSaliencyBaseConv(in_channels, in_channels, 3, 1),
        )
        self.reg_convs = nn.Sequential(
            ACMSaliencyBaseConv(in_channels, in_channels, 3, 1),
            ACMSaliencyBaseConv(in_channels, in_channels, 3, 1),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1, 1, 0)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1, 1, 0)

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return torch.cat([self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)


class ASKCResNetFPNDet(nn.Module):
    def __init__(self, layer_blocks, channels, fuse_mode='AsymBi', num_classes=1):
        super(ASKCResNetFPNDet, self).__init__()

        stem_width = channels[0]
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),
            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),
            nn.Conv2d(stem_width, stem_width * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width * 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer1 = self._make_layer(ResidualBlock, layer_blocks[0], channels[1], channels[1], stride=1)
        self.layer2 = self._make_layer(ResidualBlock, layer_blocks[1], channels[1], channels[2], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, layer_blocks[2], channels[2], channels[3], stride=2)

        self.fuse23 = self._fuse_layer(channels[3], channels[2], channels[2], fuse_mode)
        self.fuse12 = self._fuse_layer(channels[2], channels[1], channels[1], fuse_mode)

        self.head = ACMDetHead(channels[2], num_classes)

    def forward(self, x):
        _, _, hei, wid = x.shape
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        out = self.layer3(c2)

        out = F.interpolate(out, size=[hei // 8, wid // 8], mode='bilinear')
        out = self.fuse23(out, c2)

        pred = self.head(out)
        return [pred]

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layers = [block(in_channels, out_channels, stride, downsample)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layers)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            return BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        if fuse_mode == 'AsymBi':
            return AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        return BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)


class ASKCResUNetSaliencyDet(nn.Module):
    def __init__(self, layer_blocks, channels, fuse_mode='AsymBi', num_classes=1):
        super(ASKCResUNetSaliencyDet, self).__init__()
        self.backbone = ASKCResUNet(layer_blocks, channels, fuse_mode)
        self.conv = nn.Sequential(
            ACMSaliencyBaseConv(1, 4, 3, 2),
            ACMSaliencyBaseConv(4, 16, 3, 2),
            ACMSaliencyBaseConv(16, 64, 3, 2),
            ACMSaliencyBaseConv(64, 256, 3, 1),
        )
        self.head = ACMSaliencyDetHead(256, num_classes)

    def forward(self, x):
        saliency = self.backbone(x)
        feat = self.conv(saliency)
        return [self.head(feat)]


class ASKCResUNetDet(nn.Module):
    def __init__(self, layer_blocks, channels, fuse_mode='AsymBi', num_classes=1):
        super(ASKCResUNetDet, self).__init__()

        stem_width = int(channels[0])
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),
            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),
            nn.Conv2d(stem_width, 2 * stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * stem_width),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer1 = self._make_layer(ResidualBlock, layer_blocks[0], channels[1], channels[1], stride=1)
        self.layer2 = self._make_layer(ResidualBlock, layer_blocks[1], channels[1], channels[2], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, layer_blocks[2], channels[2], channels[3], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)
        self.fuse2 = self._fuse_layer(channels[2], channels[2], channels[2], fuse_mode)
        self.uplayer2 = self._make_layer(ResidualBlock, layer_blocks[1], channels[2], channels[2], stride=1)

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)
        self.fuse1 = self._fuse_layer(channels[1], channels[1], channels[1], fuse_mode)
        self.uplayer1 = self._make_layer(ResidualBlock, layer_blocks[0], channels[1], channels[1], stride=1)

        self.head = ACMDetHead(channels[2], num_classes)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        deconv2 = self.deconv2(c3)
        fuse2 = self.fuse2(deconv2, c2)
        up2 = self.uplayer2(fuse2)

        pred = self.head(up2)
        return [pred]

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layers = [block(in_channels, out_channels, stride, downsample)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layers)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            return BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        if fuse_mode == 'AsymBi':
            return AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        return BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
