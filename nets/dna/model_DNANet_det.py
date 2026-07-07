import torch
import torch.nn as nn
import torch.nn.functional as F

from .load_param_data import load_param
from .model_DNANet import DNANet, Res_CBAM_block


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


class DNANetDetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DNANetDetHead, self).__init__()
        hidden = max(64, in_channels * 4)
        self.stem = ConvBNAct(in_channels, hidden, k=1, s=1, p=0)
        self.cls_conv = nn.Sequential(
            ConvBNAct(hidden, hidden, 3, 1, 1),
            ConvBNAct(hidden, hidden, 3, 1, 1),
        )
        self.reg_conv = nn.Sequential(
            ConvBNAct(hidden, hidden, 3, 1, 1),
            ConvBNAct(hidden, hidden, 3, 1, 1),
        )
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(hidden, 4, 1, 1, 0)
        self.obj_pred = nn.Conv2d(hidden, 1, 1, 1, 0)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        return torch.cat([self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=None):
        super(BaseConv, self).__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DNANetSaliencyDetHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super(DNANetSaliencyDetHead, self).__init__()
        self.cls_convs = nn.Sequential(
            BaseConv(in_channels, in_channels, 3, 1),
            BaseConv(in_channels, in_channels, 3, 1),
        )
        self.reg_convs = nn.Sequential(
            BaseConv(in_channels, in_channels, 3, 1),
            BaseConv(in_channels, in_channels, 3, 1),
        )
        self.cls_preds = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        self.reg_preds = nn.Conv2d(in_channels, 4, 1, 1, 0)
        self.obj_preds = nn.Conv2d(in_channels, 1, 1, 1, 0)

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return torch.cat([self.reg_preds(reg_feat), self.obj_preds(reg_feat), self.cls_preds(cls_feat)], dim=1)


class DNANetSaliencyDet(nn.Module):
    """DNANet detector that follows the original saliency-output-to-box-head route."""

    def __init__(self, input_channels=3, num_classes=1, channel_size="three", backbone="resnet_18"):
        super(DNANetSaliencyDet, self).__init__()
        nb_filter, num_blocks = load_param(channel_size, backbone)
        self.backbone = DNANet(
            num_classes=1,
            input_channels=input_channels,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=True,
        )
        self.conv = nn.Sequential(
            BaseConv(4, 16, 3, 2),
            BaseConv(16, 64, 3, 2),
            BaseConv(64, 256, 3, 2),
        )
        self.head = DNANetSaliencyDetHead(256, num_classes)

    def forward(self, x):
        saliency_outputs = self.backbone(x)
        if not isinstance(saliency_outputs, (list, tuple)):
            saliency_outputs = [saliency_outputs]
        if len(saliency_outputs) < 4:
            saliency_outputs = list(saliency_outputs) + [saliency_outputs[-1]] * (4 - len(saliency_outputs))

        feat = torch.cat(saliency_outputs[:4], dim=1)
        feat = self.conv(feat)
        return [self.head(feat)]


class DNANetDet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, channel_size="three", backbone="resnet_18"):
        super(DNANetDet, self).__init__()
        nb_filter, num_blocks = load_param(channel_size, backbone)
        block = Res_CBAM_block

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1] * 2 + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2] * 2 + nb_filter[3] + nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] * 3 + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], 1, 1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], 1, 1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], 1, 1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], 1, 1)

        self.det_x4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[2], 1, 1)
        self.det_x3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[2], 1, 1)
        self.det_x2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[2], 1, 1)
        self.det_x1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[2], 1, 1)
        self.det_fuse = self._make_layer(block, nb_filter[2] * 4, nb_filter[2], num_blocks[1])

        self.head = DNANetDetHead(nb_filter[2], num_classes)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = [block(input_channels, output_channels)]
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        return self.det_fuse(
            torch.cat(
                [
                    self.up_4(self.det_x4_1x1(x4_0)),
                    self.up(self.det_x3_1x1(x3_1)),
                    self.det_x2_1x1(x2_2),
                    self.down(self.det_x1_1x1(x1_3)),
                ],
                1,
            )
        )

    def forward(self, x):
        feat = self.forward_features(x)
        feat = F.interpolate(
            feat,
            size=(max(1, x.shape[-2] // 8), max(1, x.shape[-1] // 8)),
            mode="bilinear",
            align_corners=False,
        )
        return [self.head(feat)]

