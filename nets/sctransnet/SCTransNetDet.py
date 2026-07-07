import torch
import torch.nn as nn

from .SCTransNet import ChannelTransformer, Res_block, UpBlock_attention


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SCTransYOLOXHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, hidden_channels=128):
        super().__init__()
        self.stem = ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cls_convs = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )
        self.reg_convs = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(hidden_channels, 4, 1)
        self.obj_pred = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return torch.cat([self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)


class SCTransNetDet(nn.Module):
    """SCTransNet backbone with a single-scale YOLOX-style detection head.

    The head is attached to d4, whose spatial stride is 8 relative to the input.
    It returns a list to match YOLOX loss: [B, 5 + num_classes, H/8, W/8].
    """

    def __init__(self, config, n_channels=1, num_classes=1, img_size=512, vis=False):
        super().__init__()
        self.vis = vis
        in_channels = config.base_channel
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.mtc = ChannelTransformer(
            config,
            vis,
            img_size,
            channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
            patchSize=config.patch_sizes,
        )
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.head = SCTransYOLOXHead(in_channels * 4, num_classes=num_classes)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = [block(input_channels, output_channels)]
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))

        f1, f2, f3, f4 = x1, x2, x3, x4
        x1, x2, x3, x4, _ = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4

        d4 = self.up_decoder4(d5, x4)
        return [self.head(d4)]
