import torch
import torch.nn as nn


class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        bottleneck_channels = max(1, int(channels // r))

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, xh, xl):
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * torch.mul(xl, topdown_wei) + 2 * torch.mul(xh, bottomup_wei)
        return self.post(xs)
