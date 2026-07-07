import torch.nn as nn


class DeformConv(nn.Module):
    """Fallback for DQAligner DCN when the custom CUDA extension is unavailable.

    The offset argument is accepted for API compatibility but ignored.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=128,
        bias=True,
        lr_mult=0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input, offset=None):
        return self.conv(input)
