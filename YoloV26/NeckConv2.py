import torch
import torch.nn as nn


def autopad(kernel_size, padding=None, dilation=1):
    if padding is not None:
        return padding

    if isinstance(kernel_size, int):
        effective_kernel = dilation * (kernel_size - 1) + 1
        return effective_kernel // 2

    effective_kernel = [dilation * (k - 1) + 1 for k in kernel_size]
    return [k // 2 for k in effective_kernel]


class NeckConv2(nn.Module):
    """
    Neck downsample Conv 2

    Input : [B, 128, 40, 40]
    Output: [B, 128, 20, 20]

    Conv:
        128 -> 128
        kernel = 3
        stride = 2
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=autopad(3),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(128)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x