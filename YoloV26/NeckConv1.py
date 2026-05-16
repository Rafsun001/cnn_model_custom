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


class NeckConv1(nn.Module):
    """
    Neck downsample Conv 1

    Input : [B, 64, 80, 80]
    Output: [B, 64, 40, 40]

    Conv:
        64 -> 64
        kernel = 3
        stride = 2
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=autopad(3),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x