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


class BackboneConv1(nn.Module):
    """
    Backbone Conv 1

    Input : [B, 3, 640, 640]
    Output: [B, 16, 320, 320]

    Conv:
        3 -> 16
        kernel = 3
        stride = 2
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=autopad(3),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(16)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x