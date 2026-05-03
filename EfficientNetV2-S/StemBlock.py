import torch.nn as nn


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(24, eps=1e-3)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x
