import torch
import torch.nn as nn

class BackboneConv2(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(32)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
