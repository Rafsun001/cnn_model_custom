import torch.nn as nn

from StochasticDepth import StochasticDepth
class FusedMBConvBlock10(nn.Module):
    def __init__(self, drop_path_rate=0.0):
        super().__init__()

        self.expand_conv = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.expand_bn = nn.BatchNorm2d(256, eps=1e-3)
        self.silu = nn.SiLU(inplace=True)

        self.project_conv = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(64, eps=1e-3)

        self.use_skip_connection = True
        self.stochastic_depth = StochasticDepth(drop_path_rate)

    def forward(self, x):
        identity = x

        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.silu(out)

        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_skip_connection:
            out = self.stochastic_depth(out)
            out = out + identity

        return out
