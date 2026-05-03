import torch.nn as nn

from StochasticDepth import StochasticDepth
class FusedMBConvBlock2(nn.Module):
    def __init__(self, drop_path_rate=0.0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=24,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(24, eps=1e-3)
        self.silu = nn.SiLU(inplace=True)

        self.use_skip_connection = True
        self.stochastic_depth = StochasticDepth(drop_path_rate)

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.silu(out)

        if self.use_skip_connection:
            out = self.stochastic_depth(out)
            out = out + identity

        return out
