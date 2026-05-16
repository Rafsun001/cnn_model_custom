import torch.nn as nn

from StochasticDepth import StochasticDepth
class MBConvSEBlock17(nn.Module):
    def __init__(self, drop_path_rate=0.0):
        super().__init__()

        self.expand_conv = nn.Conv2d(
            in_channels=128,
            out_channels=768,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.expand_bn = nn.BatchNorm2d(768, eps=1e-3)
        self.silu = nn.SiLU(inplace=True)

        self.depthwise_conv = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=768,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(768, eps=1e-3)

        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Conv2d(
            in_channels=768,
            out_channels=32,
            kernel_size=1,
        )
        self.se_expand = nn.Conv2d(
            in_channels=32,
            out_channels=768,
            kernel_size=1,
        )
        self.se_sigmoid = nn.Sigmoid()

        self.project_conv = nn.Conv2d(
            in_channels=768,
            out_channels=160,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(160, eps=1e-3)

        self.use_skip_connection = False
        self.stochastic_depth = StochasticDepth(drop_path_rate)

    def forward(self, x):
        identity = x

        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = self.silu(out)

        out = self.depthwise_conv(out)
        out = self.depthwise_bn(out)
        out = self.silu(out)

        scale = self.se_pool(out)
        scale = self.se_reduce(scale)
        scale = self.silu(scale)
        scale = self.se_expand(scale)
        scale = self.se_sigmoid(scale)

        out = out * scale

        out = self.project_conv(out)
        out = self.project_bn(out)

        if self.use_skip_connection:
            out = self.stochastic_depth(out)
            out = out + identity

        return out
