import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    """
    Exact ResNet bottleneck block style.

    Structure:
        1x1 conv -> 3x3 conv -> 1x1 conv

    Expansion:
        output_channels = bottleneck_channels * 4

    This version places downsampling stride in conv1, matching the
    original ResNet bottleneck design from the paper-style architecture.
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        out_channels = bottleneck_channels * self.expansion

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)

        return out
