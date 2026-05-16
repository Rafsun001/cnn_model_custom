from __future__ import annotations

import torch
import torch.nn as nn

from BottleneckBlock import BottleneckBlock


class ResNet50(nn.Module):
    """
    Exact ResNet-50 backbone structure.

    Architecture:
        Stem:
            7x7 conv, stride 2, padding 3
            BatchNorm
            ReLU
            3x3 maxpool, stride 2, padding 1

        Residual stages:
            conv2_x: 3 bottleneck blocks,  output 256 channels
            conv3_x: 4 bottleneck blocks,  output 512 channels
            conv4_x: 6 bottleneck blocks,  output 1024 channels
            conv5_x: 3 bottleneck blocks,  output 2048 channels

        Head:
            AdaptiveAvgPool2d
            Linear classifier

    For TinyImageNet, use num_classes=200.
    For ImageNet-1K, use num_classes=1000.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.stage1 = self._make_stage(
            bottleneck_channels=64,
            num_blocks=3,
            stride=1,
        )
        self.stage2 = self._make_stage(
            bottleneck_channels=128,
            num_blocks=4,
            stride=2,
        )
        self.stage3 = self._make_stage(
            bottleneck_channels=256,
            num_blocks=6,
            stride=2,
        )
        self.stage4 = self._make_stage(
            bottleneck_channels=512,
            num_blocks=3,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

        self.initialize_weights()

    def _make_stage(
        self,
        bottleneck_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = []

        layers.append(
            BottleneckBlock(
                in_channels=self.in_channels,
                bottleneck_channels=bottleneck_channels,
                stride=stride,
            )
        )

        self.in_channels = bottleneck_channels * BottleneckBlock.expansion

        for _ in range(1, num_blocks):
            layers.append(
                BottleneckBlock(
                    in_channels=self.in_channels,
                    bottleneck_channels=bottleneck_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet50TinyImageNet200(ResNet50):
    """
    Backward-compatible class name for your existing training code.
    """
    def __init__(self, num_classes: int = 200):
        super().__init__(num_classes=num_classes)


def resnet50(num_classes: int = 200) -> ResNet50:
    return ResNet50(num_classes=num_classes)
