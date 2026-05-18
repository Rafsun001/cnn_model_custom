import torch.nn as nn
import torch.nn.functional as F


class ASPPImagePoolingBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        original_spatial_size = x.shape[2:]

        x = self.avgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = F.interpolate(
            x,
            size=original_spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        return x
