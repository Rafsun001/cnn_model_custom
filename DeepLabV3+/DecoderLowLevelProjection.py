import torch.nn as nn


class DecoderLowLevelProjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low_level_feature):
        x = self.conv(low_level_feature)
        x = self.bn(x)
        x = self.relu(x)
        return x
