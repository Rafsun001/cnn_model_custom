import torch.nn as nn


class ASPPBranch1x1Conv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
