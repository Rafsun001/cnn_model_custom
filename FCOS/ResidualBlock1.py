import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.skip_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.skip_bn = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        identity = self.skip_conv(identity)
        identity = self.skip_bn(identity)

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
