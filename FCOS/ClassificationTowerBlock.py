import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationTowerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(32, 256)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(32, 256)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn3 = nn.GroupNorm(32, 256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn4 = nn.GroupNorm(32, 256)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.gn4(x)
        x = self.relu4(x)

        return x
