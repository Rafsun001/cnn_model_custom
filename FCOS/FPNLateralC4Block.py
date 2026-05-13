import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNLateralC4Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, c4):
        return self.conv(c4)
