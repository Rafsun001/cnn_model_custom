import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResidualBlock14 import ResidualBlock14
from ResidualBlock15 import ResidualBlock15
from ResidualBlock16 import ResidualBlock16

class ResNetStage4Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.block14 = ResidualBlock14()
        self.block15 = ResidualBlock15()
        self.block16 = ResidualBlock16()

    def forward(self, x):
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        return x
