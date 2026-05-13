import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResidualBlock8 import ResidualBlock8
from ResidualBlock9 import ResidualBlock9
from ResidualBlock10 import ResidualBlock10
from ResidualBlock11 import ResidualBlock11
from ResidualBlock12 import ResidualBlock12
from ResidualBlock13 import ResidualBlock13

class ResNetStage3Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.block8 = ResidualBlock8()
        self.block9 = ResidualBlock9()
        self.block10 = ResidualBlock10()
        self.block11 = ResidualBlock11()
        self.block12 = ResidualBlock12()
        self.block13 = ResidualBlock13()

    def forward(self, x):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        return x
