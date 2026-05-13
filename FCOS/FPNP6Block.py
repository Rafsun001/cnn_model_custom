import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNP6Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, p5):
        return self.conv(p5)
