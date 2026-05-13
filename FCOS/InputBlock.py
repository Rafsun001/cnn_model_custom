import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class InputBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Input must be 4D: (N, C, H, W)")
        if x.shape[1] != 3:
            raise ValueError("Input image must have 3 channels")
        return x
