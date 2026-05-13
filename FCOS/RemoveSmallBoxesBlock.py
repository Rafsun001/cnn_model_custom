import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class RemoveSmallBoxesBlock(nn.Module):

    def __init__(self, min_size: float = 0.0):
        super().__init__()
        self.min_size = min_size

    def forward(self, boxes):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths >= self.min_size) & (heights >= self.min_size)
        return keep
