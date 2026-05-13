import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceToBoxBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, locations, distances):
        x = locations[:, 0]
        y = locations[:, 1]

        left = distances[:, 0]
        top = distances[:, 1]
        right = distances[:, 2]
        bottom = distances[:, 3]

        x1 = x - left
        y1 = y - top
        x2 = x + right
        y2 = y + bottom

        boxes = torch.stack((x1, y1, x2, y2), dim=1)
        return boxes
