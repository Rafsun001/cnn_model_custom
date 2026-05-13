import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLevelLocationGeneratorBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature_height: int, feature_width: int, stride: int, device: torch.device):
        shifts_x = torch.arange(
            0,
            feature_width * stride,
            step=stride,
            dtype=torch.float32,
            device=device,
        ) + stride / 2.0

        shifts_y = torch.arange(
            0,
            feature_height * stride,
            step=stride,
            dtype=torch.float32,
            device=device,
        ) + stride / 2.0

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        locations = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)
        return locations
