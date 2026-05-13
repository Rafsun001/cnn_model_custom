import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigBlock import ConfigBlock

class RegressionRangeBuilderBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, locations_per_level: List[torch.Tensor]):
        ranges_per_level = []

        for level_name, locations in zip(ConfigBlock.LEVEL_NAMES, locations_per_level):
            min_value, max_value = ConfigBlock.REGRESSION_RANGES[level_name]
            ranges = locations.new_zeros((locations.shape[0], 2))
            ranges[:, 0] = min_value
            ranges[:, 1] = max_value
            ranges_per_level.append(ranges)

        return torch.cat(ranges_per_level, dim=0)
