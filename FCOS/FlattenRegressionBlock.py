import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlattenRegressionBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, regression):
        n, c, h, w = regression.shape
        regression = regression.permute(0, 2, 3, 1).contiguous()
        regression = regression.view(n, h * w, c)
        return regression
