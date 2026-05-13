import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlattenCenternessBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, centerness):
        n, c, h, w = centerness.shape
        centerness = centerness.permute(0, 2, 3, 1).contiguous()
        centerness = centerness.view(n, h * w)
        return centerness
