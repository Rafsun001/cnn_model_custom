import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNTopDownP4ToP3Block(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, p4_inner, p3_lateral):
        p4_up = F.interpolate(p4_inner, size=p3_lateral.shape[-2:], mode="nearest")
        p3_inner = p3_lateral + p4_up
        return p3_inner
