import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNTopDownP5ToP4Block(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, p5_lateral, p4_lateral):
        p5_up = F.interpolate(p5_lateral, size=p4_lateral.shape[-2:], mode="nearest")
        p4_inner = p4_lateral + p5_up
        return p4_inner
