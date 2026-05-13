import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from FPNLateralC3Block import FPNLateralC3Block
from FPNLateralC4Block import FPNLateralC4Block
from FPNLateralC5Block import FPNLateralC5Block
from FPNTopDownP5ToP4Block import FPNTopDownP5ToP4Block
from FPNTopDownP4ToP3Block import FPNTopDownP4ToP3Block
from FPNSmoothP3Block import FPNSmoothP3Block
from FPNSmoothP4Block import FPNSmoothP4Block
from FPNSmoothP5Block import FPNSmoothP5Block
from FPNP6Block import FPNP6Block
from FPNP7Block import FPNP7Block

class FPNBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.lateral_c3 = FPNLateralC3Block()
        self.lateral_c4 = FPNLateralC4Block()
        self.lateral_c5 = FPNLateralC5Block()

        self.topdown_p5_to_p4 = FPNTopDownP5ToP4Block()
        self.topdown_p4_to_p3 = FPNTopDownP4ToP3Block()

        self.smooth_p3 = FPNSmoothP3Block()
        self.smooth_p4 = FPNSmoothP4Block()
        self.smooth_p5 = FPNSmoothP5Block()

        self.p6_block = FPNP6Block()
        self.p7_block = FPNP7Block()

    def forward(self, c3, c4, c5):
        p3_lateral = self.lateral_c3(c3)
        p4_lateral = self.lateral_c4(c4)
        p5_lateral = self.lateral_c5(c5)

        p4_inner = self.topdown_p5_to_p4(p5_lateral, p4_lateral)
        p3_inner = self.topdown_p4_to_p3(p4_inner, p3_lateral)

        p3 = self.smooth_p3(p3_inner)
        p4 = self.smooth_p4(p4_inner)
        p5 = self.smooth_p5(p5_lateral)

        p6 = self.p6_block(p5)
        p7 = self.p7_block(p6)

        return p3, p4, p5, p6, p7
