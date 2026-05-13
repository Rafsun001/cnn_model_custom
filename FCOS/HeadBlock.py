import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from SharedHeadBlock import SharedHeadBlock
from ScaleP3Block import ScaleP3Block
from ScaleP4Block import ScaleP4Block
from ScaleP5Block import ScaleP5Block
from ScaleP6Block import ScaleP6Block
from ScaleP7Block import ScaleP7Block

class HeadBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.shared_head = SharedHeadBlock()

        self.scale_p3 = ScaleP3Block()
        self.scale_p4 = ScaleP4Block()
        self.scale_p5 = ScaleP5Block()
        self.scale_p6 = ScaleP6Block()
        self.scale_p7 = ScaleP7Block()

    def forward(self, p3, p4, p5, p6, p7):
        p3_cls, p3_ctr, p3_reg = self.shared_head(p3)
        p4_cls, p4_ctr, p4_reg = self.shared_head(p4)
        p5_cls, p5_ctr, p5_reg = self.shared_head(p5)
        p6_cls, p6_ctr, p6_reg = self.shared_head(p6)
        p7_cls, p7_ctr, p7_reg = self.shared_head(p7)

        p3_reg = self.scale_p3(p3_reg)
        p4_reg = self.scale_p4(p4_reg)
        p5_reg = self.scale_p5(p5_reg)
        p6_reg = self.scale_p6(p6_reg)
        p7_reg = self.scale_p7(p7_reg)

        predictions = {
            "P3": {
                "classification": p3_cls,
                "centerness": p3_ctr,
                "regression": p3_reg,
            },
            "P4": {
                "classification": p4_cls,
                "centerness": p4_ctr,
                "regression": p4_reg,
            },
            "P5": {
                "classification": p5_cls,
                "centerness": p5_ctr,
                "regression": p5_reg,
            },
            "P6": {
                "classification": p6_cls,
                "centerness": p6_ctr,
                "regression": p6_reg,
            },
            "P7": {
                "classification": p7_cls,
                "centerness": p7_ctr,
                "regression": p7_reg,
            },
        }
        return predictions
