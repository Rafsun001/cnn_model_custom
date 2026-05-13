import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet50BackboneForBlock import ResNet50BackboneForBlock
from FPNBlock import FPNBlock
from HeadBlock import HeadBlock

class ResNet50FinalBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = ResNet50BackboneForBlock()
        self.fpn = FPNBlock()
        self.head = HeadBlock()

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5, p6, p7 = self.fpn(c3, c4, c5)
        predictions = self.head(p3, p4, p5, p6, p7)
        return predictions
