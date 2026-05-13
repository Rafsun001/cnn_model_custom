import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from InputBlock import InputBlock
from ResNet50StemBlock import ResNet50StemBlock
from ResNetStage1Block import ResNetStage1Block
from ResNetStage2Block import ResNetStage2Block
from ResNetStage3Block import ResNetStage3Block
from ResNetStage4Block import ResNetStage4Block

class ResNet50BackboneForBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_block = InputBlock()
        self.stem = ResNet50StemBlock()
        self.stage1 = ResNetStage1Block()
        self.stage2 = ResNetStage2Block()
        self.stage3 = ResNetStage3Block()
        self.stage4 = ResNetStage4Block()

    def forward(self, x):
        x = self.input_block(x)
        x = self.stem(x)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return c3, c4, c5
