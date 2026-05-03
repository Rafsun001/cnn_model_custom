import torch.nn as nn

from StochasticDepth import get_stage_drop_path_rates
from FusedMBConvBlock1 import FusedMBConvBlock1
from FusedMBConvBlock2 import FusedMBConvBlock2
class Stage1(nn.Module):
    def __init__(self, drop_path_rates=None):
        super().__init__()
        drop_path_rates = get_stage_drop_path_rates(drop_path_rates, 2)
        self.block1 = FusedMBConvBlock1(drop_path_rate=drop_path_rates[0])
        self.block2 = FusedMBConvBlock2(drop_path_rate=drop_path_rates[1])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
