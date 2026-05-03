import torch.nn as nn

from StochasticDepth import get_stage_drop_path_rates
from FusedMBConvBlock7 import FusedMBConvBlock7
from FusedMBConvBlock8 import FusedMBConvBlock8
from FusedMBConvBlock9 import FusedMBConvBlock9
from FusedMBConvBlock10 import FusedMBConvBlock10
class Stage3(nn.Module):
    def __init__(self, drop_path_rates=None):
        super().__init__()
        drop_path_rates = get_stage_drop_path_rates(drop_path_rates, 4)
        self.block7 = FusedMBConvBlock7(drop_path_rate=drop_path_rates[0])
        self.block8 = FusedMBConvBlock8(drop_path_rate=drop_path_rates[1])
        self.block9 = FusedMBConvBlock9(drop_path_rate=drop_path_rates[2])
        self.block10 = FusedMBConvBlock10(drop_path_rate=drop_path_rates[3])

    def forward(self, x):
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        return x
