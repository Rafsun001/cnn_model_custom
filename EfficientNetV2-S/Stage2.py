import torch.nn as nn

from StochasticDepth import get_stage_drop_path_rates
from FusedMBConvBlock3 import FusedMBConvBlock3
from FusedMBConvBlock4 import FusedMBConvBlock4
from FusedMBConvBlock5 import FusedMBConvBlock5
from FusedMBConvBlock6 import FusedMBConvBlock6
class Stage2(nn.Module):
    def __init__(self, drop_path_rates=None):
        super().__init__()
        drop_path_rates = get_stage_drop_path_rates(drop_path_rates, 4)
        self.block3 = FusedMBConvBlock3(drop_path_rate=drop_path_rates[0])
        self.block4 = FusedMBConvBlock4(drop_path_rate=drop_path_rates[1])
        self.block5 = FusedMBConvBlock5(drop_path_rate=drop_path_rates[2])
        self.block6 = FusedMBConvBlock6(drop_path_rate=drop_path_rates[3])

    def forward(self, x):
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x
