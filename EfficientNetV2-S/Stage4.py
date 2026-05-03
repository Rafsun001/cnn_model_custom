import torch.nn as nn

from StochasticDepth import get_stage_drop_path_rates
from MBConvSEBlock11 import MBConvSEBlock11
from MBConvSEBlock12 import MBConvSEBlock12
from MBConvSEBlock13 import MBConvSEBlock13
from MBConvSEBlock14 import MBConvSEBlock14
from MBConvSEBlock15 import MBConvSEBlock15
from MBConvSEBlock16 import MBConvSEBlock16
class Stage4(nn.Module):
    def __init__(self, drop_path_rates=None):
        super().__init__()
        drop_path_rates = get_stage_drop_path_rates(drop_path_rates, 6)
        self.block11 = MBConvSEBlock11(drop_path_rate=drop_path_rates[0])
        self.block12 = MBConvSEBlock12(drop_path_rate=drop_path_rates[1])
        self.block13 = MBConvSEBlock13(drop_path_rate=drop_path_rates[2])
        self.block14 = MBConvSEBlock14(drop_path_rate=drop_path_rates[3])
        self.block15 = MBConvSEBlock15(drop_path_rate=drop_path_rates[4])
        self.block16 = MBConvSEBlock16(drop_path_rate=drop_path_rates[5])

    def forward(self, x):
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        return x
