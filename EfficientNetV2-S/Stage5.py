import torch.nn as nn

from StochasticDepth import get_stage_drop_path_rates
from MBConvSEBlock17 import MBConvSEBlock17
from MBConvSEBlock18 import MBConvSEBlock18
from MBConvSEBlock19 import MBConvSEBlock19
from MBConvSEBlock20 import MBConvSEBlock20
from MBConvSEBlock21 import MBConvSEBlock21
from MBConvSEBlock22 import MBConvSEBlock22
from MBConvSEBlock23 import MBConvSEBlock23
from MBConvSEBlock24 import MBConvSEBlock24
from MBConvSEBlock25 import MBConvSEBlock25
class Stage5(nn.Module):
    def __init__(self, drop_path_rates=None):
        super().__init__()
        drop_path_rates = get_stage_drop_path_rates(drop_path_rates, 9)
        self.block17 = MBConvSEBlock17(drop_path_rate=drop_path_rates[0])
        self.block18 = MBConvSEBlock18(drop_path_rate=drop_path_rates[1])
        self.block19 = MBConvSEBlock19(drop_path_rate=drop_path_rates[2])
        self.block20 = MBConvSEBlock20(drop_path_rate=drop_path_rates[3])
        self.block21 = MBConvSEBlock21(drop_path_rate=drop_path_rates[4])
        self.block22 = MBConvSEBlock22(drop_path_rate=drop_path_rates[5])
        self.block23 = MBConvSEBlock23(drop_path_rate=drop_path_rates[6])
        self.block24 = MBConvSEBlock24(drop_path_rate=drop_path_rates[7])
        self.block25 = MBConvSEBlock25(drop_path_rate=drop_path_rates[8])

    def forward(self, x):
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block23(x)
        x = self.block24(x)
        x = self.block25(x)
        return x
