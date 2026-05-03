import torch.nn as nn

from StochasticDepth import get_stage_drop_path_rates
from MBConvSEBlock26 import MBConvSEBlock26
from MBConvSEBlock27 import MBConvSEBlock27
from MBConvSEBlock28 import MBConvSEBlock28
from MBConvSEBlock29 import MBConvSEBlock29
from MBConvSEBlock30 import MBConvSEBlock30
from MBConvSEBlock31 import MBConvSEBlock31
from MBConvSEBlock32 import MBConvSEBlock32
from MBConvSEBlock33 import MBConvSEBlock33
from MBConvSEBlock34 import MBConvSEBlock34
from MBConvSEBlock35 import MBConvSEBlock35
from MBConvSEBlock36 import MBConvSEBlock36
from MBConvSEBlock37 import MBConvSEBlock37
from MBConvSEBlock38 import MBConvSEBlock38
from MBConvSEBlock39 import MBConvSEBlock39
from MBConvSEBlock40 import MBConvSEBlock40
class Stage6(nn.Module):
    def __init__(self, drop_path_rates=None):
        super().__init__()
        drop_path_rates = get_stage_drop_path_rates(drop_path_rates, 15)
        self.block26 = MBConvSEBlock26(drop_path_rate=drop_path_rates[0])
        self.block27 = MBConvSEBlock27(drop_path_rate=drop_path_rates[1])
        self.block28 = MBConvSEBlock28(drop_path_rate=drop_path_rates[2])
        self.block29 = MBConvSEBlock29(drop_path_rate=drop_path_rates[3])
        self.block30 = MBConvSEBlock30(drop_path_rate=drop_path_rates[4])
        self.block31 = MBConvSEBlock31(drop_path_rate=drop_path_rates[5])
        self.block32 = MBConvSEBlock32(drop_path_rate=drop_path_rates[6])
        self.block33 = MBConvSEBlock33(drop_path_rate=drop_path_rates[7])
        self.block34 = MBConvSEBlock34(drop_path_rate=drop_path_rates[8])
        self.block35 = MBConvSEBlock35(drop_path_rate=drop_path_rates[9])
        self.block36 = MBConvSEBlock36(drop_path_rate=drop_path_rates[10])
        self.block37 = MBConvSEBlock37(drop_path_rate=drop_path_rates[11])
        self.block38 = MBConvSEBlock38(drop_path_rate=drop_path_rates[12])
        self.block39 = MBConvSEBlock39(drop_path_rate=drop_path_rates[13])
        self.block40 = MBConvSEBlock40(drop_path_rate=drop_path_rates[14])

    def forward(self, x):
        x = self.block26(x)
        x = self.block27(x)
        x = self.block28(x)
        x = self.block29(x)
        x = self.block30(x)
        x = self.block31(x)
        x = self.block32(x)
        x = self.block33(x)
        x = self.block34(x)
        x = self.block35(x)
        x = self.block36(x)
        x = self.block37(x)
        x = self.block38(x)
        x = self.block39(x)
        x = self.block40(x)
        return x
