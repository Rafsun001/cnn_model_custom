import torch.nn as nn
from ResidualBlock4 import ResidualBlock4
from ResidualBlock5 import ResidualBlock5
from ResidualBlock6 import ResidualBlock6
from ResidualBlock7 import ResidualBlock7


class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block4 = ResidualBlock4()
        self.block5 = ResidualBlock5()
        self.block6 = ResidualBlock6()
        self.block7 = ResidualBlock7()

    def forward(self, x):
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x
