import torch.nn as nn
from ResidualBlock1 import ResidualBlock1
from ResidualBlock2 import ResidualBlock2
from ResidualBlock3 import ResidualBlock3


class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock1()
        self.block2 = ResidualBlock2()
        self.block3 = ResidualBlock3()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
