import torch.nn as nn
from .ResNet50Stage3Block1 import ResNet50Stage3Block1
from .ResNet50Stage3Block2 import ResNet50Stage3Block2
from .ResNet50Stage3Block3 import ResNet50Stage3Block3
from .ResNet50Stage3Block4 import ResNet50Stage3Block4
from .ResNet50Stage3Block5 import ResNet50Stage3Block5
from .ResNet50Stage3Block6 import ResNet50Stage3Block6


class ResNet50Stage3(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ResNet50Stage3Block1()
        self.block2 = ResNet50Stage3Block2()
        self.block3 = ResNet50Stage3Block3()
        self.block4 = ResNet50Stage3Block4()
        self.block5 = ResNet50Stage3Block5()
        self.block6 = ResNet50Stage3Block6()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x
