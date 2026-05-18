import torch.nn as nn
from .ResNet50Stage2Block1 import ResNet50Stage2Block1
from .ResNet50Stage2Block2 import ResNet50Stage2Block2
from .ResNet50Stage2Block3 import ResNet50Stage2Block3
from .ResNet50Stage2Block4 import ResNet50Stage2Block4


class ResNet50Stage2(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ResNet50Stage2Block1()
        self.block2 = ResNet50Stage2Block2()
        self.block3 = ResNet50Stage2Block3()
        self.block4 = ResNet50Stage2Block4()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
