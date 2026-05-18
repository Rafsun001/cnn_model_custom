import torch.nn as nn
from .ResNet50Stage4Block1 import ResNet50Stage4Block1
from .ResNet50Stage4Block2 import ResNet50Stage4Block2
from .ResNet50Stage4Block3 import ResNet50Stage4Block3


class ResNet50Stage4(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ResNet50Stage4Block1()
        self.block2 = ResNet50Stage4Block2()
        self.block3 = ResNet50Stage4Block3()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
