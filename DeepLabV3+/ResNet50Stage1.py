import torch.nn as nn
from .ResNet50Stage1Block1 import ResNet50Stage1Block1
from .ResNet50Stage1Block2 import ResNet50Stage1Block2
from .ResNet50Stage1Block3 import ResNet50Stage1Block3


class ResNet50Stage1(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ResNet50Stage1Block1()
        self.block2 = ResNet50Stage1Block2()
        self.block3 = ResNet50Stage1Block3()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
