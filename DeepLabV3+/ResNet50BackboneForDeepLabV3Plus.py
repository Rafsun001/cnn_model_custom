import torch.nn as nn
from .ResNet50Stage1 import ResNet50Stage1
from .ResNet50Stage2 import ResNet50Stage2
from .ResNet50Stage3 import ResNet50Stage3
from .ResNet50Stage4 import ResNet50Stage4
from .ResNet50Stem import ResNet50Stem


class ResNet50BackboneForDeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = ResNet50Stem()
        self.stage1 = ResNet50Stage1()
        self.stage2 = ResNet50Stage2()
        self.stage3 = ResNet50Stage3()
        self.stage4 = ResNet50Stage4()

    def forward(self, x):
        x = self.stem(x)

        stage1_output = self.stage1(x)
        low_level_feature = stage1_output

        stage2_output = self.stage2(stage1_output)
        stage3_output = self.stage3(stage2_output)
        stage4_output = self.stage4(stage3_output)

        high_level_feature = stage4_output

        return low_level_feature, high_level_feature
