import torch.nn as nn

from StochasticDepth import make_drop_path_rates
from InputBlock import InputBlock
from StemBlock import StemBlock
from Stage1 import Stage1
from Stage2 import Stage2
from Stage3 import Stage3
from Stage4 import Stage4
from Stage5 import Stage5
from Stage6 import Stage6
from HeadBlock import HeadBlock
from GlobalAveragePoolingBlock import GlobalAveragePoolingBlock
from ClassifierBlock import ClassifierBlock
class EfficientNetV2SFinalBlock(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.2, drop_path_rate=0.2):
        super().__init__()

        self.input_block = InputBlock(image_channels=3)
        self.stem_block = StemBlock()

        drop_path_rates = make_drop_path_rates(
            total_blocks=40,
            final_drop_path_rate=drop_path_rate,
        )

        self.stage1 = Stage1(drop_path_rates=drop_path_rates[0:2])
        self.stage2 = Stage2(drop_path_rates=drop_path_rates[2:6])
        self.stage3 = Stage3(drop_path_rates=drop_path_rates[6:10])
        self.stage4 = Stage4(drop_path_rates=drop_path_rates[10:16])
        self.stage5 = Stage5(drop_path_rates=drop_path_rates[16:25])
        self.stage6 = Stage6(drop_path_rates=drop_path_rates[25:40])

        self.head_block = HeadBlock()
        self.pooling_block = GlobalAveragePoolingBlock()
        self.classifier_block = ClassifierBlock(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.stem_block(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)

        x = self.head_block(x)
        x = self.pooling_block(x)
        x = self.classifier_block(x)

        return x
