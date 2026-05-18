import torch.nn as nn

from InputBlock import InputBlock
from StemBlock import StemBlock
from DownsampleBlock1 import DownsampleBlock1
from DownsampleBlock2 import DownsampleBlock2
from DownsampleBlock3 import DownsampleBlock3
from Stage1 import Stage1
from Stage2 import Stage2
from Stage3 import Stage3
from Stage4 import Stage4
from GlobalAveragePoolingBlock import GlobalAveragePoolingBlock
from ClassifierHeadBlock import ClassifierHeadBlock


class ConvNeXtBaseFinalBlock(nn.Module):
    def __init__(self, num_classes=1000, drop_path_rate=0.5, layer_scale_init_value=1e-6):
        super().__init__()


        drop_path_probs = []
        for block_index in range(36):
            if 36 == 1:
                prob = drop_path_rate
            else:
                prob = drop_path_rate * block_index / (36 - 1)
            drop_path_probs.append(prob)

        self.input_block = InputBlock(image_channels=3)
        self.stem = StemBlock()

        self.stage1 = Stage1(
            drop_path_probs=drop_path_probs[0:3],
            layer_scale_init_value=layer_scale_init_value,
        )
        self.downsample1 = DownsampleBlock1()

        self.stage2 = Stage2(
            drop_path_probs=drop_path_probs[3:6],
            layer_scale_init_value=layer_scale_init_value,
        )
        self.downsample2 = DownsampleBlock2()

        self.stage3 = Stage3(
            drop_path_probs=drop_path_probs[6:33],
            layer_scale_init_value=layer_scale_init_value,
        )
        self.downsample3 = DownsampleBlock3()

        self.stage4 = Stage4(
            drop_path_probs=drop_path_probs[33:36],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.global_pool = GlobalAveragePoolingBlock()
        self.classifier = ClassifierHeadBlock(num_classes=num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_block(x)

        x = self.stem(x)

        x = self.stage1(x)
        x = self.downsample1(x)

        x = self.stage2(x)
        x = self.downsample2(x)

        x = self.stage3(x)
        x = self.downsample3(x)

        x = self.stage4(x)

        x = self.global_pool(x)
        x = self.classifier(x)

        return x
