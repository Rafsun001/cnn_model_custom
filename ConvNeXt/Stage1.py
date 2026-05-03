import torch.nn as nn

from ConvNeXtBlock1 import ConvNeXtBlock1
from ConvNeXtBlock2 import ConvNeXtBlock2
from ConvNeXtBlock3 import ConvNeXtBlock3


class Stage1(nn.Module):
    def __init__(self, drop_path_probs, layer_scale_init_value=1e-6):
        super().__init__()

        self.block1 = ConvNeXtBlock1(
            drop_path_prob=drop_path_probs[0],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block2 = ConvNeXtBlock2(
            drop_path_prob=drop_path_probs[1],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block3 = ConvNeXtBlock3(
            drop_path_prob=drop_path_probs[2],
            layer_scale_init_value=layer_scale_init_value,
        )


    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        return x
