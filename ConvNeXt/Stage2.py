import torch.nn as nn

from ConvNeXtBlock4 import ConvNeXtBlock4
from ConvNeXtBlock5 import ConvNeXtBlock5
from ConvNeXtBlock6 import ConvNeXtBlock6


class Stage2(nn.Module):
    def __init__(self, drop_path_probs, layer_scale_init_value=1e-6):
        super().__init__()

        self.block4 = ConvNeXtBlock4(
            drop_path_prob=drop_path_probs[0],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block5 = ConvNeXtBlock5(
            drop_path_prob=drop_path_probs[1],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block6 = ConvNeXtBlock6(
            drop_path_prob=drop_path_probs[2],
            layer_scale_init_value=layer_scale_init_value,
        )


    def forward(self, x):
        x = self.block4(x)

        x = self.block5(x)

        x = self.block6(x)

        return x
