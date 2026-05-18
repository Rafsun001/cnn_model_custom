import torch.nn as nn

from ConvNeXtBlock34 import ConvNeXtBlock34
from ConvNeXtBlock35 import ConvNeXtBlock35
from ConvNeXtBlock36 import ConvNeXtBlock36


class Stage4(nn.Module):
    def __init__(self, drop_path_probs, layer_scale_init_value=1e-6):
        super().__init__()

        self.block34 = ConvNeXtBlock34(
            drop_path_prob=drop_path_probs[0],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block35 = ConvNeXtBlock35(
            drop_path_prob=drop_path_probs[1],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block36 = ConvNeXtBlock36(
            drop_path_prob=drop_path_probs[2],
            layer_scale_init_value=layer_scale_init_value,
        )


    def forward(self, x):
        x = self.block34(x)

        x = self.block35(x)

        x = self.block36(x)

        return x
