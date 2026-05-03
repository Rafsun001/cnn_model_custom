import torch.nn as nn

from ConvNeXtBlock7 import ConvNeXtBlock7
from ConvNeXtBlock8 import ConvNeXtBlock8
from ConvNeXtBlock9 import ConvNeXtBlock9
from ConvNeXtBlock10 import ConvNeXtBlock10
from ConvNeXtBlock11 import ConvNeXtBlock11
from ConvNeXtBlock12 import ConvNeXtBlock12
from ConvNeXtBlock13 import ConvNeXtBlock13
from ConvNeXtBlock14 import ConvNeXtBlock14
from ConvNeXtBlock15 import ConvNeXtBlock15
from ConvNeXtBlock16 import ConvNeXtBlock16
from ConvNeXtBlock17 import ConvNeXtBlock17
from ConvNeXtBlock18 import ConvNeXtBlock18
from ConvNeXtBlock19 import ConvNeXtBlock19
from ConvNeXtBlock20 import ConvNeXtBlock20
from ConvNeXtBlock21 import ConvNeXtBlock21
from ConvNeXtBlock22 import ConvNeXtBlock22
from ConvNeXtBlock23 import ConvNeXtBlock23
from ConvNeXtBlock24 import ConvNeXtBlock24
from ConvNeXtBlock25 import ConvNeXtBlock25
from ConvNeXtBlock26 import ConvNeXtBlock26
from ConvNeXtBlock27 import ConvNeXtBlock27
from ConvNeXtBlock28 import ConvNeXtBlock28
from ConvNeXtBlock29 import ConvNeXtBlock29
from ConvNeXtBlock30 import ConvNeXtBlock30
from ConvNeXtBlock31 import ConvNeXtBlock31
from ConvNeXtBlock32 import ConvNeXtBlock32
from ConvNeXtBlock33 import ConvNeXtBlock33


class Stage3(nn.Module):
    def __init__(self, drop_path_probs, layer_scale_init_value=1e-6):
        super().__init__()

        self.block7 = ConvNeXtBlock7(
            drop_path_prob=drop_path_probs[0],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block8 = ConvNeXtBlock8(
            drop_path_prob=drop_path_probs[1],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block9 = ConvNeXtBlock9(
            drop_path_prob=drop_path_probs[2],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block10 = ConvNeXtBlock10(
            drop_path_prob=drop_path_probs[3],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block11 = ConvNeXtBlock11(
            drop_path_prob=drop_path_probs[4],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block12 = ConvNeXtBlock12(
            drop_path_prob=drop_path_probs[5],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block13 = ConvNeXtBlock13(
            drop_path_prob=drop_path_probs[6],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block14 = ConvNeXtBlock14(
            drop_path_prob=drop_path_probs[7],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block15 = ConvNeXtBlock15(
            drop_path_prob=drop_path_probs[8],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block16 = ConvNeXtBlock16(
            drop_path_prob=drop_path_probs[9],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block17 = ConvNeXtBlock17(
            drop_path_prob=drop_path_probs[10],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block18 = ConvNeXtBlock18(
            drop_path_prob=drop_path_probs[11],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block19 = ConvNeXtBlock19(
            drop_path_prob=drop_path_probs[12],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block20 = ConvNeXtBlock20(
            drop_path_prob=drop_path_probs[13],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block21 = ConvNeXtBlock21(
            drop_path_prob=drop_path_probs[14],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block22 = ConvNeXtBlock22(
            drop_path_prob=drop_path_probs[15],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block23 = ConvNeXtBlock23(
            drop_path_prob=drop_path_probs[16],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block24 = ConvNeXtBlock24(
            drop_path_prob=drop_path_probs[17],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block25 = ConvNeXtBlock25(
            drop_path_prob=drop_path_probs[18],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block26 = ConvNeXtBlock26(
            drop_path_prob=drop_path_probs[19],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block27 = ConvNeXtBlock27(
            drop_path_prob=drop_path_probs[20],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block28 = ConvNeXtBlock28(
            drop_path_prob=drop_path_probs[21],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block29 = ConvNeXtBlock29(
            drop_path_prob=drop_path_probs[22],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block30 = ConvNeXtBlock30(
            drop_path_prob=drop_path_probs[23],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block31 = ConvNeXtBlock31(
            drop_path_prob=drop_path_probs[24],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block32 = ConvNeXtBlock32(
            drop_path_prob=drop_path_probs[25],
            layer_scale_init_value=layer_scale_init_value,
        )

        self.block33 = ConvNeXtBlock33(
            drop_path_prob=drop_path_probs[26],
            layer_scale_init_value=layer_scale_init_value,
        )


    def forward(self, x):
        x = self.block7(x)

        x = self.block8(x)

        x = self.block9(x)

        x = self.block10(x)

        x = self.block11(x)

        x = self.block12(x)

        x = self.block13(x)

        x = self.block14(x)

        x = self.block15(x)

        x = self.block16(x)

        x = self.block17(x)

        x = self.block18(x)

        x = self.block19(x)

        x = self.block20(x)

        x = self.block21(x)

        x = self.block22(x)

        x = self.block23(x)

        x = self.block24(x)

        x = self.block25(x)

        x = self.block26(x)

        x = self.block27(x)

        x = self.block28(x)

        x = self.block29(x)

        x = self.block30(x)

        x = self.block31(x)

        x = self.block32(x)

        x = self.block33(x)

        return x
