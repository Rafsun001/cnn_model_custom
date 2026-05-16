import torch
import torch.nn as nn
from NeckUpsample1 import NeckUpsample1
from NeckConcat1 import NeckConcat1
from NeckC3k21 import NeckC3k21
from NeckUpsample2 import NeckUpsample2
from NeckConcat2 import NeckConcat2
from NeckC3k22 import NeckC3k22
from NeckConv1 import NeckConv1
from NeckConcat3 import NeckConcat3
from NeckC3k23 import NeckC3k23
from NeckConv2 import NeckConv2
from NeckConcat4 import NeckConcat4
from NeckC3k24 import NeckC3k24

class NeckStage(nn.Module):

    def __init__(self):
        super().__init__()

        self.upsample11 = NeckUpsample1()
        self.concat12 = NeckConcat1()
        self.c3k2_13 = NeckC3k21()

        self.upsample14 = NeckUpsample2()
        self.concat15 = NeckConcat2()
        self.c3k2_16 = NeckC3k22()

        self.conv17 = NeckConv1()
        self.concat18 = NeckConcat3()
        self.c3k2_19 = NeckC3k23()

        self.conv20 = NeckConv2()
        self.concat21 = NeckConcat4()
        self.c3k2_22 = NeckC3k24()

    def forward(self, p3_backbone, p4_backbone, p5_backbone):
        x = self.upsample11(p5_backbone)
        x = self.concat12(x, p4_backbone)
        neck_p4_first = self.c3k2_13(x)

        x = self.upsample14(neck_p4_first)
        x = self.concat15(x, p3_backbone)
        p3 = self.c3k2_16(x)

        x = self.conv17(p3)
        x = self.concat18(x, neck_p4_first)
        p4 = self.c3k2_19(x)

        x = self.conv20(p4)
        x = self.concat21(x, p5_backbone)
        p5 = self.c3k2_22(x)

        return p3, p4, p5
