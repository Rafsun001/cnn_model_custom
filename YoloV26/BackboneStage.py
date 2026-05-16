import torch
import torch.nn as nn
from BackboneInput1 import BackboneInput1
from BackboneConv1 import BackboneConv1
from BackboneConv2 import BackboneConv2
from BackboneC3k21 import BackboneC3k21
from BackboneConv3 import BackboneConv3
from BackboneC3k22 import BackboneC3k22
from BackboneConv4 import BackboneConv4
from BackboneC3k23 import BackboneC3k23
from BackboneConv5 import BackboneConv5
from BackboneC3k24 import BackboneC3k24
from BackboneSPPF1 import BackboneSPPF1
from BackboneC2PSA1 import BackboneC2PSA1

class BackboneStage(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_block = BackboneInput1()

        self.conv0 = BackboneConv1()
        self.conv1 = BackboneConv2()
        self.c3k2_2 = BackboneC3k21()
        self.conv3 = BackboneConv3()
        self.c3k2_4 = BackboneC3k22()
        self.conv5 = BackboneConv4()
        self.c3k2_6 = BackboneC3k23()
        self.conv7 = BackboneConv5()
        self.c3k2_8 = BackboneC3k24()
        self.sppf9 = BackboneSPPF1()
        self.c2psa10 = BackboneC2PSA1()

    def forward(self, x):
        x = self.input_block(x)

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.c3k2_2(x)

        x = self.conv3(x)
        p3_backbone = self.c3k2_4(x)

        x = self.conv5(p3_backbone)
        p4_backbone = self.c3k2_6(x)

        x = self.conv7(p4_backbone)
        x = self.c3k2_8(x)
        x = self.sppf9(x)
        p5_backbone = self.c2psa10(x)

        return p3_backbone, p4_backbone, p5_backbone
