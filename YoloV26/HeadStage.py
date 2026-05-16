import torch
import torch.nn as nn

from HeadDetect1 import HeadDetect1


class HeadStage(nn.Module):
    """
    Head stage wrapper.

    Input:
        p3 = [B, 64, 80, 80]
        p4 = [B, 128, 40, 40]
        p5 = [B, 256, 20, 20]

    Training output:
        {
            "one2many": [p3_pred, p4_pred, p5_pred],
            "one2one":  [p3_pred, p4_pred, p5_pred],
        }

    Eval output:
        detections = [B, 300, 6]
    """

    def __init__(self):
        super().__init__()

        self.detect23 = HeadDetect1()

    def forward(self, p3, p4, p5):
        output = self.detect23(p3, p4, p5)
        return output

    def bias_init(self):
        self.detect23.bias_init()

    def fuse(self):
        self.detect23.fuse()