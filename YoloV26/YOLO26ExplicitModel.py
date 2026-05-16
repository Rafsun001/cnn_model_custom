import torch
import torch.nn as nn

from YOLO26FinalBlock import YOLO26FinalBlock


class YOLO26ExplicitModel(nn.Module):
    """
    Main YOLO26-style model wrapper.

    Normal behavior:

        model.train()
            output = {
                "one2many": [P3, P4, P5],
                "one2one":  [P3, P4, P5],
            }

        model.eval()
            output = [B, 300, 6]

    Raw prediction behavior:

        model(images, raw=True)

    This returns raw dual-head predictions even in eval mode.
    Useful for validation loss calculation.
    """

    def __init__(self):
        super().__init__()

        self.model = YOLO26FinalBlock()

    def forward(self, x, raw=False):
        if raw:
            return self.model.forward_raw_predictions(x)

        output = self.model(x)
        return output

    def bias_init(self):
        self.model.bias_init()

    def fuse(self):
        self.model.fuse()