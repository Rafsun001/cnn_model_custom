import torch
import torch.nn as nn

from BackboneStage import BackboneStage
from NeckStage import NeckStage
from HeadStage import HeadStage


class YOLO26FinalBlock(nn.Module):
    """
    Full YOLO26-style model block.

    Flow:
        input image
            ↓
        BackboneStage
            ↓
        NeckStage
            ↓
        HeadStage

    Training output:
        {
            "one2many": [P3, P4, P5],
            "one2one":  [P3, P4, P5],
        }

    Eval output:
        detections = [B, 300, 6]

    Detection format:
        [x1, y1, x2, y2, score, class_id]
    """

    def __init__(self):
        super().__init__()

        self.backbone = BackboneStage()
        self.neck = NeckStage()
        self.head = HeadStage()

    def forward(self, x):
        p3, p4, p5 = self.forward_features(x)
        output = self.head(p3, p4, p5)
        return output

    def forward_features(self, x):
        """
        Returns final neck features:
            p3 = [B, 64, 80, 80]
            p4 = [B, 128, 40, 40]
            p5 = [B, 256, 20, 20]
        """

        p3_backbone, p4_backbone, p5_backbone = self.backbone(x)

        p3, p4, p5 = self.neck(
            p3_backbone=p3_backbone,
            p4_backbone=p4_backbone,
            p5_backbone=p5_backbone,
        )

        return p3, p4, p5

    def forward_raw_predictions(self, x):
        """
        Returns raw dual-head predictions even if the model is in eval mode.

        This is useful for validation loss calculation, because in eval mode
        the normal forward() returns decoded detections instead of raw feature maps.

        Output:
            {
                "one2many": [P3, P4, P5],
                "one2one":  [P3, P4, P5],
            }
        """

        p3, p4, p5 = self.forward_features(x)

        detect_module = self.head.detect23

        p3_one2many, p3_one2one = detect_module.detect_p3(p3)
        p4_one2many, p4_one2one = detect_module.detect_p4(p4)
        p5_one2many, p5_one2one = detect_module.detect_p5(p5)

        predictions = {
            "one2many": [
                p3_one2many,
                p4_one2many,
                p5_one2many,
            ],
            "one2one": [
                p3_one2one,
                p4_one2one,
                p5_one2one,
            ],
        }

        return predictions

    def bias_init(self):
        """
        Initialize detection head biases.
        Call this once after model creation if needed.
        """

        self.head.bias_init()

    def fuse(self):
        """
        Remove one-to-many branches for inference/export style use.
        """

        self.head.fuse()