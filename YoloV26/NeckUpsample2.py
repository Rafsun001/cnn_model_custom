import torch
import torch.nn as nn


class NeckUpsample2(nn.Module):
    """
    Neck upsample 2

    Input : [B, C, H, W]
    Output: [B, C, 2H, 2W]

    Uses nearest-neighbor upsampling.
    """

    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

    def forward(self, x):
        x = self.upsample(x)
        return x