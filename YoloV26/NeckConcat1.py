import torch
import torch.nn as nn


class NeckConcat1(nn.Module):
    """
    Neck concat 1

    Concatenates two feature maps along channel dimension.

    Input:
        x1 = [B, C1, H, W]
        x2 = [B, C2, H, W]

    Output:
        out = [B, C1 + C2, H, W]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        return out