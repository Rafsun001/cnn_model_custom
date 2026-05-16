import torch
import torch.nn as nn

class BackboneInput1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor: (N, C, H, W)")

        if x.shape[1] != 3:
            raise ValueError("Input must have 3 channels")

        return x
