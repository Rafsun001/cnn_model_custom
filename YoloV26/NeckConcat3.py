import torch
import torch.nn as nn

class NeckConcat3(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        return out
