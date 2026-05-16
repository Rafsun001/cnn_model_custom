import torch.nn as nn


class GlobalAveragePoolingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x
