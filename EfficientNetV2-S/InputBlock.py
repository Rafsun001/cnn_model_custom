import torch.nn as nn


class InputBlock(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Input must be a 4D tensor: (batch_size, channels, height, width)")

        if x.shape[1] != self.image_channels:
            raise ValueError(f"Expected {self.image_channels} channels, but got {x.shape[1]}")

        return x
