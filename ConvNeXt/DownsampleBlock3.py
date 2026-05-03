import torch.nn as nn


class DownsampleBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        self.downsample_conv = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_conv(x)
        return x
