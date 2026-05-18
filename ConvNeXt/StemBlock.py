import torch.nn as nn


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.patchify_conv = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
        )

        self.layer_norm = nn.LayerNorm(128, eps=1e-6)

    def forward(self, x):
        x = self.patchify_conv(x)


        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)

        return x
