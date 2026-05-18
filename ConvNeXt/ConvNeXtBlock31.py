import torch
import torch.nn as nn

from DropPathBlock import DropPathBlock


class ConvNeXtBlock31(nn.Module):
    def __init__(self, drop_path_prob=0.0, layer_scale_init_value=1e-6):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=512,
            bias=True,
        )

        self.layer_norm = nn.LayerNorm(512, eps=1e-6)

        self.pointwise_expand = nn.Linear(
            in_features=512,
            out_features=2048,
            bias=True,
        )

        self.gelu = nn.GELU()

        self.pointwise_project = nn.Linear(
            in_features=2048,
            out_features=512,
            bias=True,
        )

        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones(512),
            requires_grad=True,
        )

        self.drop_path = DropPathBlock(drop_path_prob)

    def forward(self, x):
        identity = x

        out = self.depthwise_conv(x)


        out = out.permute(0, 2, 3, 1)

        out = self.layer_norm(out)
        out = self.pointwise_expand(out)
        out = self.gelu(out)
        out = self.pointwise_project(out)


        out = self.layer_scale * out


        out = out.permute(0, 3, 1, 2)


        out = self.drop_path(out)

        out = out + identity
        return out
