import torch
import torch.nn as nn


def autopad(kernel_size, padding=None, dilation=1):
    if padding is not None:
        return padding

    if isinstance(kernel_size, int):
        effective_kernel = dilation * (kernel_size - 1) + 1
        return effective_kernel // 2

    effective_kernel = [dilation * (k - 1) + 1 for k in kernel_size]
    return [k // 2 for k in effective_kernel]


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=1.0):
        super().__init__()

        hidden_channels = int(c2 * e)

        self.cv1 = Conv(c1, hidden_channels, k=3, s=1)
        self.cv2 = Conv(hidden_channels, c2, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))

        if self.add:
            out = out + x

        return out


class C3k(nn.Module):
    """
    C3k inner block.

    This is used when c3k=True.
    It keeps the same input/output channels.
    """

    def __init__(self, c1, c2, n=2, shortcut=True):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Bottleneck(
                    c1=c1 if i == 0 else c2,
                    c2=c2,
                    shortcut=shortcut,
                    e=1.0,
                )
                for i in range(n)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class BackboneC3k24(nn.Module):
    """
    Official-style C3k2 block

    Configuration:
        C3k2(c1=256, c2=256, n=1, c3k=True, e=0.5)

    Input : [B, 256, H, W]
    Output: [B, 256, H, W]

    Flow:
        cv1: 256 -> 256
        split: 128 + 128
        C3k on second 128
        concat: 128 + 128 + 128 = 384
        cv2: 384 -> 256
    """

    def __init__(self):
        super().__init__()

        self.hidden_channels = 128

        self.cv1 = Conv(
            c1=256,
            c2=256,
            k=1,
            s=1,
        )

        self.m = nn.ModuleList(
            [
                C3k(
                    c1=128,
                    c2=128,
                    n=2,
                    shortcut=True,
                )
            ]
        )

        self.cv2 = Conv(
            c1=384,
            c2=256,
            k=1,
            s=1,
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        for block in self.m:
            y.append(block(y[-1]))

        out = torch.cat(y, dim=1)
        out = self.cv2(out)

        return out