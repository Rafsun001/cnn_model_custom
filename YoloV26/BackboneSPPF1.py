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


class BackboneSPPF1(nn.Module):
    """
    Official-style SPPF block

    Configuration:
        SPPF(c1=256, c2=256, k=5, n=3, shortcut=True)

    Input : [B, 256, H, W]
    Output: [B, 256, H, W]

    Important:
        The first 1x1 conv uses act=False.
    """

    def __init__(self):
        super().__init__()

        self.shortcut = True

        hidden_channels = 128

        self.cv1 = Conv(
            c1=256,
            c2=hidden_channels,
            k=1,
            s=1,
            act=False,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=5,
            stride=1,
            padding=5 // 2,
        )

        self.cv2 = Conv(
            c1=hidden_channels * 4,
            c2=256,
            k=1,
            s=1,
            act=True,
        )

    def forward(self, x):
        identity = x

        x = self.cv1(x)

        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        out = torch.cat([x, y1, y2, y3], dim=1)
        out = self.cv2(out)

        if self.shortcut:
            out = out + identity

        return out

""" 
#### ---> overview of the architecture:
Input
  ↓
1x1 Conv: 256 → 128
  ↓
MaxPool once
  ↓
MaxPool twice
  ↓
MaxPool three times
  ↓
Concatenate all features
  ↓
1x1 Conv: 512 → 256
  ↓
Add shortcut
  ↓
Output

#### ---> Detailed flow:

Input: [B, 256, H, W]
        |
        v
Conv 1x1: 256 → 128
        |
        v
x:  [B, 128, H, W]
        |
        +---------------------+
        |                     |
        v                     |
MaxPool 1                    |
        |                     |
        v                     |
y1: [B, 128, H, W]           |
        |                     |
        v                     |
MaxPool 2                    |
        |                     |
        v                     |
y2: [B, 128, H, W]           |
        |                     |
        v                     |
MaxPool 3                    |
        |                     |
        v                     |
y3: [B, 128, H, W]           |
        |                     |
        +----------+----------+
                   |
                   v
Concat: x + y1 + y2 + y3
                   |
                   v
[B, 512, H, W]
                   |
                   v
Conv 1x1: 512 → 256
                   |
                   v
Add shortcut
                   |
                   v
Output: [B, 256, H, W]

"""