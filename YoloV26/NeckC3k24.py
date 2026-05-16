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


class Attention(nn.Module):
    def __init__(self, channels, num_heads=2, attn_ratio=0.5):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads

        self.head_dim = channels // num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim ** -0.5

        qkv_channels = num_heads * (
            self.key_dim + self.key_dim + self.head_dim
        )

        self.qkv = Conv(
            c1=channels,
            c2=qkv_channels,
            k=1,
            s=1,
            act=False,
        )

        self.proj = Conv(
            c1=channels,
            c2=channels,
            k=1,
            s=1,
            act=False,
        )

        self.pe = Conv(
            c1=channels,
            c2=channels,
            k=3,
            s=1,
            g=channels,
            act=False,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        num_positions = height * width

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            self.num_heads,
            self.key_dim + self.key_dim + self.head_dim,
            num_positions,
        )

        q, k, v = qkv.split(
            [self.key_dim, self.key_dim, self.head_dim],
            dim=2,
        )

        attn = torch.matmul(q.transpose(-2, -1), k)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(batch_size, channels, height, width)

        v_image = v.reshape(batch_size, channels, height, width)
        out = out + self.pe(v_image)

        out = self.proj(out)

        return out


class PSABlock(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()

        self.attn = Attention(
            channels=channels,
            num_heads=max(channels // 64, 1),
            attn_ratio=0.5,
        )

        self.ffn = nn.Sequential(
            Conv(
                c1=channels,
                c2=channels * 2,
                k=1,
                s=1,
                act=True,
            ),
            Conv(
                c1=channels * 2,
                c2=channels,
                k=1,
                s=1,
                act=False,
            ),
        )

        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            x = x + self.attn(x)
            x = x + self.ffn(x)
        else:
            x = self.attn(x)
            x = self.ffn(x)

        return x


class C3kWithAttention(nn.Module):
    """
    Inner block for attention-enabled C3k2.

    Flow:
        Bottleneck
        PSA block
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            Bottleneck(
                c1=channels,
                c2=channels,
                shortcut=True,
                e=1.0,
            ),
            PSABlock(
                channels=channels,
                shortcut=True,
            ),
        )

    def forward(self, x):
        return self.block(x)


class NeckC3k24(nn.Module):
    """
    Neck C3k2 block 4 with attention

    Configuration:
        C3k2(c1=384, c2=256, n=1, c3k=True, e=0.5, attn=True)

    Input : [B, 384, H, W]
    Output: [B, 256, H, W]

    Flow:
        cv1: 384 -> 256
        split: 128 + 128
        Bottleneck + PSA on second 128
        concat: 128 + 128 + 128 = 384
        cv2: 384 -> 256
    """

    def __init__(self):
        super().__init__()

        self.hidden_channels = 128

        self.cv1 = Conv(
            c1=384,
            c2=256,
            k=1,
            s=1,
        )

        self.m = nn.ModuleList(
            [
                C3kWithAttention(
                    channels=128,
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