import copy
import math
import torch
import torch.nn as nn

from YoloV26Config import NUM_CLASSES, REG_MAX


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


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(
            c1=c1,
            c2=c2,
            k=k,
            s=s,
            p=None,
            g=math.gcd(c1, c2),
            d=d,
            act=act,
        )


class HeadDetectP3(nn.Module):
    """
    YOLO26-style P3 detection head.

    Input:
        p3 = [B, 64, 80, 80]

    Output:
        one2many = [B, 84, 80, 80]
        one2one  = [B, 84, 80, 80]

    84 = 4 box values + 80 class logits
    because REG_MAX = 1.
    """

    def __init__(self):
        super().__init__()

        in_channels = 64

        box_hidden_channels = max(16, 64 // 4, 4 * REG_MAX)
        class_hidden_channels = max(64, min(NUM_CLASSES, 100))

        self.one2many_box = nn.Sequential(
            Conv(in_channels, box_hidden_channels, k=3, s=1),
            Conv(box_hidden_channels, box_hidden_channels, k=3, s=1),
            nn.Conv2d(box_hidden_channels, 4 * REG_MAX, kernel_size=1),
        )

        self.one2many_cls = nn.Sequential(
            nn.Sequential(
                DWConv(in_channels, in_channels, k=3, s=1),
                Conv(in_channels, class_hidden_channels, k=1, s=1),
            ),
            nn.Sequential(
                DWConv(class_hidden_channels, class_hidden_channels, k=3, s=1),
                Conv(class_hidden_channels, class_hidden_channels, k=1, s=1),
            ),
            nn.Conv2d(class_hidden_channels, NUM_CLASSES, kernel_size=1),
        )

        # Official end-to-end style: one-to-one heads are copies of one-to-many heads.
        self.one2one_box = copy.deepcopy(self.one2many_box)
        self.one2one_cls = copy.deepcopy(self.one2many_cls)

    def forward(self, x):
        # Official end-to-end style detaches features for the one-to-one branch.
        x_detached = x.detach()

        one2one_box = self.one2one_box(x_detached)
        one2one_cls = self.one2one_cls(x_detached)
        one2one = torch.cat([one2one_box, one2one_cls], dim=1)

        if self.one2many_box is None or self.one2many_cls is None:
            return one2one, one2one

        one2many_box = self.one2many_box(x)
        one2many_cls = self.one2many_cls(x)
        one2many = torch.cat([one2many_box, one2many_cls], dim=1)

        return one2many, one2one

    def bias_init(self, stride=8, image_size=640):
        # Box branch bias.
        if self.one2many_box is not None:
            self.one2many_box[-1].bias.data[:] = 1.0
        self.one2one_box[-1].bias.data[:] = 1.0

        # Class branch bias.
        bias_value = math.log(5 / NUM_CLASSES / (image_size / stride) ** 2)
        if self.one2many_cls is not None:
            self.one2many_cls[-1].bias.data[:NUM_CLASSES] = bias_value
        self.one2one_cls[-1].bias.data[:NUM_CLASSES] = bias_value

    def fuse(self):
        # For inference/export only, the one-to-many branch can be removed.
        self.one2many_box = None
        self.one2many_cls = None
