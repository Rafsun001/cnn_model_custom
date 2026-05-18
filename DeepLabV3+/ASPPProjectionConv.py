import torch.nn as nn


class ASPPProjectionConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1280, 256, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
