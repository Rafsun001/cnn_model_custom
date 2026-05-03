import torch.nn as nn


class ResidualBlock13(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)


        self.conv2 = nn.Conv2d(
            256, 256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(256)


        self.conv3 = nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)

        return out
