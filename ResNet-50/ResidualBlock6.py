import torch.nn as nn


class ResidualBlock6(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)


        self.conv2 = nn.Conv2d(
            128, 128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(128)


        self.conv3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)

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
