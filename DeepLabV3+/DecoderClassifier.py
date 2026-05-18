import torch.nn as nn


class DecoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
