import torch.nn as nn


class ClassifierHeadBlock(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc(x)
        return x
