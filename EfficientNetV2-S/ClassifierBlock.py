import torch
import torch.nn as nn


class ClassifierBlock(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
