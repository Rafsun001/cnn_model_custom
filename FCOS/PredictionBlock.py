import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from Constants import NUM_CLASSES

class PredictionBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.classification_pred = nn.Conv2d(256, NUM_CLASSES, kernel_size=3, stride=1, padding=1, bias=True)
        self.centerness_pred = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.regression_pred = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, classification_feature, regression_feature):
        classification = self.classification_pred(classification_feature)

        centerness = self.centerness_pred(classification_feature)

        regression = self.regression_pred(regression_feature)
        return classification, centerness, regression
