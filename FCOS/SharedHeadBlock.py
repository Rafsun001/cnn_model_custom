import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ClassificationTowerBlock import ClassificationTowerBlock
from RegressionTowerBlock import RegressionTowerBlock
from PredictionBlock import PredictionBlock

class SharedHeadBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.classification_tower = ClassificationTowerBlock()
        self.regression_tower = RegressionTowerBlock()
        self.prediction = PredictionBlock()

    def forward(self, x):
        classification_feature = self.classification_tower(x)
        regression_feature = self.regression_tower(x)
        classification, centerness, regression = self.prediction(classification_feature, regression_feature)
        return classification, centerness, regression
