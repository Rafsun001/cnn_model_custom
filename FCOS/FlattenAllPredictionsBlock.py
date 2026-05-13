import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigBlock import ConfigBlock
from FlattenClassificationBlock import FlattenClassificationBlock
from FlattenCenternessBlock import FlattenCenternessBlock
from FlattenRegressionBlock import FlattenRegressionBlock

class FlattenAllPredictionsBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten_classification = FlattenClassificationBlock()
        self.flatten_centerness = FlattenCenternessBlock()
        self.flatten_regression = FlattenRegressionBlock()

    def forward(self, predictions: Dict[str, Dict[str, torch.Tensor]]):
        all_classification = []
        all_centerness = []
        all_regression = []

        for level_name in ConfigBlock.LEVEL_NAMES:
            level_outputs = predictions[level_name]

            classification = self.flatten_classification(level_outputs["classification"])
            centerness = self.flatten_centerness(level_outputs["centerness"])
            regression = self.flatten_regression(level_outputs["regression"])

            all_classification.append(classification)
            all_centerness.append(centerness)
            all_regression.append(regression)

        classification_logits = torch.cat(all_classification, dim=1)
        centerness_logits = torch.cat(all_centerness, dim=1)
        regression_distances = torch.cat(all_regression, dim=1)

        return classification_logits, centerness_logits, regression_distances
