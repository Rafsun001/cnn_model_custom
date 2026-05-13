import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigBlock import ConfigBlock
from SingleLevelLocationGeneratorBlock import SingleLevelLocationGeneratorBlock

class AllLevelLocationGeneratorBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.single_level_generator = SingleLevelLocationGeneratorBlock()

    def forward(self, predictions: Dict[str, Dict[str, torch.Tensor]]):
        locations_per_level = []

        for level_name in ConfigBlock.LEVEL_NAMES:
            classification = predictions[level_name]["classification"]
            _, _, feature_height, feature_width = classification.shape
            stride = ConfigBlock.STRIDES[level_name]
            locations = self.single_level_generator(
                feature_height=feature_height,
                feature_width=feature_width,
                stride=stride,
                device=classification.device,
            )
            locations_per_level.append(locations)

        return locations_per_level
