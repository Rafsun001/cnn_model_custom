import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PredictionBlock import PredictionBlock
from ResNet50FinalBlock import ResNet50FinalBlock

class ResNet50Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = ResNet50FinalBlock()
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        prior_probability = 0.01
        bias_value = -math.log((1.0 - prior_probability) / prior_probability)
        for module in self.modules():
            if isinstance(module, PredictionBlock):
                nn.init.constant_(module.classification_pred.bias, bias_value)

    def forward(self, x):
        return self.model(x)
