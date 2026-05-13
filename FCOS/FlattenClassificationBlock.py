import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlattenClassificationBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, classification):
        n, c, h, w = classification.shape
        classification = classification.permute(0, 2, 3, 1).contiguous()
        classification = classification.view(n, h * w, c)
        return classification
