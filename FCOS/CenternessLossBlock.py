import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class CenternessLossBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, centerness_logits, centerness_targets, labels):
        positive_mask = labels > 0

        if not positive_mask.any():
            return centerness_logits.sum() * 0.0

        positive_logits = centerness_logits[positive_mask]
        positive_targets = centerness_targets[positive_mask]

        loss = F.binary_cross_entropy_with_logits(
            positive_logits,
            positive_targets,
            reduction="sum",
        )

        normalizer = positive_mask.sum().clamp(min=1).to(dtype=loss.dtype)
        return loss / normalizer
