import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossBlock(nn.Module):

    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classification_logits, labels):
        target_classes = torch.zeros_like(classification_logits)

        positive_mask = labels > 0

        if positive_mask.any():
            positive_labels = labels[positive_mask] - 1
            target_classes[positive_mask] = F.one_hot(
                positive_labels,
                num_classes=self.num_classes,
            ).to(dtype=classification_logits.dtype)

        probability = torch.sigmoid(classification_logits)

        binary_cross_entropy = F.binary_cross_entropy_with_logits(
            classification_logits,
            target_classes,
            reduction="none",
        )

        p_t = probability * target_classes + (1.0 - probability) * (1.0 - target_classes)
        alpha_factor = self.alpha * target_classes + (1.0 - self.alpha) * (1.0 - target_classes)
        modulating_factor = (1.0 - p_t).pow(self.gamma)

        loss = alpha_factor * modulating_factor * binary_cross_entropy
        normalizer = positive_mask.sum().clamp(min=1).to(dtype=loss.dtype)

        return loss.sum() / normalizer
