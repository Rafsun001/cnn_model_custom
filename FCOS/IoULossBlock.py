import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from DistanceToBoxBlock import DistanceToBoxBlock
from AlignedBoxIoUBlock import AlignedBoxIoUBlock

class IoULossBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.distance_to_box = DistanceToBoxBlock()
        self.aligned_iou = AlignedBoxIoUBlock()

    def forward(self, locations, regression_distances, regression_targets, labels):
        positive_mask = labels > 0

        if not positive_mask.any():
            return regression_distances.sum() * 0.0

        batch_size, total_locations, _ = regression_distances.shape
        expanded_locations = locations[None, :, :].expand(batch_size, total_locations, 2)

        positive_locations = expanded_locations[positive_mask]
        positive_predictions = regression_distances[positive_mask]
        positive_targets = regression_targets[positive_mask]

        predicted_boxes = self.distance_to_box(positive_locations, positive_predictions)
        target_boxes = self.distance_to_box(positive_locations, positive_targets)

        iou = self.aligned_iou(predicted_boxes, target_boxes)
        loss = -torch.log(iou.clamp(min=1e-6))

        normalizer = positive_mask.sum().clamp(min=1).to(dtype=loss.dtype)
        return loss.sum() / normalizer
