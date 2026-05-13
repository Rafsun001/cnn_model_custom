import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from AllLevelLocationGeneratorBlock import AllLevelLocationGeneratorBlock
from FlattenAllPredictionsBlock import FlattenAllPredictionsBlock
from AllImageTargetAssignmentBlock import AllImageTargetAssignmentBlock
from FocalLossBlock import FocalLossBlock
from IoULossBlock import IoULossBlock
from CenternessLossBlock import CenternessLossBlock

class LossComputationBlock(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.location_generator = AllLevelLocationGeneratorBlock()
        self.target_assigner = AllImageTargetAssignmentBlock(num_classes=num_classes)
        self.flatten_predictions = FlattenAllPredictionsBlock()
        self.focal_loss = FocalLossBlock(num_classes=num_classes)
        self.iou_loss = IoULossBlock()
        self.centerness_loss = CenternessLossBlock()

    def forward(self, predictions, targets):
        if targets is None:
            raise ValueError("Training mode requires targets.")

        locations_per_level = self.location_generator(predictions)
        locations = torch.cat(locations_per_level, dim=0)

        labels, regression_targets, centerness_targets = self.target_assigner(
            locations_per_level=locations_per_level,
            targets=targets,
        )

        classification_logits, centerness_logits, regression_distances = self.flatten_predictions(predictions)

        classification_loss = self.focal_loss(classification_logits, labels)
        regression_loss = self.iou_loss(locations, regression_distances, regression_targets, labels)
        centerness_loss = self.centerness_loss(centerness_logits, centerness_targets, labels)

        total_loss = classification_loss + regression_loss + centerness_loss

        return {
            "loss_total": total_loss,
            "loss_cls": classification_loss,
            "loss_reg": regression_loss,
            "loss_centerness": centerness_loss,
        }
