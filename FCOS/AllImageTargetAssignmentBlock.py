import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from RegressionRangeBuilderBlock import RegressionRangeBuilderBlock
from SingleImageTargetAssignmentBlock import SingleImageTargetAssignmentBlock

class AllImageTargetAssignmentBlock(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.range_builder = RegressionRangeBuilderBlock()
        self.single_image_assigner = SingleImageTargetAssignmentBlock(num_classes=num_classes)

    def forward(self, locations_per_level: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        locations = torch.cat(locations_per_level, dim=0)
        regression_ranges = self.range_builder(locations_per_level)

        all_labels = []
        all_regression_targets = []
        all_centerness_targets = []

        for target in targets:
            labels, regression_targets, centerness_targets = self.single_image_assigner(
                locations=locations,
                regression_ranges=regression_ranges,
                target=target,
            )

            all_labels.append(labels)
            all_regression_targets.append(regression_targets)
            all_centerness_targets.append(centerness_targets)

        labels = torch.stack(all_labels, dim=0)
        regression_targets = torch.stack(all_regression_targets, dim=0)
        centerness_targets = torch.stack(all_centerness_targets, dim=0)

        return labels, regression_targets, centerness_targets
