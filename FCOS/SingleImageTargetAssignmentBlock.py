import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleImageTargetAssignmentBlock(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, locations, regression_ranges, target):
        device = locations.device
        total_locations = locations.shape[0]

        labels = torch.zeros((total_locations,), dtype=torch.long, device=device)
        regression_targets = torch.zeros((total_locations, 4), dtype=torch.float32, device=device)
        centerness_targets = torch.zeros((total_locations,), dtype=torch.float32, device=device)

        if target is None:
            return labels, regression_targets, centerness_targets

        boxes = target.get("boxes", None)
        target_labels = target.get("labels", None)

        if boxes is None or target_labels is None:
            return labels, regression_targets, centerness_targets

        boxes = boxes.to(device=device, dtype=torch.float32)
        target_labels = target_labels.to(device=device, dtype=torch.long)

        if boxes.numel() == 0:
            return labels, regression_targets, centerness_targets

        if boxes.dim() != 2 or boxes.shape[1] != 4:
            raise ValueError("target['boxes'] must have shape (num_boxes, 4)")

        if target_labels.dim() != 1 or target_labels.shape[0] != boxes.shape[0]:
            raise ValueError("target['labels'] must have shape (num_boxes,)")

        if torch.any(target_labels < 1) or torch.any(target_labels > self.num_classes):
            raise ValueError("FCOS labels must be in the range [1, num_classes]. Use 0 only for background internally.")

        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        valid_boxes = (box_widths > 0) & (box_heights > 0)

        boxes = boxes[valid_boxes]
        target_labels = target_labels[valid_boxes]

        if boxes.numel() == 0:
            return labels, regression_targets, centerness_targets

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        xs = locations[:, 0]
        ys = locations[:, 1]

        left = xs[:, None] - boxes[:, 0][None, :]
        top = ys[:, None] - boxes[:, 1][None, :]
        right = boxes[:, 2][None, :] - xs[:, None]
        bottom = boxes[:, 3][None, :] - ys[:, None]

        all_regression_targets = torch.stack((left, top, right, bottom), dim=2)

        inside_box = all_regression_targets.min(dim=2).values > 0
        max_regression_distance = all_regression_targets.max(dim=2).values

        lower_bound = regression_ranges[:, 0][:, None]
        upper_bound = regression_ranges[:, 1][:, None]

        inside_level_range = (max_regression_distance >= lower_bound) & (max_regression_distance <= upper_bound)
        valid_match = inside_box & inside_level_range

        inf = torch.tensor(1e8, dtype=torch.float32, device=device)
        candidate_areas = areas[None, :].repeat(total_locations, 1)
        candidate_areas[~valid_match] = inf

        min_area, matched_box_index = candidate_areas.min(dim=1)
        positive_locations = min_area < inf

        labels[positive_locations] = target_labels[matched_box_index[positive_locations]]
        regression_targets[positive_locations] = all_regression_targets[
            positive_locations,
            matched_box_index[positive_locations],
        ]

        positive_regression_targets = regression_targets[positive_locations]

        if positive_regression_targets.numel() > 0:
            left_right = positive_regression_targets[:, [0, 2]]
            top_bottom = positive_regression_targets[:, [1, 3]]

            min_left_right = left_right.min(dim=1).values
            max_left_right = left_right.max(dim=1).values.clamp(min=1e-6)

            min_top_bottom = top_bottom.min(dim=1).values
            max_top_bottom = top_bottom.max(dim=1).values.clamp(min=1e-6)

            centerness = torch.sqrt(
                (min_left_right / max_left_right) *
                (min_top_bottom / max_top_bottom)
            )
            centerness_targets[positive_locations] = centerness

        return labels, regression_targets, centerness_targets
