import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from DistanceToBoxBlock import DistanceToBoxBlock

class SingleLevelDecodeBlock(nn.Module):

    def __init__(
        self,
        num_classes: int,
        score_threshold: float = 0.05,
        topk_candidates: int = 1000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.topk_candidates = topk_candidates
        self.distance_to_box = DistanceToBoxBlock()

    def forward(self, locations, classification_logits, centerness_logits, regression_distances):

        classification_logits = classification_logits.permute(1, 2, 0).reshape(-1, self.num_classes)
        centerness_logits = centerness_logits.permute(1, 2, 0).reshape(-1)
        regression_distances = regression_distances.permute(1, 2, 0).reshape(-1, 4)

        classification_scores = torch.sigmoid(classification_logits)
        centerness_scores = torch.sigmoid(centerness_logits)

        final_scores = classification_scores * centerness_scores[:, None]

        candidate_mask = final_scores > self.score_threshold
        candidate_indices = candidate_mask.nonzero(as_tuple=False)

        if candidate_indices.numel() == 0:
            empty_boxes = regression_distances.new_zeros((0, 4))
            empty_scores = regression_distances.new_zeros((0,))
            empty_labels = torch.zeros((0,), dtype=torch.long, device=regression_distances.device)
            return empty_boxes, empty_scores, empty_labels

        point_indices = candidate_indices[:, 0]
        class_indices = candidate_indices[:, 1]

        scores = final_scores[point_indices, class_indices]
        labels = class_indices + 1

        if scores.numel() > self.topk_candidates:
            scores, topk_indices = scores.topk(self.topk_candidates)
            point_indices = point_indices[topk_indices]
            labels = labels[topk_indices]

        selected_locations = locations[point_indices]
        selected_distances = regression_distances[point_indices]

        boxes = self.distance_to_box(selected_locations, selected_distances)

        return boxes, scores, labels
