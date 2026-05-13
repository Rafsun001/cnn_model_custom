import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ConfigBlock import ConfigBlock
from ClipBoxesToImageBlock import ClipBoxesToImageBlock
from RemoveSmallBoxesBlock import RemoveSmallBoxesBlock
from SingleLevelDecodeBlock import SingleLevelDecodeBlock
from ClassWiseNMSBlock import ClassWiseNMSBlock

class SingleImagePostProcessBlock(nn.Module):

    def __init__(
        self,
        num_classes: int,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.6,
        topk_candidates: int = 1000,
        max_detections_per_image: int = 100,
        min_box_size: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.decode_single_level = SingleLevelDecodeBlock(
            num_classes=num_classes,
            score_threshold=score_threshold,
            topk_candidates=topk_candidates,
        )
        self.clip_boxes = ClipBoxesToImageBlock()
        self.remove_small_boxes = RemoveSmallBoxesBlock(min_size=min_box_size)
        self.class_wise_nms = ClassWiseNMSBlock(nms_threshold=nms_threshold)
        self.max_detections_per_image = max_detections_per_image

    def forward(self, predictions, locations_per_level, image_height: int, image_width: int, image_index: int):
        all_boxes = []
        all_scores = []
        all_labels = []

        for level_index, level_name in enumerate(ConfigBlock.LEVEL_NAMES):
            level_outputs = predictions[level_name]

            classification_logits = level_outputs["classification"][image_index]
            centerness_logits = level_outputs["centerness"][image_index]
            regression_distances = level_outputs["regression"][image_index]
            locations = locations_per_level[level_index]

            boxes, scores, labels = self.decode_single_level(
                locations=locations,
                classification_logits=classification_logits,
                centerness_logits=centerness_logits,
                regression_distances=regression_distances,
            )

            if boxes.numel() == 0:
                continue

            boxes = self.clip_boxes(boxes, image_height=image_height, image_width=image_width)
            keep = self.remove_small_boxes(boxes)

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if boxes.numel() == 0:
                continue

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        device = next(iter(predictions.values()))["classification"].device

        if len(all_boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            }

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)

        keep = self.class_wise_nms(boxes, scores, labels)

        if keep.numel() > self.max_detections_per_image:
            keep = keep[:self.max_detections_per_image]

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
