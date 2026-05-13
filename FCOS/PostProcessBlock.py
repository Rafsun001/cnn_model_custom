import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from AllLevelLocationGeneratorBlock import AllLevelLocationGeneratorBlock
from SingleImagePostProcessBlock import SingleImagePostProcessBlock

class PostProcessBlock(nn.Module):

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
        self.location_generator = AllLevelLocationGeneratorBlock()
        self.single_image_postprocess = SingleImagePostProcessBlock(
            num_classes=num_classes,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            topk_candidates=topk_candidates,
            max_detections_per_image=max_detections_per_image,
            min_box_size=min_box_size,
        )

    def forward(self, predictions, image_height: int, image_width: int):
        locations_per_level = self.location_generator(predictions)

        batch_size = predictions["P3"]["classification"].shape[0]
        results = []

        for image_index in range(batch_size):
            result = self.single_image_postprocess(
                predictions=predictions,
                locations_per_level=locations_per_level,
                image_height=image_height,
                image_width=image_width,
                image_index=image_index,
            )
            results.append(result)

        return results
