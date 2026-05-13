import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from Constants import NUM_CLASSES
from PredictionBlock import PredictionBlock
from ResNet50FinalBlock import ResNet50FinalBlock
from LossComputationBlock import LossComputationBlock
from PostProcessBlock import PostProcessBlock

class ResNet50EndToEndModel(nn.Module):

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.6,
        topk_candidates: int = 1000,
        max_detections_per_image: int = 100,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.model = ResNet50FinalBlock()

        self.loss_computation = LossComputationBlock(num_classes=num_classes)

        self.postprocess = PostProcessBlock(
            num_classes=num_classes,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            topk_candidates=topk_candidates,
            max_detections_per_image=max_detections_per_image,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        prior_probability = 0.01
        bias_value = -math.log((1.0 - prior_probability) / prior_probability)

        for module in self.modules():
            if isinstance(module, PredictionBlock):
                nn.init.constant_(module.classification_pred.bias, bias_value)

    def forward(self, images, targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        predictions = self.model(images)

        if self.training:
            if targets is None:
                raise ValueError("When the model is in training mode, targets must be provided.")
            return self.loss_computation(predictions, targets)

        image_height = images.shape[-2]
        image_width = images.shape[-1]

        detections = self.postprocess(
            predictions=predictions,
            image_height=image_height,
            image_width=image_width,
        )

        return detections
