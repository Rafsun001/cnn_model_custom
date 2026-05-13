import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignedBoxIoUBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, boxes1, boxes2):
        left_top = torch.maximum(boxes1[:, :2], boxes2[:, :2])
        right_bottom = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])

        intersection_wh = (right_bottom - left_top).clamp(min=0)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

        area1_wh = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0)
        area2_wh = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0)

        area1 = area1_wh[:, 0] * area1_wh[:, 1]
        area2 = area2_wh[:, 0] * area2_wh[:, 1]

        union = area1 + area2 - intersection
        iou = intersection / union.clamp(min=1e-6)
        return iou
