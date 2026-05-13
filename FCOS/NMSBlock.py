import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PairwiseBoxIoUBlock import PairwiseBoxIoUBlock

class NMSBlock(nn.Module):

    def __init__(self, nms_threshold: float = 0.6):
        super().__init__()
        self.nms_threshold = nms_threshold
        self.pairwise_iou = PairwiseBoxIoUBlock()

    def forward(self, boxes, scores):
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            current = order[0]
            keep.append(current)

            if order.numel() == 1:
                break

            current_box = boxes[current].view(1, 4)
            remaining_boxes = boxes[order[1:]]

            ious = self.pairwise_iou(current_box, remaining_boxes).view(-1)
            order = order[1:][ious <= self.nms_threshold]

        return torch.stack(keep, dim=0)
