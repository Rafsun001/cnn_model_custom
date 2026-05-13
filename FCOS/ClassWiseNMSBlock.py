import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from NMSBlock import NMSBlock

class ClassWiseNMSBlock(nn.Module):

    def __init__(self, nms_threshold: float = 0.6):
        super().__init__()
        self.nms = NMSBlock(nms_threshold=nms_threshold)

    def forward(self, boxes, scores, labels):
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        keep_indices = []

        unique_labels = labels.unique()
        for class_label in unique_labels:
            class_mask = labels == class_label
            class_indices = class_mask.nonzero(as_tuple=False).view(-1)

            class_keep = self.nms(boxes[class_indices], scores[class_indices])
            keep_indices.append(class_indices[class_keep])

        keep_indices = torch.cat(keep_indices, dim=0)
        keep_indices = keep_indices[scores[keep_indices].argsort(descending=True)]

        return keep_indices
