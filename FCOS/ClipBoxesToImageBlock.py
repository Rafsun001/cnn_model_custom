import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipBoxesToImageBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, boxes, image_height: int, image_width: int):
        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=image_width)
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=image_width)
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=image_height)
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=image_height)
        return boxes
