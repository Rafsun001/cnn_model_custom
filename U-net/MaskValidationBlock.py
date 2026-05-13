from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


NUM_CLASSES = 4
IGNORE_INDEX = 255
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SAVE_PATH = "unet_multiclass_explicit_best.pth"

class MaskValidationBlock(nn.Module):

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, masks: torch.Tensor):
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        if masks.dim() != 3:
            raise ValueError("Masks must have shape (N, H, W) for multi-class segmentation")

        if masks.dtype != torch.long:
            masks = masks.long()

        valid_pixels = masks != self.ignore_index

        if valid_pixels.any():
            min_value = masks[valid_pixels].min()
            max_value = masks[valid_pixels].max()

            if min_value < 0 or max_value >= self.num_classes:
                raise ValueError(
                    f"Mask class IDs must be in [0, {self.num_classes - 1}] "
                    f"or equal to ignore_index={self.ignore_index}. "
                    f"Found min={int(min_value)}, max={int(max_value)}."
                )

        return masks
