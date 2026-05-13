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

class CrossEntropySegmentationLossBlock(nn.Module):

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, masks):
        if logits.dim() != 4:
            raise ValueError("Logits must have shape (N, C, H, W)")

        if masks.dim() != 3:
            raise ValueError("Masks must have shape (N, H, W)")

        if logits.shape[0] != masks.shape[0]:
            raise ValueError("Logits and masks must have the same batch size")

        if logits.shape[-2:] != masks.shape[-2:]:
            raise ValueError("Logits and masks must have the same height and width")

        return self.loss(logits, masks)
