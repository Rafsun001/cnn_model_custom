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

class InputBlock(nn.Module):

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError("Input must be 4D: (N, C, H, W)")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input image must have {self.in_channels} channels")
        return x
