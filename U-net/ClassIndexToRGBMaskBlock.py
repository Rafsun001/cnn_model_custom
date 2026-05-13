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

class ClassIndexToRGBMaskBlock(nn.Module):

    def __init__(self, class_to_color: Dict[int, Tuple[int, int, int]]):
        super().__init__()
        self.class_to_color = class_to_color

    def forward(self, class_mask: np.ndarray):
        height, width = class_mask.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, rgb_color in self.class_to_color.items():
            color_mask[class_mask == class_id] = rgb_color

        return color_mask
