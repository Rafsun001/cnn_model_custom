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

class RGBMaskToClassIndexBlock(nn.Module):

    def __init__(self, color_to_class: Dict[Tuple[int, int, int], int], ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.color_to_class = color_to_class
        self.ignore_index = ignore_index

    def forward(self, mask_rgb: np.ndarray):
        height, width, _ = mask_rgb.shape
        class_mask = np.full((height, width), fill_value=self.ignore_index, dtype=np.int64)

        for rgb_color, class_id in self.color_to_class.items():
            rgb_color_array = np.array(rgb_color, dtype=np.uint8)
            matched_pixels = np.all(mask_rgb == rgb_color_array, axis=-1)
            class_mask[matched_pixels] = class_id

        return class_mask
