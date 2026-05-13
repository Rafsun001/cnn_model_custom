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

class PixelAccuracyBlock(nn.Module):

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, masks):
        predictions = torch.argmax(logits, dim=1)
        valid_pixels = masks != self.ignore_index

        if valid_pixels.sum() == 0:
            return logits.new_tensor(0.0)

        correct = (predictions[valid_pixels] == masks[valid_pixels]).sum().float()
        total = valid_pixels.sum().float()
        accuracy = correct / total.clamp(min=1.0)
        return accuracy
