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

class OutputBlock(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.conv(x)
