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

class DataLoaderBuilderBlock:

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        val_fraction: float = 0.2,
        num_workers: int = NUM_WORKERS,
    ):
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers

    def __call__(self, dataset: Dataset):
        if len(dataset) < 2:
            raise ValueError("Dataset must contain at least 2 image-mask pairs")

        val_size = max(1, int(self.val_fraction * len(dataset)))
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader
