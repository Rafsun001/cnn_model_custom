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

class CheckpointSaveBlock:

    def __init__(self, save_path: str = SAVE_PATH):
        self.save_path = save_path

    def __call__(self, model, optimizer, epoch: int, best_miou: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "num_classes": model.num_classes,
                "ignore_index": model.ignore_index,
            },
            self.save_path,
        )
