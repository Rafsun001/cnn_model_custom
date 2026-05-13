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
from UNetMulticlassArchitectureModel import UNetMulticlassArchitectureModel
from UNetLossComputationBlock import UNetLossComputationBlock
from UNetPostProcessBlock import UNetPostProcessBlock

class UNetMulticlassEndToEndModel(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.model = UNetMulticlassArchitectureModel(num_classes=num_classes)
        self.loss_computation = UNetLossComputationBlock(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.postprocess = UNetPostProcessBlock()

    def forward(self, images, masks: Optional[torch.Tensor] = None):
        logits = self.model(images)

        if masks is not None:
            return self.loss_computation(logits, masks)

        if self.training:
            raise ValueError("When training, masks must be provided.")

        return self.postprocess(logits)
