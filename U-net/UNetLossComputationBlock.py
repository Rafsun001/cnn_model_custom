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
from MaskValidationBlock import MaskValidationBlock
from CrossEntropySegmentationLossBlock import CrossEntropySegmentationLossBlock
from PixelAccuracyBlock import PixelAccuracyBlock
from MeanIoUBlock import MeanIoUBlock

class UNetLossComputationBlock(nn.Module):

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.mask_validation = MaskValidationBlock(num_classes=num_classes, ignore_index=ignore_index)
        self.cross_entropy_loss = CrossEntropySegmentationLossBlock(ignore_index=ignore_index)
        self.pixel_accuracy = PixelAccuracyBlock(ignore_index=ignore_index)
        self.mean_iou = MeanIoUBlock(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, masks):
        masks = self.mask_validation(masks)

        loss_ce = self.cross_entropy_loss(logits, masks)
        accuracy = self.pixel_accuracy(logits, masks)
        miou = self.mean_iou(logits, masks)

        return {
            "loss_total": loss_ce,
            "loss_ce": loss_ce,
            "pixel_accuracy": accuracy.detach(),
            "mean_iou": miou.detach(),
        }
