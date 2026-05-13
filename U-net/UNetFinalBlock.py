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
from InputBlock import InputBlock
from OutputBlock import OutputBlock
from UNetFeatureExtractorBlock import UNetFeatureExtractorBlock
from UNetDecoderBlock import UNetDecoderBlock

class UNetFinalBlock(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.input_block = InputBlock(in_channels=3)
        self.feature_extractor = UNetFeatureExtractorBlock()
        self.decoder = UNetDecoderBlock()
        self.output_block = OutputBlock(num_classes=num_classes)

    def forward(self, x):
        x = self.input_block(x)
        bottleneck, skip1, skip2, skip3, skip4 = self.feature_extractor(x)
        x = self.decoder(bottleneck, skip1, skip2, skip3, skip4)
        logits = self.output_block(x)
        return logits
