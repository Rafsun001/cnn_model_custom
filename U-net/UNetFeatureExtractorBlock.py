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
from EncoderBlock1 import EncoderBlock1
from DownsampleBlock1 import DownsampleBlock1
from EncoderBlock2 import EncoderBlock2
from DownsampleBlock2 import DownsampleBlock2
from EncoderBlock3 import EncoderBlock3
from DownsampleBlock3 import DownsampleBlock3
from EncoderBlock4 import EncoderBlock4
from DownsampleBlock4 import DownsampleBlock4
from BottleneckBlock import BottleneckBlock

class UNetFeatureExtractorBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder1 = EncoderBlock1()
        self.downsample1 = DownsampleBlock1()

        self.encoder2 = EncoderBlock2()
        self.downsample2 = DownsampleBlock2()

        self.encoder3 = EncoderBlock3()
        self.downsample3 = DownsampleBlock3()

        self.encoder4 = EncoderBlock4()
        self.downsample4 = DownsampleBlock4()

        self.bottleneck = BottleneckBlock()

    def forward(self, x):
        skip1 = self.encoder1(x)
        x = self.downsample1(skip1)

        skip2 = self.encoder2(x)
        x = self.downsample2(skip2)

        skip3 = self.encoder3(x)
        x = self.downsample3(skip3)

        skip4 = self.encoder4(x)
        x = self.downsample4(skip4)

        bottleneck = self.bottleneck(x)

        return bottleneck, skip1, skip2, skip3, skip4
