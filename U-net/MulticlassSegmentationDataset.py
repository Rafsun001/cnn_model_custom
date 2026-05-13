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
from RGBMaskToClassIndexBlock import RGBMaskToClassIndexBlock

class MulticlassSegmentationDataset(Dataset):

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int = IMAGE_SIZE,
        num_classes: int = NUM_CLASSES,
        color_to_class: Optional[Dict[Tuple[int, int, int], int]] = None,
        ignore_index: int = IGNORE_INDEX,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.color_to_class = color_to_class
        self.ignore_index = ignore_index

        self.rgb_to_class = None
        if self.color_to_class is not None:
            self.rgb_to_class = RGBMaskToClassIndexBlock(
                color_to_class=self.color_to_class,
                ignore_index=self.ignore_index,
            )

        self.image_paths = sorted([
            path for path in self.image_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def find_mask_path(self, image_path: Path):
        exact_mask_path = self.mask_dir / image_path.name

        if exact_mask_path.exists():
            return exact_mask_path

        for extension in [".png", ".jpg", ".jpeg", ".bmp"]:
            possible_mask_path = self.mask_dir / f"{image_path.stem}{extension}"
            if possible_mask_path.exists():
                return possible_mask_path

        raise FileNotFoundError(f"No mask found for image: {image_path.name}")

    def load_image(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = (image_np - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        image_np = image_np.transpose(2, 0, 1)

        return torch.from_numpy(image_np).float()

    def load_mask(self, mask_path: Path):
        if self.color_to_class is None:
            mask = Image.open(mask_path)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = np.array(mask)

            if mask_np.ndim == 3:
                raise ValueError(
                    f"Mask {mask_path.name} appears to be RGB. "
                    f"Provide color_to_class mapping to convert colors into class IDs."
                )

            mask_np = mask_np.astype(np.int64)
        else:
            mask = Image.open(mask_path).convert("RGB")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_rgb = np.array(mask, dtype=np.uint8)
            mask_np = self.rgb_to_class(mask_rgb)

        valid_pixels = mask_np != self.ignore_index

        if valid_pixels.any():
            min_value = mask_np[valid_pixels].min()
            max_value = mask_np[valid_pixels].max()

            if min_value < 0 or max_value >= self.num_classes:
                raise ValueError(
                    f"Invalid class value found in {mask_path.name}. "
                    f"Allowed class IDs are 0 to {self.num_classes - 1}, "
                    f"or ignore_index={self.ignore_index}. "
                    f"Found min={int(min_value)}, max={int(max_value)}."
                )

        return torch.from_numpy(mask_np).long()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.find_mask_path(image_path)

        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)

        return image, mask
