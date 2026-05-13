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
from ClassIndexToRGBMaskBlock import ClassIndexToRGBMaskBlock

class SingleImagePredictionBlock:

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        class_to_color: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ):
        self.image_size = image_size
        self.class_to_color = class_to_color
        self.color_converter = None
        if class_to_color is not None:
            self.color_converter = ClassIndexToRGBMaskBlock(class_to_color=class_to_color)

    def load_image(self, image_path: str, device: torch.device):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = (image_np - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        image_np = image_np.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(device)

        return image_tensor, original_size

    def __call__(
        self,
        model,
        image_path: str,
        output_mask_path: str,
        device: torch.device,
        output_color_path: Optional[str] = None,
    ):
        model.eval()

        image_tensor, original_size = self.load_image(image_path=image_path, device=device)

        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_mask = outputs["predicted_masks"].squeeze(0).cpu().numpy().astype(np.uint8)

        predicted_mask_image = Image.fromarray(predicted_mask)
        predicted_mask_image = predicted_mask_image.resize(original_size, Image.NEAREST)
        predicted_mask_image.save(output_mask_path)

        if output_color_path is not None and self.color_converter is not None:
            resized_class_mask = np.array(predicted_mask_image, dtype=np.uint8)
            color_mask = self.color_converter(resized_class_mask)
            color_mask_image = Image.fromarray(color_mask)
            color_mask_image.save(output_color_path)

        return output_mask_path
