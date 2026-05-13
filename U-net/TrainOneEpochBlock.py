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

class TrainOneEpochBlock:

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, model, dataloader, optimizer):
        model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        total_miou = 0.0

        progress_bar = tqdm(dataloader, desc="Training")

        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = model(images, masks)
            loss = outputs["loss_total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(outputs["loss_total"].detach())
            total_accuracy += float(outputs["pixel_accuracy"].detach())
            total_miou += float(outputs["mean_iou"].detach())

            progress_bar.set_postfix(
                loss=float(outputs["loss_total"].detach()),
                acc=float(outputs["pixel_accuracy"].detach()),
                miou=float(outputs["mean_iou"].detach()),
            )

        number_of_batches = max(1, len(dataloader))

        return {
            "loss": total_loss / number_of_batches,
            "pixel_accuracy": total_accuracy / number_of_batches,
            "mean_iou": total_miou / number_of_batches,
        }
