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
from UNetMulticlassEndToEndModel import UNetMulticlassEndToEndModel
from MulticlassSegmentationDataset import MulticlassSegmentationDataset
from DataLoaderBuilderBlock import DataLoaderBuilderBlock
from TrainOneEpochBlock import TrainOneEpochBlock
from ValidateOneEpochBlock import ValidateOneEpochBlock
from CheckpointSaveBlock import CheckpointSaveBlock

class FullTrainingPipelineBlock:

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        image_size: int = IMAGE_SIZE,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        learning_rate: float = LEARNING_RATE,
        save_path: str = SAVE_PATH,
        color_to_class: Optional[Dict[Tuple[int, int, int], int]] = None,
        device: Optional[torch.device] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.color_to_class = color_to_class
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self):
        dataset = MulticlassSegmentationDataset(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            image_size=self.image_size,
            num_classes=self.num_classes,
            color_to_class=self.color_to_class,
            ignore_index=self.ignore_index,
        )

        dataloader_builder = DataLoaderBuilderBlock(batch_size=self.batch_size)
        train_loader, val_loader = dataloader_builder(dataset)

        model = UNetMulticlassEndToEndModel(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        trainer = TrainOneEpochBlock(device=self.device)
        validator = ValidateOneEpochBlock(device=self.device)
        checkpoint_saver = CheckpointSaveBlock(save_path=self.save_path)

        best_miou = 0.0

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")

            train_metrics = trainer(model=model, dataloader=train_loader, optimizer=optimizer)
            val_metrics = validator(model=model, dataloader=val_loader)

            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Acc:  {train_metrics['pixel_accuracy']:.4f}")
            print(f"Train mIoU: {train_metrics['mean_iou']:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f}")
            print(f"Val Acc:    {val_metrics['pixel_accuracy']:.4f}")
            print(f"Val mIoU:   {val_metrics['mean_iou']:.4f}")

            if val_metrics["mean_iou"] > best_miou:
                best_miou = val_metrics["mean_iou"]
                checkpoint_saver(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_miou=best_miou,
                )
                print(f"Best model saved with mIoU: {best_miou:.4f}")

        return model
