import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataset_classes import TinyImageNetTrainDataset, TinyImageNetValDataset
from ImageProcessor import ImageProcessor


@dataclass
class DataLoaderConfig:
    batch_size: int = 128
    num_workers: int = 0
    shuffle_train: bool = True
    seed: int = 42
    pin_memory: bool = torch.cuda.is_available()


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TinyImageNetDataModule:
    def __init__(
        self,
        data_config: DataLoaderConfig,
        image_processor: ImageProcessor,
    ):
        self.data_config = data_config
        self.image_processor = image_processor

        self.train_dataset = None
        self.val_dataset = None
        self.class_to_idx = None

    def setup(self):
        train_transform = self.image_processor.get_train_transform()
        val_transform = self.image_processor.get_val_transform()


        self.train_dataset = TinyImageNetTrainDataset(
            transform=train_transform,
        )


        self.class_to_idx = self.train_dataset.class_to_idx

        self.val_dataset = TinyImageNetValDataset(
            transform=val_transform,
        )

    def get_train_loader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before get_train_loader().")

        generator = torch.Generator()
        generator.manual_seed(self.data_config.seed)

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=self.data_config.shuffle_train,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    def get_val_loader(self):
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before get_val_loader().")

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            worker_init_fn=_seed_worker,
        )
