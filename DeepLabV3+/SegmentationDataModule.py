from torch.utils.data import DataLoader
from .SegmentationDataset import SegmentationDataset
from .SegmentationTrainTransform import SegmentationTrainTransform
from .SegmentationValTransform import SegmentationValTransform


class SegmentationDataModule:
    def __init__(self, config):
        self.config = config

        self.train_transform = SegmentationTrainTransform(
            image_size=config.image_size,
            ignore_index=config.ignore_index,
        )

        self.val_transform = SegmentationValTransform(
            image_size=config.image_size,
            ignore_index=config.ignore_index,
        )

    def train_dataset(self):
        dataset = SegmentationDataset(
            images_dir=self.config.train_images_dir,
            masks_dir=self.config.train_masks_dir,
            transform=self.train_transform,
        )

        return dataset

    def val_dataset(self):
        dataset = SegmentationDataset(
            images_dir=self.config.val_images_dir,
            masks_dir=self.config.val_masks_dir,
            transform=self.val_transform,
        )

        return dataset

    def train_loader(self):
        loader = DataLoader(
            dataset=self.train_dataset(),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        return loader

    def val_loader(self):
        loader = DataLoader(
            dataset=self.val_dataset(),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return loader
