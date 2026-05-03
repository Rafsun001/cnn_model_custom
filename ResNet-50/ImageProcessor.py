from dataclasses import dataclass
from typing import Tuple

from torchvision import transforms


@dataclass
class ImageProcessingConfig:
    image_size: int = 64
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_strong_aug: bool = True
    random_resized_crop_scale: Tuple[float, float] = (0.7, 1.0)
    random_resized_crop_ratio: Tuple[float, float] = (0.75, 1.33)
    hflip_prob: float = 0.5
    color_jitter: float = 0.4
    color_jitter_hue: float = 0.1
    color_jitter_prob: float = 0.8
    random_grayscale_prob: float = 0.1
    random_erasing_prob: float = 0.25


class ImageProcessor:
    def __init__(self, config: ImageProcessingConfig):
        self.config = config

    def get_train_transform(self):
        if self.config.use_strong_aug:
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    self.config.image_size,
                    scale=self.config.random_resized_crop_scale,
                    ratio=self.config.random_resized_crop_ratio,
                ),
                transforms.RandomHorizontalFlip(p=self.config.hflip_prob),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        self.config.color_jitter,
                        self.config.color_jitter,
                        self.config.color_jitter,
                        self.config.color_jitter_hue,
                    )
                ], p=self.config.color_jitter_prob),
                transforms.RandomGrayscale(p=self.config.random_grayscale_prob),
                transforms.ToTensor(),
                transforms.Normalize(self.config.mean, self.config.std),
                transforms.RandomErasing(p=self.config.random_erasing_prob),
            ])

        return transforms.Compose([
            transforms.RandomCrop(self.config.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])

    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])
