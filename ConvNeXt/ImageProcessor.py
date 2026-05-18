from dataclasses import dataclass
from typing import Tuple

from torchvision import transforms


@dataclass
class ImageProcessingConfig:
    image_size: int = 64
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    hflip_prob: float = 0.5
    use_strong_aug: bool = False
    random_erasing_prob: float = 0.0


class ImageProcessor:
    def __init__(self, config: ImageProcessingConfig):
        self.config = config

    def get_train_transform(self):
        transform_steps = [
            transforms.RandomCrop(self.config.image_size, padding=4),
            transforms.RandomHorizontalFlip(p=self.config.hflip_prob),
        ]

        if self.config.use_strong_aug:
            transform_steps.append(transforms.RandAugment())

        transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])

        if self.config.random_erasing_prob > 0.0:
            transform_steps.append(
                transforms.RandomErasing(p=self.config.random_erasing_prob),
            )

        return transforms.Compose(transform_steps)

    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])
