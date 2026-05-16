from dataclasses import dataclass
from typing import Tuple

from torchvision import transforms


@dataclass
class ImageProcessingConfig:
    
    image_size: int = 224
    val_resize_size: int = 256
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    hflip_prob: float = 0.5


class ImageProcessor:
    def __init__(self, config: ImageProcessingConfig):
        self.config = config

    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.config.image_size),
            transforms.RandomHorizontalFlip(p=self.config.hflip_prob),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])

    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.config.val_resize_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.config.mean, self.config.std),
        ])
