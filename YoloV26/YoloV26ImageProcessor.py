from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


@dataclass
class YoloV26ImageProcessingConfig:
    image_size: int = 640

    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    use_train_aug: bool = True

    hflip_prob: float = 0.5

    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


class YoloV26ImageProcessor:
    """
    Simple training/validation processor.

    This keeps your custom project clean and stable.

    It supports:
        - RGB conversion
        - resize to square image_size
        - color jitter
        - horizontal flip with box update
        - tensor conversion
        - normalization
    """

    def __init__(self, config: YoloV26ImageProcessingConfig):
        self.config = config

        self.color_jitter = transforms.ColorJitter(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
        )

    def process_train(self, image, targets):
        image = self._to_rgb(image)
        image = image.resize(
            (self.config.image_size, self.config.image_size),
            Image.BILINEAR,
        )

        if self.config.use_train_aug:
            image = self.color_jitter(image)

            if torch.rand(1).item() < self.config.hflip_prob:
                image = F.hflip(image)
                targets = self._hflip_targets(targets)

        image = F.to_tensor(image)
        image = F.normalize(
            image,
            mean=self.config.mean,
            std=self.config.std,
        )

        return image, targets

    def process_val(self, image, targets):
        image = self._to_rgb(image)
        image = image.resize(
            (self.config.image_size, self.config.image_size),
            Image.BILINEAR,
        )

        image = F.to_tensor(image)
        image = F.normalize(
            image,
            mean=self.config.mean,
            std=self.config.std,
        )

        return image, targets

    def _hflip_targets(self, targets):
        if targets.numel() == 0:
            return targets

        targets = targets.clone()

        x1 = targets[:, 1].clone()
        x2 = targets[:, 3].clone()

        targets[:, 1] = self.config.image_size - x2
        targets[:, 3] = self.config.image_size - x1

        return targets

    def _to_rgb(self, image):
        if image.mode != "RGB":
            return image.convert("RGB")

        return image