import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageOps


class SegmentationTrainTransform:
    def __init__(self, image_size, ignore_index):
        self.image_size = image_size
        self.ignore_index = ignore_index

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def random_scale(self, image, mask):
        scale = random.uniform(0.5, 2.0)

        width, height = image.size

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize((new_width, new_height), Image.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.NEAREST)

        return image, mask

    def pad_if_needed(self, image, mask):
        width, height = image.size

        pad_width = max(self.image_size - width, 0)
        pad_height = max(self.image_size - height, 0)

        if pad_width > 0 or pad_height > 0:
            image = ImageOps.expand(
                image,
                border=(0, 0, pad_width, pad_height),
                fill=0,
            )

            mask = ImageOps.expand(
                mask,
                border=(0, 0, pad_width, pad_height),
                fill=self.ignore_index,
            )

        return image, mask

    def random_crop(self, image, mask):
        width, height = image.size

        left = random.randint(0, width - self.image_size)
        top = random.randint(0, height - self.image_size)

        right = left + self.image_size
        bottom = top + self.image_size

        image = image.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))

        return image, mask

    def random_horizontal_flip(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return image, mask

    def to_tensor_and_normalize(self, image, mask):
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_np = image_np.transpose(2, 0, 1)

        mask_np = np.array(mask)

        if mask_np.ndim == 3:
            raise ValueError(
                "Mask has 3 channels. This code expects class-index masks "
                "with shape [H, W], where each pixel value is a class id."
            )

        image_tensor = torch.from_numpy(image_np.copy()).float()
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64)).long()

        return image_tensor, mask_tensor

    def __call__(self, image, mask):
        image, mask = self.random_scale(image, mask)
        image, mask = self.pad_if_needed(image, mask)
        image, mask = self.random_crop(image, mask)
        image, mask = self.random_horizontal_flip(image, mask)

        image_tensor, mask_tensor = self.to_tensor_and_normalize(image, mask)

        return image_tensor, mask_tensor
