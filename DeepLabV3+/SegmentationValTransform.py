import torch
import numpy as np
from PIL import Image
from PIL import ImageOps


class SegmentationValTransform:
    def __init__(self, image_size, ignore_index):
        self.image_size = image_size
        self.ignore_index = ignore_index

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def resize_preserve_aspect_ratio_and_pad(self, image, mask):
        width, height = image.size

        scale = min(
            self.image_size / width,
            self.image_size / height,
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        image = image.resize((new_width, new_height), Image.BILINEAR)
        mask = mask.resize((new_width, new_height), Image.NEAREST)

        pad_width = self.image_size - new_width
        pad_height = self.image_size - new_height

        left = pad_width // 2
        top = pad_height // 2
        right = pad_width - left
        bottom = pad_height - top

        image = ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=0,
        )

        mask = ImageOps.expand(
            mask,
            border=(left, top, right, bottom),
            fill=self.ignore_index,
        )

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
        image, mask = self.resize_preserve_aspect_ratio_and_pad(image, mask)
        image_tensor, mask_tensor = self.to_tensor_and_normalize(image, mask)

        return image_tensor, mask_tensor
