from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from YoloV26ImageProcessor import YoloV26ImageProcessor


@dataclass
class YoloV26DataConfig:
    data_root: str = "YoloV26/datasets/yolo"
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = torch.cuda.is_available()
    image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    )


class YoloV26DetectionDataset(Dataset):
    """
    YOLO-format detection dataset.

    Expected folder structure:
        data_root/
            images/
                train/
                val/
            labels/
                train/
                val/

    Label format:
        class_id cx cy w h

    Label values should be normalized between 0 and 1.

    Returned target format:
        [class_id, x1, y1, x2, y2]
        pixel coordinates after resizing to image_size.
    """

    def __init__(
        self,
        data_root,
        split,
        image_processor: YoloV26ImageProcessor,
        image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.data_root = Path(data_root)
        self.split = split

        self.image_dir = self.data_root / "images" / split
        self.label_dir = self.data_root / "labels" / split

        self.image_processor = image_processor
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        self.image_paths = self._find_images()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        targets = self._load_targets(image_path)

        if self.split == "train":
            image, targets = self.image_processor.process_train(
                image=image,
                targets=targets,
            )
        else:
            image, targets = self.image_processor.process_val(
                image=image,
                targets=targets,
            )

        return image, targets

    def _find_images(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")

        image_paths = [
            path for path in self.image_dir.rglob("*")
            if path.suffix.lower() in self.image_extensions
        ]

        image_paths = sorted(image_paths)

        if not image_paths:
            raise RuntimeError(f"No images found in: {self.image_dir}")

        return image_paths

    def _load_targets(self, image_path):
        label_path = self.label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        rows = []

        with open(label_path, "r", encoding="utf-8") as label_file:
            for line in label_file:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                class_id = float(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                rows.append([class_id, cx, cy, width, height])

        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)

        targets = torch.tensor(rows, dtype=torch.float32)
        targets = self._normalized_xywh_to_pixel_xyxy(targets)

        return targets

    def _normalized_xywh_to_pixel_xyxy(self, targets):
        image_size = self.image_processor.config.image_size

        class_ids = targets[:, 0:1]

        cx = targets[:, 1] * image_size
        cy = targets[:, 2] * image_size
        width = targets[:, 3] * image_size
        height = targets[:, 4] * image_size

        x1 = (cx - width * 0.5).clamp(min=0.0, max=float(image_size))
        y1 = (cy - height * 0.5).clamp(min=0.0, max=float(image_size))
        x2 = (cx + width * 0.5).clamp(min=0.0, max=float(image_size))
        y2 = (cy + height * 0.5).clamp(min=0.0, max=float(image_size))

        return torch.cat(
            [
                class_ids,
                x1[:, None],
                y1[:, None],
                x2[:, None],
                y2[:, None],
            ],
            dim=1,
        )


def yolo_v26_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]

    return images, targets


class YoloV26DataModule:
    def __init__(
        self,
        data_config: YoloV26DataConfig,
        image_processor: YoloV26ImageProcessor,
    ):
        self.data_config = data_config
        self.image_processor = image_processor

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        self.train_dataset = YoloV26DetectionDataset(
            data_root=self.data_config.data_root,
            split="train",
            image_processor=self.image_processor,
            image_extensions=self.data_config.image_extensions,
        )

        self.val_dataset = YoloV26DetectionDataset(
            data_root=self.data_config.data_root,
            split="val",
            image_processor=self.image_processor,
            image_extensions=self.data_config.image_extensions,
        )

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=yolo_v26_collate_fn,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            collate_fn=yolo_v26_collate_fn,
        )