from dataclasses import dataclass


@dataclass
class TrainingConfig:
    train_images_dir: str = "data/train/images"
    train_masks_dir: str = "data/train/masks"

    val_images_dir: str = "data/val/images"
    val_masks_dir: str = "data/val/masks"

    num_classes: int = 21
    ignore_index: int = 255

    image_size: int = 224

    batch_size: int = 4
    num_workers: int = 2

    epochs: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    checkpoint_dir: str = "checkpoints_deeplabv3plus"
    best_checkpoint_name: str = "best_model.pth"
    latest_checkpoint_name: str = "latest_checkpoint.pth"

    seed: int = 42
    use_amp: bool = True


    freeze_batch_norm: bool = False
