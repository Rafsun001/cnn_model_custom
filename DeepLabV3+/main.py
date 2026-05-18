from .DeepLabV3PlusResNet50 import DeepLabV3PlusResNet50
from .DeepLabV3PlusTrainer import DeepLabV3PlusTrainer
from .SegmentationDataModule import SegmentationDataModule
from .TrainingConfig import TrainingConfig
from .set_global_seed import set_global_seed


def main():
    config = TrainingConfig(
        train_images_dir="data/train/images",
        train_masks_dir="data/train/masks",
        val_images_dir="data/val/images",
        val_masks_dir="data/val/masks",

        num_classes=21,
        ignore_index=255,

        image_size=224,
        batch_size=4,
        num_workers=2,

        epochs=100,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4,

        checkpoint_dir="checkpoints_deeplabv3plus",
        use_amp=True,

        freeze_batch_norm=False,
    )

    set_global_seed(config.seed)

    data_module = SegmentationDataModule(config)

    train_loader = data_module.train_loader()
    val_loader = data_module.val_loader()

    model = DeepLabV3PlusResNet50(
        num_classes=config.num_classes,
    )

    trainer = DeepLabV3PlusTrainer(
        model=model,
        config=config,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    main()
