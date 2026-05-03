

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from DataLoader_builder import DataLoaderConfig, TinyImageNetDataModule
from ImageProcessor import ImageProcessor, ImageProcessingConfig
from model import ResNet50TinyImageNet200
from train_evaluation import TrainingConfig, Trainer, set_global_seed


def main():
    print("=" * 70)
    print("ResNet-50 Training on TinyImageNet200 (from Hugging Face)")
    print("=" * 70)

    seed = 42
    set_global_seed(seed)

    print("\n[1/5] Creating image processor...")
    image_config = ImageProcessingConfig(
        image_size=64,
        use_strong_aug=True,
        color_jitter=0.4,
        random_erasing_prob=0.25,
    )
    image_processor = ImageProcessor(image_config)
    print("  OK Image processor created")

    print("\n[2/5] Loading TinyImageNet200 from Hugging Face...")
    print("  This may take a few minutes on first run.")
    data_config = DataLoaderConfig(
        batch_size=128,
        num_workers=0,
        seed=seed,
    )
    data_module = TinyImageNetDataModule(
        data_config=data_config,
        image_processor=image_processor,
    )
    data_module.setup()
    print("  OK Datasets loaded and ready")

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")

    print("\n[3/5] Creating ResNet-50 model (200 classes)...")
    model = ResNet50TinyImageNet200(num_classes=200)
    print("  OK ResNet-50 model created")

    print("\n[4/5] Configuring training...")
    training_config = TrainingConfig(
        epochs=100,
        learning_rate=0.05,
        momentum=0.9,
        weight_decay=1e-4,
        label_smoothing=0.1,
        mixup_alpha=0.2,
        use_ema=True,
        ema_decay=0.9999,
        use_amp=True,
        warmup_epochs=5,
        min_lr=1e-5,
        use_cosine=True,
        max_grad_norm=0.0,
        seed=seed,
        checkpoint_dir=str(current_dir / "checkpoints_resnet50_tinyimagenet"),
        resume_from_checkpoint=None,
        metrics_filename="metrics.csv",
        save_ema_as_model=True,
    )
    print(f"  - Epochs: {training_config.epochs}")
    print(f"  - Batch size: {data_config.batch_size}")
    print(f"  - Learning rate: {training_config.learning_rate}")
    print(f"  - Device: {training_config.device}")

    print("\n[5/5] Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    print("\nTraining Progress:")
    print("-" * 70)
    trainer.train()
    print("-" * 70)

    print("\n" + "=" * 70)
    print("OK Training completed!")
    print(f"OK Best model saved to: {training_config.checkpoint_dir}/best_model.pth")
    print(f"OK Latest resume checkpoint: {training_config.checkpoint_dir}/latest_checkpoint.pth")
    print(f"OK Metrics saved to: {training_config.checkpoint_dir}/{training_config.metrics_filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()
