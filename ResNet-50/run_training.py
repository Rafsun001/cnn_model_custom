import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from DataLoader_builder import DataLoaderConfig, TinyImageNetDataModule
from ImageProcessor import ImageProcessor, ImageProcessingConfig
from model import ResNet50
from train_evaluation import TrainingConfig, Trainer, set_global_seed


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    print("=" * 70)
    print("Exact ResNet-50 Training on TinyImageNet-200")
    print("=" * 70)

    seed = 42
    set_global_seed(seed)

    print("\n[1/5] Creating ImageNet-style image processor...")
    image_config = ImageProcessingConfig(
        image_size=224,
        val_resize_size=256,
    )
    image_processor = ImageProcessor(image_config)
    print("  OK image processor created")

    print("\n[2/5] Loading TinyImageNet-200 from Hugging Face...")
    data_config = DataLoaderConfig(
        batch_size=64,
        num_workers=0,
        seed=seed,
        pin_memory=True,
        persistent_workers=False,
    )

    data_module = TinyImageNetDataModule(
        data_config=data_config,
        image_processor=image_processor,
    )
    data_module.setup()

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    print(f"  OK train batches: {len(train_loader)}")
    print(f"  OK val batches: {len(val_loader)}")

    print("\n[3/5] Creating exact ResNet-50 model...")
    model = ResNet50(num_classes=200)

    print(f"  OK model created")
    print(f"  Parameters: {count_parameters(model):,}")

    print("\n[4/5] Configuring training...")
    training_config = TrainingConfig(
        epochs=100,
        learning_rate=0.025,
        momentum=0.9,
        weight_decay=1e-4,
        label_smoothing=0.0,
        warmup_epochs=5,
        min_lr=1e-5,
        max_grad_norm=0.0,
        seed=seed,
        use_amp=True,
        deterministic=False,
        checkpoint_dir=str(current_dir / "checkpoints_resnet50_exact"),
        resume_from_checkpoint=None,
        metrics_filename="metrics.csv",
    )

    print(f"  Epochs: {training_config.epochs}")
    print(f"  Batch size: {data_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Device: {training_config.device}")

    print("\n[5/5] Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )
    trainer.train()

    print("\n" + "=" * 70)
    print("Training completed.")
    print(f"Best model: {training_config.checkpoint_dir}/best_model.pth")
    print(f"Latest checkpoint: {training_config.checkpoint_dir}/latest_checkpoint.pth")
    print(f"Metrics: {training_config.checkpoint_dir}/{training_config.metrics_filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()
