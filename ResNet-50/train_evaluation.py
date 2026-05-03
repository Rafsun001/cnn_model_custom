import csv
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.amp import GradScaler, autocast
    USE_TORCH_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    USE_TORCH_AMP = False

from DataLoader_builder import DataLoaderConfig, TinyImageNetDataModule
from ImageProcessor import ImageProcessor, ImageProcessingConfig
from model import ResNet50TinyImageNet200


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    use_ema: bool = True
    ema_decay: float = 0.9999
    use_amp: bool = True
    warmup_epochs: int = 5
    min_lr: float = 1e-5
    use_cosine: bool = True
    max_grad_norm: float = 0.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    metrics_filename: str = "metrics.csv"
    save_ema_as_model: bool = True


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class Evaluator:
    def __init__(self, val_loader, loss_function, device: str):
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.device = device

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            loss = self.loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predicted_labels = outputs.argmax(dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_images += labels.size(0)

        average_loss = total_loss / total_images
        accuracy = 100.0 * total_correct / total_images

        return average_loss, accuracy


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: TrainingConfig,
    ):
        set_global_seed(config.seed)
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.config = config

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        self.base_lr = config.learning_rate
        self.warmup_epochs = config.warmup_epochs
        self.use_amp = config.use_amp and config.device.startswith("cuda") and torch.cuda.is_available()
        if USE_TORCH_AMP:
            self.scaler = GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = GradScaler(enabled=self.use_amp)
        self.mixup_alpha = config.mixup_alpha
        self.use_ema = config.use_ema
        self.ema_decay = config.ema_decay
        self.ema_state = None
        self.max_grad_norm = config.max_grad_norm
        self.scheduler = None
        if config.use_cosine and config.epochs > config.warmup_epochs:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs - config.warmup_epochs,
                eta_min=config.min_lr,
            )

        self.evaluator = Evaluator(
            val_loader=val_loader,
            loss_function=self.loss_function,
            device=config.device,
        )

        self.best_val_accuracy = -1.0
        self.previous_val_accuracy = -1.0
        self.saved_checkpoints = []
        self.start_epoch = 1
        self.metrics_path = Path(config.checkpoint_dir) / config.metrics_filename
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        if self.use_ema:
            self._init_ema()

        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def train(self):
        print(f"\n{'='*70}")
        print("Starting Training with Checkpoint Monitoring")
        print(f"{'='*70}\n")

        if self.start_epoch == 1:
            self._init_metrics_file()

        if self.start_epoch > self.config.epochs:
            print(
                f"Checkpoint is already past the requested {self.config.epochs} epochs. "
                "Nothing to train."
            )
            return

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self._set_warmup_lr(epoch)
            train_loss, train_accuracy = self.train_one_epoch(epoch)

            if self.use_ema:
                ema_backup = self._swap_ema_weights()
                val_loss, val_accuracy = self.evaluator.evaluate(self.model)
                self._restore_weights(ema_backup)
            else:
                val_loss, val_accuracy = self.evaluator.evaluate(self.model)


            epoch_info = (
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:6.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:6.2f}% | "
                f"LR: {self._get_current_lr():.6f}"
            )
            print(epoch_info)
            self._write_metrics_row(epoch, train_loss, train_accuracy, val_loss, val_accuracy)


            if val_accuracy > self.previous_val_accuracy:
                improvement = val_accuracy - self.previous_val_accuracy
                self.save_checkpoint(
                    f"epoch_{epoch:03d}.pth",
                    epoch=epoch,
                    val_accuracy=val_accuracy,
                    is_epoch_improvement=True,
                    improvement=improvement
                )
                print(f"  OK Saved epoch_{epoch:03d}.pth (improved +{improvement:.2f}%)")


            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_checkpoint(
                    "best_model.pth",
                    epoch=epoch,
                    val_accuracy=val_accuracy,
                    is_best=True
                )
                print(f"  BEST Saved best_model.pth (new best: {val_accuracy:.2f}%)")


            self.previous_val_accuracy = val_accuracy

            if self.scheduler is not None and epoch > self.warmup_epochs:
                self.scheduler.step()

            self.save_checkpoint(
                "latest_checkpoint.pth",
                epoch=epoch,
                val_accuracy=val_accuracy,
                track=False,
                save_ema_as_model=False,
            )


        self._print_checkpoint_summary()

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in self.train_loader:
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            labels_for_accuracy = labels
            if self.mixup_alpha > 0:
                images, labels_a, labels_b, mixup_lam = self._mixup_batch(images, labels)
            else:
                labels_a = labels
                labels_b = labels
                mixup_lam = 1.0
            self.optimizer.zero_grad()

            if USE_TORCH_AMP:
                autocast_context = autocast("cuda", enabled=self.use_amp)
            else:
                autocast_context = autocast(enabled=self.use_amp)

            with autocast_context:
                outputs = self.model(images)
                loss = (
                    mixup_lam * self.loss_function(outputs, labels_a)
                    + (1.0 - mixup_lam) * self.loss_function(outputs, labels_b)
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.use_ema:
                self._update_ema()

            total_loss += loss.item() * images.size(0)

            predicted_labels = outputs.argmax(dim=1)
            total_correct += (predicted_labels == labels_for_accuracy).sum().item()
            total_images += labels.size(0)

        average_loss = total_loss / total_images
        accuracy = 100.0 * total_correct / total_images

        return average_loss, accuracy

    def _mixup_batch(self, images, labels):
        if self.mixup_alpha <= 0:
            return images, labels, labels, 1.0

        mixup_dist = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = mixup_dist.sample().item()

        batch_size = images.size(0)
        perm = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[perm]
        labels_a = labels
        labels_b = labels[perm]

        return mixed_images, labels_a, labels_b, lam

    def _init_ema(self):
        self.ema_state = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _update_ema(self):
        if self.ema_state is None:
            return

        with torch.no_grad():
            for name, tensor in self.model.state_dict().items():
                ema_tensor = self.ema_state[name]
                if torch.is_floating_point(tensor):
                    ema_tensor.mul_(self.ema_decay).add_(tensor, alpha=1.0 - self.ema_decay)
                else:
                    ema_tensor.copy_(tensor)

    def _swap_ema_weights(self):
        if self.ema_state is None:
            return None

        backup_state = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.ema_state, strict=True)
        return backup_state

    def _restore_weights(self, backup_state):
        if backup_state is None:
            return

        self.model.load_state_dict(backup_state, strict=True)

    def _set_warmup_lr(self, epoch):
        if self.warmup_epochs <= 0:
            return

        if epoch <= self.warmup_epochs:
            warmup_lr = self.base_lr * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr
        elif epoch == self.warmup_epochs + 1:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.base_lr

    def _get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _clone_state_dict(self, state_dict):
        return {
            name: tensor.detach().clone()
            for name, tensor in state_dict.items()
        }

    def _checkpoint_model_state(self, save_ema_as_model: bool):
        if save_ema_as_model and self.use_ema and self.ema_state is not None:
            return self._clone_state_dict(self.ema_state), "ema"

        return self._clone_state_dict(self.model.state_dict()), "raw"

    def save_checkpoint(
        self,
        filename,
        epoch=None,
        val_accuracy=None,
        is_best=False,
        is_epoch_improvement=False,
        improvement=0.0,
        track=True,
        save_ema_as_model=None,
    ):
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        if save_ema_as_model is None:
            save_ema_as_model = self.config.save_ema_as_model

        model_state, model_state_kind = self._checkpoint_model_state(save_ema_as_model)

        checkpoint = {
            "model_state_dict": model_state,
            "model_state_kind": model_state_kind,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_val_accuracy": self.best_val_accuracy,
            "previous_val_accuracy": self.previous_val_accuracy,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "training_config": asdict(self.config),
            "saved_checkpoints": self.saved_checkpoints,
        }

        if self.use_ema and self.ema_state is not None:
            checkpoint["ema_state_dict"] = self._clone_state_dict(self.ema_state)

        torch.save(checkpoint, checkpoint_path)


        checkpoint_info = {
            "filename": filename,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "is_best": is_best,
            "is_epoch_improvement": is_epoch_improvement,
            "improvement": improvement,
            "path": str(checkpoint_path),
            "model_state_kind": model_state_kind,
        }
        if track:
            self.saved_checkpoints.append(checkpoint_info)

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("scaler_state_dict")
        if self.use_amp and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        ema_state = checkpoint.get("ema_state_dict")
        if self.use_ema and ema_state is not None:
            self.ema_state = {
                name: tensor.to(self.config.device)
                for name, tensor in ema_state.items()
            }

        self.best_val_accuracy = checkpoint.get("best_val_accuracy", self.best_val_accuracy)
        self.previous_val_accuracy = checkpoint.get("previous_val_accuracy", self.previous_val_accuracy)
        self.saved_checkpoints = checkpoint.get("saved_checkpoints", self.saved_checkpoints)
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

        print(f"Resumed from {checkpoint_path} at epoch {self.start_epoch}.")

    def _init_metrics_file(self):
        with open(self.metrics_path, "w", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
            ])

    def _write_metrics_row(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
        with open(self.metrics_path, "a", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_accuracy:.4f}",
                f"{val_loss:.6f}",
                f"{val_accuracy:.4f}",
                f"{self._get_current_lr():.8f}",
            ])

    def _print_checkpoint_summary(self):
        print(f"\n{'='*70}")
        print("CHECKPOINT SUMMARY")
        print(f"{'='*70}\n")

        summary_lines = [
            "=" * 70,
            "TRAINING CHECKPOINT SUMMARY",
            "=" * 70,
            "",
            f"Total checkpoints saved: {len(self.saved_checkpoints)}",
            "",
            "Saved Checkpoints:",
            "-" * 70,
        ]

        for i, ckpt in enumerate(self.saved_checkpoints, 1):
            flags = []
            if ckpt["is_best"]:
                flags.append("BEST")
            if ckpt["is_epoch_improvement"]:
                flags.append(f"IMPROVED +{ckpt['improvement']:.2f}%")

            flag_str = " | ".join(flags) if flags else ""
            line = (
                f"{i:2d}. Epoch {ckpt['epoch']:3d} - "
                f"Acc: {ckpt['val_accuracy']:6.2f}% - "
                f"{ckpt['filename']:20s} - "
                f"Weights: {ckpt.get('model_state_kind', 'raw')}"
            )
            if flag_str:
                line += f" [{flag_str}]"

            print(line)
            summary_lines.append(line)

        print("-" * 70)
        print(f"\nBest Model Summary:")
        print(f"  - File: best_model.pth")
        print(f"  - Best Accuracy: {self.best_val_accuracy:.2f}%")
        print(f"  - Location: {Path(self.config.checkpoint_dir) / 'best_model.pth'}")
        print(f"  - Metrics: {self.metrics_path}")

        summary_lines.extend([
            "-" * 70,
            "",
            "Best Model Summary:",
            f"  - File: best_model.pth",
            f"  - Best Accuracy: {self.best_val_accuracy:.2f}%",
            f"  - Location: {Path(self.config.checkpoint_dir) / 'best_model.pth'}",
            f"  - Metrics: {self.metrics_path}",
            "",
            "=" * 70,
        ])


        summary_path = Path(self.config.checkpoint_dir) / "training_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*70}\n")


def main():
    seed = 42
    set_global_seed(seed)

    image_config = ImageProcessingConfig(image_size=64)
    image_processor = ImageProcessor(image_config)

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

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    model = ResNet50TinyImageNet200(num_classes=200)

    training_config = TrainingConfig(
        epochs=50,
        learning_rate=0.05,
        momentum=0.9,
        weight_decay=1e-4,
        seed=seed,
        checkpoint_dir="checkpoints_resnet50_tinyimagenet",
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
