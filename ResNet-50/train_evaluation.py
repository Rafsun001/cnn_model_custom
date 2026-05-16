import csv
import math
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


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 0.025
    momentum: float = 0.9
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    warmup_epochs: int = 5
    min_lr: float = 1e-5
    max_grad_norm: float = 0.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    deterministic: bool = False
    checkpoint_dir: str = "checkpoints_resnet50_exact"
    resume_from_checkpoint: Optional[str] = None
    metrics_filename: str = "metrics.csv"


def set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


class Evaluator:
    def __init__(self, val_loader, loss_function, device: str):
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.device = device

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()

        total_loss = 0.0
        total_top1_correct = 0
        total_top5_correct = 0
        total_images = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = model(images)
            loss = self.loss_function(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            predicted_labels = outputs.argmax(dim=1)
            total_top1_correct += (predicted_labels == labels).sum().item()

            topk = min(5, outputs.size(1))
            _, top5_predictions = outputs.topk(topk, dim=1)
            total_top5_correct += (
                top5_predictions.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            )

            total_images += batch_size

        average_loss = total_loss / total_images
        top1_accuracy = 100.0 * total_top1_correct / total_images
        top5_accuracy = 100.0 * total_top5_correct / total_images

        return average_loss, top1_accuracy, top5_accuracy


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: TrainingConfig,
    ):
        set_global_seed(config.seed, deterministic=config.deterministic)

        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.config = config

        self.loss_function = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
        )

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        self.use_amp = (
            config.use_amp
            and config.device.startswith("cuda")
            and torch.cuda.is_available()
        )

        if USE_TORCH_AMP:
            self.scaler = GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = GradScaler(enabled=self.use_amp)

        self.evaluator = Evaluator(
            val_loader=val_loader,
            loss_function=self.loss_function,
            device=config.device,
        )

        self.best_val_accuracy = -1.0
        self.start_epoch = 1

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.checkpoint_dir / config.metrics_filename

        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def train(self):
        print("=" * 70)
        print("Training exact ResNet-50")
        print("=" * 70)

        if self.start_epoch == 1:
            self._init_metrics_file()

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            lr = self._adjust_learning_rate(epoch)

            train_loss, train_accuracy = self.train_one_epoch()

            val_loss, val_top1_accuracy, val_top5_accuracy = self.evaluator.evaluate(
                self.model,
            )

            self._write_metrics_row(
                epoch=epoch,
                learning_rate=lr,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_top1_accuracy=val_top1_accuracy,
                val_top5_accuracy=val_top5_accuracy,
            )

            print(
                f"Epoch {epoch:03d}/{self.config.epochs} | "
                f"LR: {lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:6.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Top1: {val_top1_accuracy:6.2f}% | "
                f"Val Top5: {val_top5_accuracy:6.2f}%"
            )

            is_best = val_top1_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_top1_accuracy
                self.save_checkpoint(
                    filename="best_model.pth",
                    epoch=epoch,
                    val_top1_accuracy=val_top1_accuracy,
                    val_top5_accuracy=val_top5_accuracy,
                )
                print(f"  BEST saved: best_model.pth ({val_top1_accuracy:.2f}%)")

            self.save_checkpoint(
                filename="latest_checkpoint.pth",
                epoch=epoch,
                val_top1_accuracy=val_top1_accuracy,
                val_top5_accuracy=val_top5_accuracy,
            )

        print("=" * 70)
        print(f"Training complete. Best Val Top1: {self.best_val_accuracy:.2f}%")
        print(f"Best model: {self.checkpoint_dir / 'best_model.pth'}")
        print(f"Metrics: {self.metrics_path}")
        print("=" * 70)

    def train_one_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in self.train_loader:
            images = images.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if USE_TORCH_AMP:
                autocast_context = autocast("cuda", enabled=self.use_amp)
            else:
                autocast_context = autocast(enabled=self.use_amp)

            with autocast_context:
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()

                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.config.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size

            predicted_labels = outputs.argmax(dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_images += batch_size

        average_loss = total_loss / total_images
        accuracy = 100.0 * total_correct / total_images

        return average_loss, accuracy

    def _adjust_learning_rate(self, epoch: int) -> float:
        if self.config.warmup_epochs > 0 and epoch <= self.config.warmup_epochs:
            lr = self.config.learning_rate * epoch / self.config.warmup_epochs
        else:
            progress_denominator = max(1, self.config.epochs - self.config.warmup_epochs)
            progress = (epoch - self.config.warmup_epochs) / progress_denominator
            progress = min(max(progress, 0.0), 1.0)

            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.config.min_lr + (
                self.config.learning_rate - self.config.min_lr
            ) * cosine_factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_top1_accuracy: float,
        val_top5_accuracy: float,
    ):
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_val_accuracy": self.best_val_accuracy,
            "val_top1_accuracy": val_top1_accuracy,
            "val_top5_accuracy": val_top5_accuracy,
            "training_config": asdict(self.config),
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.config.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scaler_state = checkpoint.get("scaler_state_dict")
        if self.use_amp and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        self.best_val_accuracy = checkpoint.get("best_val_accuracy", -1.0)
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

        print(f"Resumed from {checkpoint_path} at epoch {self.start_epoch}.")

    def _init_metrics_file(self):
        with open(self.metrics_path, "w", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_top1_accuracy",
                "val_top5_accuracy",
            ])

    def _write_metrics_row(
        self,
        epoch: int,
        learning_rate: float,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_top1_accuracy: float,
        val_top5_accuracy: float,
    ):
        with open(self.metrics_path, "a", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([
                epoch,
                f"{learning_rate:.8f}",
                f"{train_loss:.6f}",
                f"{train_accuracy:.4f}",
                f"{val_loss:.6f}",
                f"{val_top1_accuracy:.4f}",
                f"{val_top5_accuracy:.4f}",
            ])
