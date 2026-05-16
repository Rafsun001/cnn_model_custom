
import csv
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
resnet_dir = repo_root / "ResNet-50"

sys.path.insert(0, str(current_dir))
sys.path.append(str(resnet_dir))

from DataLoader_builder import DataLoaderConfig, TinyImageNetDataModule
from EfficientNetV2SModel import EfficientNetV2SModel
from ImageProcessor import ImageProcessor, ImageProcessingConfig


@dataclass
class EfficientNetTrainingConfig:
    epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.999
    use_amp: bool = True
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    use_cosine: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = str(current_dir / "checkpoints_efficientnetv2s_tinyimagenet")
    resume_from_checkpoint: Optional[str] = None
    metrics_filename: str = "metrics.csv"
    model_dropout_rate: float = 0.2
    model_drop_path_rate: float = 0.1
    save_ema_as_model: bool = True


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class EfficientNetTrainer:
    def __init__(self, model, train_loader, val_loader, config: EfficientNetTrainingConfig):
        set_global_seed(config.seed)
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        self.scheduler = None
        if config.use_cosine and config.epochs > config.warmup_epochs:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs - config.warmup_epochs,
                eta_min=config.min_lr,
            )

        self.use_amp = config.use_amp and config.device.startswith("cuda") and torch.cuda.is_available()
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.ema_state = None
        if config.use_ema:
            self._init_ema()

        self.start_epoch = 1
        self.best_primary_val_accuracy = -1.0
        self.metrics_path = Path(config.checkpoint_dir) / config.metrics_filename
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def train(self):
        if self.start_epoch == 1:
            self._init_metrics_file()

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self._set_warmup_lr(epoch)
            train_loss, train_accuracy = self.train_one_epoch()

            raw_val_loss, raw_val_accuracy, raw_val_top5_accuracy = self.evaluate(self.model)
            ema_val_loss = ""
            ema_val_accuracy = ""
            ema_val_top5_accuracy = ""
            primary_val_accuracy = raw_val_accuracy

            if self.config.use_ema:
                backup = self._swap_ema_weights()
                ema_val_loss, ema_val_accuracy, ema_val_top5_accuracy = self.evaluate(self.model)
                self._restore_weights(backup)
                primary_val_accuracy = ema_val_accuracy

            print(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:6.2f}% | "
                f"Raw Val Acc: {raw_val_accuracy:6.2f}% | "
                f"Raw Top5: {raw_val_top5_accuracy:6.2f}% | "
                f"EMA Val Acc: {self._format_optional_accuracy(ema_val_accuracy)} | "
                f"EMA Top5: {self._format_optional_accuracy(ema_val_top5_accuracy)} | "
                f"LR: {self._get_current_lr():.8f}"
            )

            self._write_metrics_row(
                epoch,
                train_loss,
                train_accuracy,
                raw_val_loss,
                raw_val_accuracy,
                raw_val_top5_accuracy,
                ema_val_loss,
                ema_val_accuracy,
                ema_val_top5_accuracy,
            )

            if primary_val_accuracy > self.best_primary_val_accuracy:
                self.best_primary_val_accuracy = primary_val_accuracy
                self.save_checkpoint(
                    "best_model.pth",
                    epoch,
                    raw_val_accuracy,
                    raw_val_top5_accuracy,
                    ema_val_accuracy,
                    ema_val_top5_accuracy,
                    is_best=True,
                )
                self.save_checkpoint(
                    "best_raw_model.pth",
                    epoch,
                    raw_val_accuracy,
                    raw_val_top5_accuracy,
                    ema_val_accuracy,
                    ema_val_top5_accuracy,
                    is_best=True,
                    save_ema_as_model=False,
                )
                if self.config.use_ema and self.ema_state is not None:
                    self.save_checkpoint(
                        "best_ema_model.pth",
                        epoch,
                        raw_val_accuracy,
                        raw_val_top5_accuracy,
                        ema_val_accuracy,
                        ema_val_top5_accuracy,
                        is_best=True,
                        save_ema_as_model=True,
                    )
                print(f"  Saved best_model.pth (primary val acc: {primary_val_accuracy:.2f}%)")

            self.save_checkpoint(
                "latest_checkpoint.pth",
                epoch,
                raw_val_accuracy,
                raw_val_top5_accuracy,
                ema_val_accuracy,
                ema_val_top5_accuracy,
                is_best=False,
                save_ema_as_model=False,
            )

            if self.scheduler is not None and epoch > self.config.warmup_epochs:
                self.scheduler.step()

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in self.train_loader:
            images = images.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)
            labels_for_accuracy = labels

            if self.config.mixup_alpha > 0.0:
                images, labels_a, labels_b, mixup_lam = self._mixup_batch(images, labels)
            else:
                labels_a = labels
                labels_b = labels
                mixup_lam = 1.0

            self.optimizer.zero_grad(set_to_none=True)

            autocast_context = autocast("cuda", enabled=self.use_amp)

            with autocast_context:
                outputs = self.model(images)
                loss = (
                    mixup_lam * self.loss_function(outputs, labels_a)
                    + (1.0 - mixup_lam) * self.loss_function(outputs, labels_b)
                )

            self.scaler.scale(loss).backward()

            if self.config.max_grad_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config.use_ema:
                self._update_ema()

            total_loss += loss.item() * images.size(0)
            predicted_labels = outputs.argmax(dim=1)
            total_correct += (predicted_labels == labels_for_accuracy).sum().item()
            total_images += labels.size(0)

        return total_loss / total_images, 100.0 * total_correct / total_images

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total_loss = 0.0
        total_top1_correct = 0
        total_top5_correct = 0
        total_images = 0

        for images, labels in self.val_loader:
            images = images.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            outputs = model(images)
            loss = self.loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_top1_correct += (outputs.argmax(dim=1) == labels).sum().item()
            topk = min(5, outputs.size(1))
            _, top5_predictions = outputs.topk(topk, dim=1)
            total_top5_correct += top5_predictions.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            total_images += labels.size(0)

        return (
            total_loss / total_images,
            100.0 * total_top1_correct / total_images,
            100.0 * total_top5_correct / total_images,
        )

    def save_checkpoint(
        self,
        filename,
        epoch,
        raw_val_accuracy,
        raw_val_top5_accuracy,
        ema_val_accuracy,
        ema_val_top5_accuracy,
        is_best,
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
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": self.ema_state,
            "epoch": epoch,
            "best_primary_val_accuracy": self.best_primary_val_accuracy,
            "raw_val_accuracy": raw_val_accuracy,
            "raw_val_top5_accuracy": raw_val_top5_accuracy,
            "ema_val_accuracy": ema_val_accuracy,
            "ema_val_top5_accuracy": ema_val_top5_accuracy,
            "is_best": is_best,
            "training_config": asdict(self.config),
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        ema_state = checkpoint.get("ema_state_dict")
        if self.config.use_ema and ema_state is not None:
            self.ema_state = {
                name: tensor.to(self.config.device)
                for name, tensor in ema_state.items()
            }

        self.best_primary_val_accuracy = checkpoint.get(
            "best_primary_val_accuracy",
            self.best_primary_val_accuracy,
        )
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from {checkpoint_path} at epoch {self.start_epoch}.")

    def _init_metrics_file(self):
        with open(self.metrics_path, "w", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "raw_val_loss",
                "raw_val_accuracy",
                "raw_val_top5_accuracy",
                "ema_val_loss",
                "ema_val_accuracy",
                "ema_val_top5_accuracy",
                "learning_rate",
            ])

    def _write_metrics_row(
        self,
        epoch,
        train_loss,
        train_accuracy,
        raw_val_loss,
        raw_val_accuracy,
        raw_val_top5_accuracy,
        ema_val_loss,
        ema_val_accuracy,
        ema_val_top5_accuracy,
    ):
        with open(self.metrics_path, "a", newline="") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_accuracy:.4f}",
                f"{raw_val_loss:.6f}",
                f"{raw_val_accuracy:.4f}",
                f"{raw_val_top5_accuracy:.4f}",
                f"{ema_val_loss:.6f}" if ema_val_loss != "" else "",
                f"{ema_val_accuracy:.4f}" if ema_val_accuracy != "" else "",
                f"{ema_val_top5_accuracy:.4f}" if ema_val_top5_accuracy != "" else "",
                f"{self._get_current_lr():.8f}",
            ])

    def _set_warmup_lr(self, epoch):
        if self.config.warmup_epochs <= 0 or epoch > self.config.warmup_epochs:
            return
        warmup_lr = self.config.learning_rate * epoch / self.config.warmup_epochs
        for group in self.optimizer.param_groups:
            group["lr"] = warmup_lr

    def _get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _mixup_batch(self, images, labels):
        mixup_dist = torch.distributions.Beta(self.config.mixup_alpha, self.config.mixup_alpha)
        lam = mixup_dist.sample().item()
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        return mixed_images, labels_a, labels_b, lam

    def _init_ema(self):
        self.ema_state = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _update_ema(self):
        with torch.no_grad():
            for name, tensor in self.model.state_dict().items():
                ema_tensor = self.ema_state[name]
                if tensor.dtype.is_floating_point:
                    ema_tensor.mul_(self.config.ema_decay).add_(tensor, alpha=1.0 - self.config.ema_decay)
                else:
                    ema_tensor.copy_(tensor)

    def _swap_ema_weights(self):
        backup = {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.ema_state, strict=True)
        return backup

    def _restore_weights(self, backup):
        self.model.load_state_dict(backup, strict=True)

    def _clone_state_dict(self, state_dict):
        return {
            name: tensor.detach().clone()
            for name, tensor in state_dict.items()
        }

    def _checkpoint_model_state(self, save_ema_as_model):
        if save_ema_as_model and self.config.use_ema and self.ema_state is not None:
            return self._clone_state_dict(self.ema_state), "ema"
        return self._clone_state_dict(self.model.state_dict()), "raw"

    @staticmethod
    def _format_optional_accuracy(value):
        if value == "":
            return "   n/a"
        return f"{value:6.2f}%"


def main():
    seed = 42
    set_global_seed(seed)

    print("=" * 70)
    print("EfficientNetV2-S Training on Tiny ImageNet")
    print("=" * 70)

    image_config = ImageProcessingConfig(
        image_size=64,
        use_strong_aug=False,
        random_erasing_prob=0.0,
    )
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

    training_config = EfficientNetTrainingConfig(seed=seed)
    model = EfficientNetV2SModel(
        num_classes=200,
        dropout_rate=training_config.model_dropout_rate,
        drop_path_rate=training_config.model_drop_path_rate,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Device: {training_config.device}")
    print(f"Optimizer: AdamW")
    print(f"LR: {training_config.learning_rate}")
    print(f"EMA enabled: {training_config.use_ema}")
    print(f"Checkpoint dir: {training_config.checkpoint_dir}")

    trainer = EfficientNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
