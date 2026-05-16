import csv
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast

current_dir = Path(__file__).resolve().parent

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from MuSGD import MuSGD
from YOLO26DetectionLoss import YOLO26DetectionLoss
from YOLO26ExplicitModel import YOLO26ExplicitModel
from YoloV26Dataset import YoloV26DataConfig, YoloV26DataModule
from YoloV26ImageProcessor import (
    YoloV26ImageProcessingConfig,
    YoloV26ImageProcessor,
)
from YoloV26Metrics import YoloV26DetectionMetrics


@dataclass
class YoloV26TrainingConfig:
    epochs: int = 100

    learning_rate: float = 0.0054
    momentum: float = 0.947
    weight_decay: float = 0.00064

    muon_w: float = 0.528
    sgd_w: float = 0.674

    use_musgd: bool = True
    use_amp: bool = True

    warmup_epochs: int = 3
    min_lr: float = 1e-5
    use_cosine: bool = True
    max_grad_norm: float = 10.0

    num_classes: int = 80
    image_size: int = 640

    score_threshold: float = 0.25
    iou_threshold: float = 0.5

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = str(current_dir / "checkpoints_yolov26")
    resume_from_checkpoint: Optional[str] = None
    metrics_filename: str = "metrics.csv"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


class YoloV26Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: YoloV26TrainingConfig,
    ):
        set_global_seed(config.seed)

        self.model = model.to(config.device)

        if hasattr(self.model, "bias_init"):
            self.model.bias_init()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.loss_function = YOLO26DetectionLoss(
            num_classes=config.num_classes,
            image_size=config.image_size,
        )

        self.metric = YoloV26DetectionMetrics(
            iou_threshold=config.iou_threshold,
            score_threshold=config.score_threshold,
        )

        if config.use_musgd:
            self.optimizer = MuSGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                muon_w=config.muon_w,
                sgd_w=config.sgd_w,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )

        self.scheduler = None

        if config.use_cosine and config.epochs > config.warmup_epochs:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs - config.warmup_epochs,
                eta_min=config.min_lr,
            )

        self.use_amp = (
            config.use_amp
            and config.device.startswith("cuda")
            and torch.cuda.is_available()
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.start_epoch = 1
        self.best_recall = -1.0

        self.metrics_path = Path(config.checkpoint_dir) / config.metrics_filename
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def train(self):
        if self.start_epoch == 1:
            self._init_metrics_file()

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self._set_warmup_lr(epoch)

            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate(epoch)

            print(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train Loss: {train_stats['loss']:.4f} | "
                f"Box: {train_stats['box_loss']:.4f} | "
                f"Cls: {train_stats['class_loss']:.4f} | "
                f"O2M: {train_stats['one2many_loss']:.4f} | "
                f"O2O: {train_stats['one2one_loss']:.4f} | "
                f"Val Loss: {val_stats['loss']:.4f} | "
                f"Precision: {val_stats['precision']:.4f} | "
                f"Recall: {val_stats['recall']:.4f} | "
                f"Mean IoU: {val_stats['mean_iou']:.4f} | "
                f"LR: {self._get_current_lr():.8f}"
            )

            self._write_metrics_row(epoch, train_stats, val_stats)

            if val_stats["recall"] > self.best_recall:
                self.best_recall = val_stats["recall"]

                self.save_checkpoint(
                    filename="best_model.pth",
                    epoch=epoch,
                    val_stats=val_stats,
                    is_best=True,
                )

                print(
                    f"  Saved best_model.pth "
                    f"(best recall: {self.best_recall:.4f})"
                )

            self.save_checkpoint(
                filename="latest_checkpoint.pth",
                epoch=epoch,
                val_stats=val_stats,
                is_best=False,
            )

            if self.scheduler is not None and epoch > self.config.warmup_epochs:
                self.scheduler.step()

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_box_loss = 0.0
        total_class_loss = 0.0
        total_positive = 0.0
        total_one2many_loss = 0.0
        total_one2one_loss = 0.0
        total_batches = 0

        for images, targets in self.train_loader:
            images = images.to(
                self.config.device,
                non_blocking=True,
            )

            targets = [
                target.to(self.config.device, non_blocking=True)
                for target in targets
            ]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                predictions = self.model(images)

                loss_dict = self.loss_function(
                    predictions=predictions,
                    targets=targets,
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                )

                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.config.max_grad_norm > 0.0:
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach().item())
            total_box_loss += float(loss_dict["box_loss"].item())
            total_class_loss += float(loss_dict["class_loss"].item())
            total_positive += float(loss_dict["positive_count"].item())
            total_one2many_loss += float(loss_dict["one2many_loss"].item())
            total_one2one_loss += float(loss_dict["one2one_loss"].item())
            total_batches += 1

        return {
            "loss": total_loss / max(1, total_batches),
            "box_loss": total_box_loss / max(1, total_batches),
            "class_loss": total_class_loss / max(1, total_batches),
            "positive_count": total_positive / max(1, total_batches),
            "one2many_loss": total_one2many_loss / max(1, total_batches),
            "one2one_loss": total_one2one_loss / max(1, total_batches),
        }

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        self.metric.reset()

        total_loss = 0.0
        total_batches = 0

        for images, targets in self.val_loader:
            images = images.to(
                self.config.device,
                non_blocking=True,
            )

            targets = [
                target.to(self.config.device, non_blocking=True)
                for target in targets
            ]

            # Raw predictions for validation loss.
            raw_predictions = self.model(images, raw=True)

            loss_dict = self.loss_function(
                predictions=raw_predictions,
                targets=targets,
                epoch=epoch,
                total_epochs=self.config.epochs,
            )

            # Normal eval output: [B, 300, 6].
            detections = self.model(images)

            self.metric.update_from_detections(
                detections=detections,
                targets=targets,
            )

            total_loss += float(loss_dict["loss"].item())
            total_batches += 1

        metric_values = self.metric.compute()
        metric_values["loss"] = total_loss / max(1, total_batches)

        return metric_values

    def save_checkpoint(self, filename, epoch, val_stats, is_best):
        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "best_recall": self.best_recall,
            "val_stats": val_stats,
            "is_best": is_best,
            "training_config": asdict(self.config),
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.config.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"],
        )

        scheduler_state = checkpoint.get("scheduler_state_dict")

        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("scaler_state_dict")

        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        self.best_recall = checkpoint.get("best_recall", self.best_recall)
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

        print(f"Resumed from {checkpoint_path} at epoch {self.start_epoch}.")

    def _init_metrics_file(self):
        with open(self.metrics_path, "w", newline="") as metrics_file:
            writer = csv.writer(metrics_file)

            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_box_loss",
                    "train_class_loss",
                    "train_positive_count",
                    "train_one2many_loss",
                    "train_one2one_loss",
                    "val_loss",
                    "precision",
                    "recall",
                    "mean_iou",
                    "learning_rate",
                ]
            )

    def _write_metrics_row(self, epoch, train_stats, val_stats):
        with open(self.metrics_path, "a", newline="") as metrics_file:
            writer = csv.writer(metrics_file)

            writer.writerow(
                [
                    epoch,
                    f"{train_stats['loss']:.6f}",
                    f"{train_stats['box_loss']:.6f}",
                    f"{train_stats['class_loss']:.6f}",
                    f"{train_stats['positive_count']:.4f}",
                    f"{train_stats['one2many_loss']:.6f}",
                    f"{train_stats['one2one_loss']:.6f}",
                    f"{val_stats['loss']:.6f}",
                    f"{val_stats['precision']:.6f}",
                    f"{val_stats['recall']:.6f}",
                    f"{val_stats['mean_iou']:.6f}",
                    f"{self._get_current_lr():.8f}",
                ]
            )

    def _set_warmup_lr(self, epoch):
        if self.config.warmup_epochs <= 0:
            return

        if epoch > self.config.warmup_epochs:
            return

        warmup_lr = (
            self.config.learning_rate
            * epoch
            / self.config.warmup_epochs
        )

        for group in self.optimizer.param_groups:
            group["lr"] = warmup_lr

    def _get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def main():
    seed = 42
    set_global_seed(seed)

    image_config = YoloV26ImageProcessingConfig(
        image_size=640,
    )

    image_processor = YoloV26ImageProcessor(
        config=image_config,
    )

    data_config = YoloV26DataConfig(
        data_root=str(current_dir / "datasets" / "yolo"),
        batch_size=8,
        num_workers=0,
    )

    data_module = YoloV26DataModule(
        data_config=data_config,
        image_processor=image_processor,
    )

    data_module.setup()

    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    training_config = YoloV26TrainingConfig(
        seed=seed,
        image_size=image_config.image_size,
    )

    model = YOLO26ExplicitModel()

    print("=" * 70)
    print("YOLO26 Training")
    print("=" * 70)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Device: {training_config.device}")
    print(f"Optimizer: {'MuSGD' if training_config.use_musgd else 'SGD'}")
    print(f"Checkpoint dir: {training_config.checkpoint_dir}")

    trainer = YoloV26Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()