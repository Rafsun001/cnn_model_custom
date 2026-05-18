import torch
import torch.nn as nn
import os
from .SegmentationMetrics import SegmentationMetrics
from .freeze_batch_norm_layers import freeze_batch_norm_layers


class DeepLabV3PlusTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.config.batch_size == 1 and not self.config.freeze_batch_norm:
            raise ValueError(
                "Batch size 1 can break BatchNorm in the ASPP image-pooling branch. "
                "Use batch_size >= 2 or set freeze_batch_norm=True."
            )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.config.ignore_index,
        )

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.poly_lr_lambda,
        )

        self.use_amp = self.config.use_amp and self.device.type == "cuda"

        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.best_miou = 0.0

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def poly_lr_lambda(self, epoch):
        power = 0.9
        progress = epoch / max(self.config.epochs, 1)
        return (1.0 - progress) ** power

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()

        if self.config.freeze_batch_norm:
            freeze_batch_norm_layers(self.model)

        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            images, masks = batch

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()

            if (batch_index + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch [{epoch}/{self.config.epochs}] "
                    f"Step [{batch_index + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {current_lr:.6f}"
                )

        average_loss = running_loss / max(len(train_loader), 1)

        return average_loss

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()

        metrics = SegmentationMetrics(
            num_classes=self.config.num_classes,
            ignore_index=self.config.ignore_index,
        )

        running_loss = 0.0

        for batch in val_loader:
            images, masks = batch

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            running_loss += loss.item()
            metrics.update(logits, masks)

        average_loss = running_loss / max(len(val_loader), 1)
        metric_result = metrics.compute()

        return average_loss, metric_result

    def save_checkpoint(self, epoch, val_loss, val_miou, checkpoint_name):
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            checkpoint_name,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou": self.best_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)

    def fit(self, train_loader, val_loader):
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_one_epoch(
                train_loader=train_loader,
                epoch=epoch,
            )

            val_loss, val_metrics = self.validate(val_loader)

            val_pixel_accuracy = val_metrics["pixel_accuracy"]
            val_miou = val_metrics["mean_iou"]

            self.scheduler.step()

            print("=" * 80)
            print(f"Epoch:              {epoch}")
            print(f"Train Loss:         {train_loss:.4f}")
            print(f"Val Loss:           {val_loss:.4f}")
            print(f"Val Pixel Accuracy: {val_pixel_accuracy:.4f}")
            print(f"Val mIoU:           {val_miou:.4f}")
            print("=" * 80)

            self.save_checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                val_miou=val_miou,
                checkpoint_name=self.config.latest_checkpoint_name,
            )

            if val_miou > self.best_miou:
                self.best_miou = val_miou

                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    val_miou=val_miou,
                    checkpoint_name=self.config.best_checkpoint_name,
                )

                print(f"Saved new best model with mIoU: {val_miou:.4f}")
