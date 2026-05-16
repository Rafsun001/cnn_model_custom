import torch
import torch.nn as nn
import torch.nn.functional as F

from BoxUtils import complete_iou_loss, dist2bbox, make_anchors
from ProgLoss import ProgLoss
from STALAssigner import STALAssigner
from YoloV26Config import NUM_CLASSES, STRIDES


class YOLO26DetectionLoss(nn.Module):
    """
    YOLO26-style dual-head detection loss.

    Expected prediction input:
        {
            "one2many": [P3, P4, P5],
            "one2one":  [P3, P4, P5],
        }

    Each prediction tensor:
        [B, 4 + num_classes, H, W]

    Target format per image:
        [class_id, x1, y1, x2, y2]
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        image_size=640,
        strides=STRIDES,
        topk_one2many=8,
        topk_one2one=1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides

        self.assigner = STALAssigner(
            num_classes=num_classes,
            image_size=image_size,
            topk_one2many=topk_one2many,
            topk_one2one=topk_one2one,
        )

        self.prog_loss = ProgLoss()

    def forward(self, predictions, targets, epoch=None, total_epochs=None):
        if not isinstance(predictions, dict):
            raise TypeError(
                "YOLO26DetectionLoss expects predictions to be a dictionary "
                "with keys 'one2many' and 'one2one'."
            )

        weights = self.prog_loss.weights(
            epoch=epoch,
            total_epochs=total_epochs,
        )

        one2many_stats = self._branch_loss(
            predictions=predictions["one2many"],
            targets=targets,
            mode="one2many",
            weights=weights,
        )

        one2one_stats = self._branch_loss(
            predictions=predictions["one2one"],
            targets=targets,
            mode="one2one",
            weights=weights,
        )

        total_loss = (
            weights["one2one"] * one2one_stats["loss"]
            + weights["one2many"] * one2many_stats["loss"]
        )

        box_loss = 0.5 * (
            one2many_stats["box_loss"] + one2one_stats["box_loss"]
        )

        class_loss = 0.5 * (
            one2many_stats["class_loss"] + one2one_stats["class_loss"]
        )

        positive_count = (
            one2many_stats["positive_count"] + one2one_stats["positive_count"]
        )

        return {
            "loss": total_loss,
            "box_loss": box_loss.detach(),
            "class_loss": class_loss.detach(),
            "negative_loss": predictions["one2one"][0].new_tensor(0.0),
            "positive_count": positive_count.detach(),
            "one2many_loss": one2many_stats["loss"].detach(),
            "one2one_loss": one2one_stats["loss"].detach(),
            "one2many_positive_count": one2many_stats["positive_count"].detach(),
            "one2one_positive_count": one2one_stats["positive_count"].detach(),
        }

    def _branch_loss(self, predictions, targets, mode, weights):
        decoded = self._decode_raw_predictions(predictions)

        pred_boxes = decoded["boxes"]
        class_logits = decoded["class_logits"]
        pred_scores = class_logits.detach().sigmoid()
        anchor_points = decoded["anchor_points"]

        assignments = self.assigner.assign(
            targets=targets,
            pred_boxes=pred_boxes.detach(),
            pred_scores=pred_scores,
            anchor_points=anchor_points,
            mode=mode,
        )

        target_boxes = assignments["target_boxes"]
        target_scores = assignments["target_scores"]
        fg_mask = assignments["fg_mask"]
        small_weights = assignments["small_weights"]

        target_scores_sum = target_scores.sum().clamp(min=1.0)

        class_loss = F.binary_cross_entropy_with_logits(
            class_logits,
            target_scores,
            reduction="none",
        ).sum() / target_scores_sum

        if fg_mask.any():
            pred_fg_boxes = pred_boxes[fg_mask]
            target_fg_boxes = target_boxes[fg_mask]
            fg_weights = small_weights[fg_mask]

            box_loss_values = complete_iou_loss(
                pred_boxes=pred_fg_boxes,
                target_boxes=target_fg_boxes,
            )

            box_loss = (box_loss_values * fg_weights).sum() / fg_weights.sum().clamp(min=1.0)
            positive_count = fg_mask.sum().to(dtype=class_logits.dtype)
        else:
            box_loss = class_logits.new_tensor(0.0)
            positive_count = class_logits.new_tensor(0.0)

        loss = (
            weights["box"] * box_loss
            + weights["class"] * class_loss
        )

        return {
            "loss": loss,
            "box_loss": box_loss,
            "class_loss": class_loss,
            "positive_count": positive_count,
        }

    def _decode_raw_predictions(self, predictions):
        all_boxes = []
        all_class_logits = []
        feature_shapes = []

        batch_size = predictions[0].shape[0]
        device = predictions[0].device
        dtype = predictions[0].dtype

        for prediction in predictions:
            _, channels, height, width = prediction.shape

            expected_channels = 4 + self.num_classes
            if channels != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels, got {channels}."
                )

            feature_shapes.append((height, width))

        anchor_points_grid, stride_tensor = make_anchors(
            feature_shapes=feature_shapes,
            strides=self.strides,
            device=device,
            dtype=dtype,
            grid_cell_offset=0.5,
        )

        start = 0

        for prediction, stride in zip(predictions, self.strides):
            _, _, height, width = prediction.shape
            num_positions = height * width

            prediction = prediction.view(
                batch_size,
                4 + self.num_classes,
                num_positions,
            )

            box_raw = prediction[:, :4, :].permute(0, 2, 1).contiguous()
            class_logits = prediction[:, 4:, :].permute(0, 2, 1).contiguous()

            level_anchor_points = anchor_points_grid[start:start + num_positions]
            level_stride_tensor = stride_tensor[start:start + num_positions]

            # Keep distances non-negative for stable distance-to-box decoding.
            box_distances = box_raw.clamp(min=0.0)

            boxes = dist2bbox(
                distance=box_distances,
                anchor_points=level_anchor_points,
                xywh=False,
            )

            boxes = boxes * level_stride_tensor.unsqueeze(0)

            all_boxes.append(boxes)
            all_class_logits.append(class_logits)

            start += num_positions

        boxes = torch.cat(all_boxes, dim=1)
        class_logits = torch.cat(all_class_logits, dim=1)

        anchor_points_pixel = anchor_points_grid * stride_tensor

        return {
            "boxes": boxes,
            "class_logits": class_logits,
            "anchor_points": anchor_points_pixel,
        }