import torch

from BoxUtils import box_iou


class YoloV26DetectionMetrics:
    """
    Lightweight detection metrics.

    Reports:
        precision
        recall
        mean_iou

    Supports detections in:
        [B, K, 6]

    Detection format:
        x1, y1, x2, y2, score, class_id
    """

    def __init__(self, iou_threshold=0.5, score_threshold=0.25):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.reset()

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        self.iou_sum = 0.0
        self.matched_count = 0

    @torch.no_grad()
    def update_from_detections(self, detections, targets):
        boxes = detections[..., 0:4]
        scores = detections[..., 4]
        labels = detections[..., 5].long()

        self.update(
            boxes=boxes,
            scores=scores,
            labels=labels,
            targets=targets,
        )

    @torch.no_grad()
    def update(self, boxes, scores, labels, targets):
        batch_size = boxes.shape[0]

        for batch_index in range(batch_size):
            pred_mask = scores[batch_index] >= self.score_threshold

            pred_boxes = boxes[batch_index][pred_mask]
            pred_scores = scores[batch_index][pred_mask]
            pred_labels = labels[batch_index][pred_mask]

            target = targets[batch_index].to(boxes.device)

            if target.numel() == 0:
                self.false_positives += int(pred_boxes.shape[0])
                continue

            target_labels = target[:, 0].long()
            target_boxes = target[:, 1:5]

            used_targets = torch.zeros(
                target_boxes.shape[0],
                dtype=torch.bool,
                device=boxes.device,
            )

            if pred_boxes.numel() == 0:
                self.false_negatives += int(target_boxes.shape[0])
                continue

            order = torch.argsort(pred_scores, descending=True)

            for pred_index in order:
                same_class = target_labels == pred_labels[pred_index]
                available = same_class & (~used_targets)

                if not available.any():
                    self.false_positives += 1
                    continue

                candidate_indices = torch.where(available)[0]

                ious = box_iou(
                    pred_boxes[pred_index].unsqueeze(0),
                    target_boxes[candidate_indices],
                ).squeeze(0)

                best_iou, local_index = ious.max(dim=0)
                target_index = candidate_indices[local_index]

                if best_iou >= self.iou_threshold:
                    self.true_positives += 1
                    self.iou_sum += float(best_iou.item())
                    self.matched_count += 1
                    used_targets[target_index] = True
                else:
                    self.false_positives += 1

            self.false_negatives += int((~used_targets).sum().item())

    def compute(self):
        precision = self.true_positives / max(
            1,
            self.true_positives + self.false_positives,
        )

        recall = self.true_positives / max(
            1,
            self.true_positives + self.false_negatives,
        )

        mean_iou = self.iou_sum / max(1, self.matched_count)

        return {
            "precision": precision,
            "recall": recall,
            "mean_iou": mean_iou,
        }