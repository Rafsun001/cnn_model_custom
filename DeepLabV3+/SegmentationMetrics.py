import torch


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.float64,
        )

    def reset(self):
        self.confusion_matrix.zero_()

    def update(self, logits, targets):
        predictions = torch.argmax(logits, dim=1)

        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        valid_mask = targets != self.ignore_index
        valid_mask = valid_mask & (targets >= 0)
        valid_mask = valid_mask & (targets < self.num_classes)

        targets = targets[valid_mask]
        predictions = predictions[valid_mask]

        indices = targets * self.num_classes + predictions

        confusion = torch.bincount(
            indices,
            minlength=self.num_classes * self.num_classes,
        )

        confusion = confusion.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion

    def compute(self):
        confusion = self.confusion_matrix

        true_positive = torch.diag(confusion)
        ground_truth = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)

        union = ground_truth + predicted - true_positive

        # IoU for each class.
        # Classes with union == 0 are absent from the validation set,
        # so they should not reduce mean IoU.
        valid_classes = union > 0

        iou = torch.zeros_like(true_positive)
        iou[valid_classes] = true_positive[valid_classes] / union[valid_classes]

        if valid_classes.any():
            mean_iou = iou[valid_classes].mean().item()
        else:
            mean_iou = 0.0

        pixel_accuracy = true_positive.sum() / torch.clamp(confusion.sum(), min=1.0)
        pixel_accuracy = pixel_accuracy.item()

        class_iou = []
        for class_index in range(self.num_classes):
            if valid_classes[class_index]:
                class_iou.append(iou[class_index].item())
            else:
                class_iou.append(None)

        return {
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": mean_iou,
            "class_iou": class_iou,
        }
