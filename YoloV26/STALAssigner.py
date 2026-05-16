import torch

from BoxUtils import box_iou, box_xyxy_to_xywh


class STALAssigner:
    """
    Small-Target-Aware Label Assignment.

    This is a clean YOLO26-style assigner for your educational implementation.

    It supports:
        - one2many assignment: multiple anchors per object
        - one2one assignment: one clean anchor per object
        - small/tiny object protection

    Input target format per image:
        [class_id, x1, y1, x2, y2]

    Output:
        target_labels: [B, N]
        target_boxes:  [B, N, 4]
        target_scores: [B, N, num_classes]
        fg_mask:       [B, N]
        small_weights: [B, N]
    """

    def __init__(
        self,
        num_classes=80,
        image_size=640,
        topk_one2many=8,
        topk_one2one=1,
        alpha=0.5,
        beta=6.0,
        small_size=32.0,
        tiny_size=8.0,
        tiny_min_candidates=4,
    ):
        self.num_classes = num_classes
        self.image_size = image_size

        self.topk_one2many = topk_one2many
        self.topk_one2one = topk_one2one

        self.alpha = alpha
        self.beta = beta

        self.small_size = small_size
        self.tiny_size = tiny_size
        self.tiny_min_candidates = tiny_min_candidates

    @torch.no_grad()
    def assign(
        self,
        targets,
        pred_boxes,
        pred_scores,
        anchor_points,
        mode="one2many",
    ):
        """
        pred_boxes:
            [B, N, 4], xyxy pixel boxes

        pred_scores:
            [B, N, num_classes], sigmoid class probabilities

        anchor_points:
            [N, 2], pixel anchor centers

        mode:
            "one2many" or "one2one"
        """

        batch_size, num_anchors, _ = pred_boxes.shape
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        target_labels = torch.full(
            (batch_size, num_anchors),
            -1,
            dtype=torch.long,
            device=device,
        )

        target_boxes = torch.zeros(
            batch_size,
            num_anchors,
            4,
            dtype=dtype,
            device=device,
        )

        target_scores = torch.zeros(
            batch_size,
            num_anchors,
            self.num_classes,
            dtype=dtype,
            device=device,
        )

        fg_mask = torch.zeros(
            batch_size,
            num_anchors,
            dtype=torch.bool,
            device=device,
        )

        small_weights = torch.ones(
            batch_size,
            num_anchors,
            dtype=dtype,
            device=device,
        )

        for batch_index in range(batch_size):
            image_targets = self._get_image_targets(
                targets=targets,
                batch_index=batch_index,
                device=device,
            )

            if image_targets.numel() == 0:
                continue

            gt_labels = image_targets[:, 0].long()
            gt_boxes = image_targets[:, 1:5].to(dtype=dtype)

            valid_label_mask = (
                (gt_labels >= 0)
                & (gt_labels < self.num_classes)
            )

            gt_labels = gt_labels[valid_label_mask]
            gt_boxes = gt_boxes[valid_label_mask]

            if gt_boxes.numel() == 0:
                continue

            assigned = self._assign_single_image(
                gt_labels=gt_labels,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes[batch_index],
                pred_scores=pred_scores[batch_index],
                anchor_points=anchor_points,
                mode=mode,
            )

            assigned_anchor_indices = assigned["anchor_indices"]
            assigned_gt_indices = assigned["gt_indices"]
            assigned_scores = assigned["scores"]
            assigned_small_weights = assigned["small_weights"]

            if assigned_anchor_indices.numel() == 0:
                continue

            assigned_labels = gt_labels[assigned_gt_indices]
            assigned_boxes = gt_boxes[assigned_gt_indices]

            target_labels[batch_index, assigned_anchor_indices] = assigned_labels
            target_boxes[batch_index, assigned_anchor_indices] = assigned_boxes

            target_scores[
                batch_index,
                assigned_anchor_indices,
                assigned_labels,
            ] = assigned_scores

            fg_mask[batch_index, assigned_anchor_indices] = True
            small_weights[batch_index, assigned_anchor_indices] = assigned_small_weights

        return {
            "target_labels": target_labels,
            "target_boxes": target_boxes,
            "target_scores": target_scores,
            "fg_mask": fg_mask,
            "small_weights": small_weights,
        }

    def _assign_single_image(
        self,
        gt_labels,
        gt_boxes,
        pred_boxes,
        pred_scores,
        anchor_points,
        mode,
    ):
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        num_anchors = pred_boxes.shape[0]
        num_gt = gt_boxes.shape[0]

        if num_gt == 0 or num_anchors == 0:
            return self._empty_result(device=device, dtype=dtype)

        inside_gt_mask = self._select_candidates_in_gts(
            anchor_points=anchor_points,
            gt_boxes=gt_boxes,
        )

        ious = box_iou(pred_boxes, gt_boxes).clamp(min=0.0, max=1.0)

        class_scores = pred_scores[:, gt_labels].clamp(min=1e-9, max=1.0)

        alignment_metric = (
            class_scores.pow(self.alpha)
            * ious.pow(self.beta)
        )

        alignment_metric = alignment_metric * inside_gt_mask.to(dtype)

        topk = self.topk_one2many if mode == "one2many" else self.topk_one2one

        selected_mask = self._select_topk_candidates(
            alignment_metric=alignment_metric,
            inside_gt_mask=inside_gt_mask,
            gt_boxes=gt_boxes,
            anchor_points=anchor_points,
            topk=topk,
            mode=mode,
        )

        if selected_mask.sum() == 0:
            return self._empty_result(device=device, dtype=dtype)

        assigned_anchor_indices, assigned_gt_indices = torch.where(selected_mask)

        metric_values = alignment_metric[
            assigned_anchor_indices,
            assigned_gt_indices,
        ]

        # If one anchor is assigned to multiple GTs, keep the strongest one.
        unique_anchor_indices = assigned_anchor_indices.unique()
        final_anchor_indices = []
        final_gt_indices = []
        final_scores = []
        final_small_weights = []

        gt_sizes = self._get_gt_sizes(gt_boxes)

        for anchor_index in unique_anchor_indices:
            candidate_mask = assigned_anchor_indices == anchor_index

            candidate_gt_indices = assigned_gt_indices[candidate_mask]
            candidate_metric_values = metric_values[candidate_mask]

            best_local_index = candidate_metric_values.argmax()
            best_gt_index = candidate_gt_indices[best_local_index]

            best_iou = ious[anchor_index, best_gt_index].clamp(min=0.0, max=1.0)

            gt_size = gt_sizes[best_gt_index]
            small_weight = self._small_weight(gt_size, dtype=dtype, device=device)

            final_anchor_indices.append(anchor_index)
            final_gt_indices.append(best_gt_index)
            final_scores.append(best_iou)
            final_small_weights.append(small_weight)

        return {
            "anchor_indices": torch.stack(final_anchor_indices).long(),
            "gt_indices": torch.stack(final_gt_indices).long(),
            "scores": torch.stack(final_scores).to(dtype=dtype),
            "small_weights": torch.stack(final_small_weights).to(dtype=dtype),
        }

    def _select_candidates_in_gts(self, anchor_points, gt_boxes):
        x = anchor_points[:, 0]
        y = anchor_points[:, 1]

        left = x[:, None] >= gt_boxes[None, :, 0]
        top = y[:, None] >= gt_boxes[None, :, 1]
        right = x[:, None] <= gt_boxes[None, :, 2]
        bottom = y[:, None] <= gt_boxes[None, :, 3]

        return left & top & right & bottom

    def _select_topk_candidates(
        self,
        alignment_metric,
        inside_gt_mask,
        gt_boxes,
        anchor_points,
        topk,
        mode,
    ):
        num_anchors, num_gt = alignment_metric.shape
        device = alignment_metric.device

        selected_mask = torch.zeros(
            num_anchors,
            num_gt,
            dtype=torch.bool,
            device=device,
        )

        gt_sizes = self._get_gt_sizes(gt_boxes)

        for gt_index in range(num_gt):
            metric_for_gt = alignment_metric[:, gt_index]
            candidate_mask = inside_gt_mask[:, gt_index]

            candidate_indices = torch.where(candidate_mask)[0]

            desired_topk = topk

            if mode == "one2many":
                if gt_sizes[gt_index] <= self.tiny_size:
                    desired_topk = max(topk, self.tiny_min_candidates)
                elif gt_sizes[gt_index] <= self.small_size:
                    desired_topk = max(topk, 5)

            if candidate_indices.numel() == 0:
                candidate_indices = self._nearest_anchor_indices(
                    anchor_points=anchor_points,
                    gt_box=gt_boxes[gt_index],
                    k=desired_topk,
                )

            candidate_scores = metric_for_gt[candidate_indices]

            if candidate_scores.sum() <= 0:
                candidate_indices = self._nearest_anchor_indices(
                    anchor_points=anchor_points,
                    gt_box=gt_boxes[gt_index],
                    k=desired_topk,
                )
                candidate_scores = metric_for_gt[candidate_indices]

            k = min(desired_topk, candidate_indices.numel())

            if k <= 0:
                continue

            _, local_top_indices = candidate_scores.topk(k=k, largest=True)
            top_anchor_indices = candidate_indices[local_top_indices]

            selected_mask[top_anchor_indices, gt_index] = True

        return selected_mask

    def _nearest_anchor_indices(self, anchor_points, gt_box, k):
        center = (gt_box[:2] + gt_box[2:]) * 0.5

        distances = ((anchor_points - center[None, :]) ** 2).sum(dim=1)

        k = min(k, distances.numel())

        _, indices = distances.topk(k=k, largest=False)

        return indices

    def _get_gt_sizes(self, gt_boxes):
        gt_xywh = box_xyxy_to_xywh(gt_boxes)
        return torch.max(gt_xywh[:, 2], gt_xywh[:, 3])

    def _small_weight(self, size, dtype, device):
        if size <= self.tiny_size:
            return torch.tensor(3.0, dtype=dtype, device=device)

        if size <= self.small_size:
            return torch.tensor(2.0, dtype=dtype, device=device)

        return torch.tensor(1.0, dtype=dtype, device=device)

    def _get_image_targets(self, targets, batch_index, device):
        if isinstance(targets, (list, tuple)):
            image_targets = targets[batch_index]
        else:
            image_targets = targets[batch_index]

        image_targets = image_targets.to(device)

        if image_targets.numel() == 0:
            return image_targets.reshape(0, 5)

        if image_targets.dim() == 1:
            image_targets = image_targets.unsqueeze(0)

        valid_mask = image_targets[:, 0] >= 0

        return image_targets[valid_mask]

    def _empty_result(self, device, dtype):
        return {
            "anchor_indices": torch.zeros(0, dtype=torch.long, device=device),
            "gt_indices": torch.zeros(0, dtype=torch.long, device=device),
            "scores": torch.zeros(0, dtype=dtype, device=device),
            "small_weights": torch.zeros(0, dtype=dtype, device=device),
        }