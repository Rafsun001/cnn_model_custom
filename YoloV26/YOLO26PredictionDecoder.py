import torch

from BoxUtils import dist2bbox, make_anchors
from YoloV26Config import NUM_CLASSES, NUM_OUTPUTS, STRIDES, MAX_DET


class YOLO26PredictionDecoder:
    """
    YOLO26-style DFL-free prediction decoder.

    It supports raw prediction formats:

    1. List format:
        [p3_pred, p4_pred, p5_pred]

    2. Dictionary format:
        {
            "one2many": [p3_pred, p4_pred, p5_pred],
            "one2one":  [p3_pred, p4_pred, p5_pred],
        }

    Each prediction tensor:
        [B, 4 + NUM_CLASSES, H, W]

    Final decoded output:
        boxes  = [B, K, 4]
        scores = [B, K]
        labels = [B, K]

    Detection output:
        detections = [B, K, 6]
        [x1, y1, x2, y2, score, class_id]
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        strides=STRIDES,
        topk=MAX_DET,
    ):
        self.num_classes = num_classes
        self.strides = strides
        self.topk = topk
        self.num_outputs = 4 + num_classes

    def decode_feature(self, prediction, stride):
        """
        Decode one feature level.

        Input:
            prediction = [B, 4 + num_classes, H, W]

        Output:
            boxes_xyxy   = [B, H*W, 4]
            scores       = [B, H*W]
            labels       = [B, H*W]
            class_logits = [B, H*W, num_classes]
        """

        batch_size, channels, height, width = prediction.shape

        expected_channels = 4 + self.num_classes
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} prediction channels, "
                f"but got {channels}."
            )

        prediction = prediction.view(
            batch_size,
            channels,
            height * width,
        )

        box_prediction = prediction[:, :4, :]
        class_logits = prediction[:, 4:4 + self.num_classes, :]

        anchor_points, stride_tensor = make_anchors(
            feature_shapes=[(height, width)],
            strides=[stride],
            device=prediction.device,
            dtype=prediction.dtype,
            grid_cell_offset=0.5,
        )

        box_prediction = box_prediction.permute(0, 2, 1).contiguous()
        boxes_xyxy = dist2bbox(
            distance=box_prediction,
            anchor_points=anchor_points,
            xywh=False,
        )

        boxes_xyxy = boxes_xyxy * stride_tensor.unsqueeze(0)

        class_logits = class_logits.permute(0, 2, 1).contiguous()
        class_scores = class_logits.sigmoid()

        scores, labels = class_scores.max(dim=-1)

        return boxes_xyxy, scores, labels, class_logits

    def decode(self, predictions, branch="one2one"):
        """
        Decode P3/P4/P5 raw predictions.

        If predictions is a dictionary, this selects:
            predictions[branch]

        branch:
            "one2one"  -> NMS-free inference branch
            "one2many" -> dense training/debug branch
        """

        predictions = self._select_prediction_branch(
            predictions=predictions,
            branch=branch,
        )

        decoded_boxes = []
        decoded_scores = []
        decoded_labels = []

        for prediction, stride in zip(predictions, self.strides):
            boxes, scores, labels, _ = self.decode_feature(
                prediction=prediction,
                stride=stride,
            )

            decoded_boxes.append(boxes)
            decoded_scores.append(scores)
            decoded_labels.append(labels)

        boxes = torch.cat(decoded_boxes, dim=1)
        scores = torch.cat(decoded_scores, dim=1)
        labels = torch.cat(decoded_labels, dim=1)

        if self.topk is not None and scores.size(1) > self.topk:
            top_scores, indices = scores.topk(
                k=self.topk,
                dim=1,
            )

            boxes = boxes.gather(
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, 4),
            )

            labels = labels.gather(
                dim=1,
                index=indices,
            )

            scores = top_scores

        return boxes, scores, labels

    def decode_to_detections(self, predictions, branch="one2one"):
        """
        Returns:
            detections = [B, K, 6]

        Last dimension:
            x1, y1, x2, y2, score, class_id
        """

        boxes, scores, labels = self.decode(
            predictions=predictions,
            branch=branch,
        )

        detections = torch.cat(
            [
                boxes,
                scores.unsqueeze(-1),
                labels.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        return detections

    def postprocess(self, boxes, class_scores, max_det=MAX_DET):
        """
        Official-style top-k postprocess without NMS.

        boxes:
            [B, N, 4]

        class_scores:
            [B, N, num_classes]

        output:
            [B, max_det, 6]
        """

        batch_size, num_anchors, _ = boxes.shape

        first_topk = min(max_det, num_anchors)

        _, top_anchor_indices = class_scores.amax(dim=-1).topk(
            k=first_topk,
            dim=1,
        )

        boxes = boxes.gather(
            dim=1,
            index=top_anchor_indices.unsqueeze(-1).expand(-1, -1, 4),
        )

        class_scores = class_scores.gather(
            dim=1,
            index=top_anchor_indices.unsqueeze(-1).expand(
                -1,
                -1,
                self.num_classes,
            ),
        )

        flat_scores = class_scores.flatten(start_dim=1)

        second_topk = min(max_det, flat_scores.shape[1])

        final_scores, final_indices = flat_scores.topk(
            k=second_topk,
            dim=1,
        )

        final_anchor_indices = final_indices // self.num_classes
        final_class_indices = final_indices % self.num_classes

        final_boxes = boxes.gather(
            dim=1,
            index=final_anchor_indices.unsqueeze(-1).expand(-1, -1, 4),
        )

        detections = torch.cat(
            [
                final_boxes,
                final_scores.unsqueeze(-1),
                final_class_indices.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        return detections

    def _select_prediction_branch(self, predictions, branch):
        if isinstance(predictions, dict):
            if branch not in predictions:
                raise KeyError(
                    f"Prediction dictionary does not contain branch '{branch}'. "
                    f"Available keys: {list(predictions.keys())}"
                )

            return predictions[branch]

        return predictions