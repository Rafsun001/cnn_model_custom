import torch
import torch.nn as nn

from HeadDetectP3 import HeadDetectP3
from HeadDetectP4 import HeadDetectP4
from HeadDetectP5 import HeadDetectP5
from YoloV26Config import NUM_CLASSES, NUM_OUTPUTS, MAX_DET, STRIDES


class HeadDetect1(nn.Module):
    """
    YOLO26-style Detect module split into educational files.

    Training output:
        {
            "one2many": [p3_pred, p4_pred, p5_pred],
            "one2one":  [p3_pred, p4_pred, p5_pred],
        }

    Each prediction:
        [B, 84, H, W]

    Eval output:
        detections = [B, 300, 6]

    Detection format:
        [x1, y1, x2, y2, score, class_id]
    """

    def __init__(self):
        super().__init__()

        self.nc = NUM_CLASSES
        self.no = NUM_OUTPUTS
        self.max_det = MAX_DET
        self.strides = STRIDES

        self.detect_p3 = HeadDetectP3()
        self.detect_p4 = HeadDetectP4()
        self.detect_p5 = HeadDetectP5()

    def forward(self, p3, p4, p5):
        p3_one2many, p3_one2one = self.detect_p3(p3)
        p4_one2many, p4_one2one = self.detect_p4(p4)
        p5_one2many, p5_one2one = self.detect_p5(p5)

        predictions = {
            "one2many": [
                p3_one2many,
                p4_one2many,
                p5_one2many,
            ],
            "one2one": [
                p3_one2one,
                p4_one2one,
                p5_one2one,
            ],
        }

        if self.training:
            return predictions

        detections = self._inference(predictions["one2one"])

        return detections

    def _inference(self, predictions):
        """
        Decode one-to-one predictions and apply top-k postprocess.

        Input:
            predictions = [P3, P4, P5]

        Output:
            detections = [B, max_det, 6]
        """

        boxes_per_level = []
        scores_per_level = []
        anchor_points = []
        stride_values = []

        for prediction, stride in zip(predictions, self.strides):
            batch_size, channels, height, width = prediction.shape

            if channels != self.no:
                raise ValueError(
                    f"Expected prediction channels {self.no}, "
                    f"but got {channels}."
                )

            prediction = prediction.view(
                batch_size,
                self.no,
                height * width,
            )

            box_prediction = prediction[:, :4, :]
            class_prediction = prediction[:, 4:4 + self.nc, :]

            boxes_per_level.append(box_prediction)
            scores_per_level.append(class_prediction)

            level_anchor_points, level_stride_values = self._make_anchors(
                height=height,
                width=width,
                stride=stride,
                device=prediction.device,
                dtype=prediction.dtype,
            )

            anchor_points.append(level_anchor_points)
            stride_values.append(level_stride_values)

        box_predictions = torch.cat(boxes_per_level, dim=2)
        class_predictions = torch.cat(scores_per_level, dim=2)

        anchor_points = torch.cat(anchor_points, dim=0)
        stride_values = torch.cat(stride_values, dim=0)

        decoded_boxes = self._decode_bboxes(
            box_predictions=box_predictions,
            anchor_points=anchor_points,
            stride_values=stride_values,
        )

        class_scores = class_predictions.sigmoid().permute(0, 2, 1).contiguous()

        detections = self._postprocess(
            boxes=decoded_boxes,
            scores=class_scores,
            max_det=self.max_det,
        )

        return detections

    def _make_anchors(self, height, width, stride, device, dtype):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        anchor_points = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        ).to(dtype=dtype)

        anchor_points = anchor_points.reshape(-1, 2)
        anchor_points = anchor_points + 0.5

        stride_values = torch.full(
            size=(height * width, 1),
            fill_value=float(stride),
            device=device,
            dtype=dtype,
        )

        return anchor_points, stride_values

    def _decode_bboxes(self, box_predictions, anchor_points, stride_values):
        """
        DFL-free reg_max=1 decoding.

        box_predictions:
            [B, 4, N]

        anchor_points:
            [N, 2]

        stride_values:
            [N, 1]

        Output:
            boxes_xyxy = [B, N, 4]
        """

        box_predictions = box_predictions.permute(0, 2, 1).contiguous()

        anchor_points = anchor_points.unsqueeze(0)
        stride_values = stride_values.unsqueeze(0)

        left_top = anchor_points - box_predictions[..., 0:2]
        right_bottom = anchor_points + box_predictions[..., 2:4]

        boxes_xyxy = torch.cat([left_top, right_bottom], dim=-1)
        boxes_xyxy = boxes_xyxy * stride_values

        return boxes_xyxy

    def _postprocess(self, boxes, scores, max_det):
        """
        Official-style top-k postprocess without NMS.

        boxes:
            [B, N, 4]

        scores:
            [B, N, num_classes]

        output:
            [B, max_det, 6]
        """

        batch_size, num_anchors, _ = boxes.shape

        first_topk = min(max_det, num_anchors)

        top_anchor_scores, top_anchor_indices = scores.amax(dim=-1).topk(
            first_topk,
            dim=1,
        )

        boxes = boxes.gather(
            dim=1,
            index=top_anchor_indices.unsqueeze(-1).expand(-1, -1, 4),
        )

        scores = scores.gather(
            dim=1,
            index=top_anchor_indices.unsqueeze(-1).expand(-1, -1, self.nc),
        )

        flattened_scores = scores.flatten(start_dim=1)

        second_topk = min(max_det, flattened_scores.shape[1])

        final_scores, final_indices = flattened_scores.topk(
            second_topk,
            dim=1,
        )

        final_anchor_indices = final_indices // self.nc
        final_class_indices = final_indices % self.nc

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

    def bias_init(self):
        self.detect_p3.bias_init(stride=self.strides[0])
        self.detect_p4.bias_init(stride=self.strides[1])
        self.detect_p5.bias_init(stride=self.strides[2])

    def fuse(self):
        self.detect_p3.fuse()
        self.detect_p4.fuse()
        self.detect_p5.fuse()