import torch


def box_xywh_to_xyxy(boxes):
    """
    Convert boxes from center format to corner format.

    Input:
        boxes = [..., 4]
        format = [cx, cy, w, h]

    Output:
        boxes = [..., 4]
        format = [x1, y1, x2, y2]
    """

    half_size = boxes[..., 2:4] * 0.5

    top_left = boxes[..., 0:2] - half_size
    bottom_right = boxes[..., 0:2] + half_size

    return torch.cat([top_left, bottom_right], dim=-1)


def box_xyxy_to_xywh(boxes):
    """
    Convert boxes from corner format to center format.

    Input:
        boxes = [..., 4]
        format = [x1, y1, x2, y2]

    Output:
        boxes = [..., 4]
        format = [cx, cy, w, h]
    """

    center = (boxes[..., 0:2] + boxes[..., 2:4]) * 0.5
    size = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0.0)

    return torch.cat([center, size], dim=-1)


def box_area(boxes):
    """
    Calculate box area.

    boxes format:
        [x1, y1, x2, y2]
    """

    width = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)

    return width * height


def box_iou(boxes1, boxes2):
    """
    Pairwise IoU.

    boxes1:
        [N, 4]

    boxes2:
        [M, 4]

    Output:
        iou = [N, M]
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    right_bottom = torch.min(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = (right_bottom - left_top).clamp(min=0.0)

    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=1e-7)


def bbox_iou(boxes1, boxes2, eps=1e-7):
    """
    Element-wise IoU.

    boxes1:
        [N, 4]

    boxes2:
        [N, 4]

    Output:
        iou = [N]
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (right_bottom - left_top).clamp(min=0.0)

    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter

    return inter / union.clamp(min=eps)


def complete_iou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU loss.

    pred_boxes:
        [N, 4], xyxy

    target_boxes:
        [N, 4], xyxy

    Output:
        loss = [N]
    """

    pred_area = box_area(pred_boxes)
    target_area = box_area(target_boxes)

    inter_left_top = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    inter_right_bottom = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])

    inter_wh = (inter_right_bottom - inter_left_top).clamp(min=0.0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    union = pred_area + target_area - inter_area
    iou = inter_area / union.clamp(min=eps)

    pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) * 0.5
    target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) * 0.5

    center_distance = ((pred_center - target_center) ** 2).sum(dim=1)

    enclosing_left_top = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
    enclosing_right_bottom = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])

    enclosing_wh = (enclosing_right_bottom - enclosing_left_top).clamp(min=0.0)
    diagonal_distance = (enclosing_wh ** 2).sum(dim=1).clamp(min=eps)

    pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp(min=eps)
    target_wh = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=eps)

    v = (4.0 / (torch.pi ** 2)) * (
        torch.atan(target_wh[:, 0] / target_wh[:, 1])
        - torch.atan(pred_wh[:, 0] / pred_wh[:, 1])
    ) ** 2

    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp(min=eps)

    ciou = iou - center_distance / diagonal_distance - alpha * v

    loss = 1.0 - ciou.clamp(min=-1.0, max=1.0)

    return loss


def make_anchors(
    feature_shapes,
    strides,
    device,
    dtype,
    grid_cell_offset=0.5,
):
    """
    Create anchor points for feature maps.

    feature_shapes:
        [(H3, W3), (H4, W4), (H5, W5)]

    strides:
        [8, 16, 32]

    Output:
        anchor_points = [N, 2]
        stride_tensor = [N, 1]

    Anchor points are in grid units, not pixel units.
    """

    anchor_points = []
    stride_tensor = []

    for (height, width), stride in zip(feature_shapes, strides):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        points = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        ).to(dtype=dtype)

        points = points.reshape(-1, 2)
        points = points + grid_cell_offset

        anchor_points.append(points)

        stride_values = torch.full(
            size=(height * width, 1),
            fill_value=float(stride),
            device=device,
            dtype=dtype,
        )

        stride_tensor.append(stride_values)

    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0)

    return anchor_points, stride_tensor


def dist2bbox(distance, anchor_points, xywh=False):
    """
    Convert distance predictions to boxes.

    distance:
        [..., 4]
        [left, top, right, bottom]

    anchor_points:
        [..., 2]
        [x, y]

    Output:
        xyxy or xywh boxes
    """

    left_top = anchor_points - distance[..., 0:2]
    right_bottom = anchor_points + distance[..., 2:4]

    if xywh:
        center = (left_top + right_bottom) * 0.5
        wh = right_bottom - left_top
        return torch.cat([center, wh], dim=-1)

    return torch.cat([left_top, right_bottom], dim=-1)


def bbox2dist(anchor_points, boxes, reg_max):
    """
    Convert boxes to left/top/right/bottom distances from anchor points.

    anchor_points:
        [N, 2]

    boxes:
        [N, 4], xyxy

    Output:
        distance = [N, 4]
    """

    left_top = anchor_points - boxes[..., :2]
    right_bottom = boxes[..., 2:] - anchor_points

    distance = torch.cat([left_top, right_bottom], dim=-1)

    return distance.clamp(min=0.0, max=reg_max - 0.01)