import torch

from YOLO26DetectionLoss import YOLO26DetectionLoss
from YOLO26ExplicitModel import YOLO26ExplicitModel
from YOLO26PredictionDecoder import YOLO26PredictionDecoder


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    x = torch.randn(1, 3, 640, 640)

    model = YOLO26ExplicitModel()
    model.bias_init()

    model.train()
    train_output = model(x)

    expected_prediction_shapes = [
        (1, 84, 80, 80),
        (1, 84, 40, 40),
        (1, 84, 20, 20),
    ]

    one2many_shapes = [tuple(prediction.shape) for prediction in train_output["one2many"]]
    one2one_shapes = [tuple(prediction.shape) for prediction in train_output["one2one"]]

    decoder = YOLO26PredictionDecoder()
    boxes, scores, labels = decoder.decode(train_output, branch="one2one")
    detections_from_decoder = decoder.decode_to_detections(train_output, branch="one2one")

    targets = [
        torch.tensor(
            [[0.0, 100.0, 120.0, 180.0, 220.0]],
            dtype=torch.float32,
        )
    ]
    loss_output = YOLO26DetectionLoss()(train_output, targets, epoch=1, total_epochs=100)

    model.eval()
    eval_output = model(x)
    raw_eval_output = model(x, raw=True)

    model.fuse()
    fused_eval_output = model(x)
    fused_raw_output = model(x, raw=True)

    print("=" * 70)
    print("YOLO26 architecture verification")
    print("=" * 70)
    print(f"Input shape:          {tuple(x.shape)}")
    print(f"One2many shapes:      {one2many_shapes}")
    print(f"One2one shapes:       {one2one_shapes}")
    print(f"Eval detections:      {tuple(eval_output.shape)}")
    print(f"Raw eval one2one:     {[tuple(t.shape) for t in raw_eval_output['one2one']]}")
    print(f"Fused detections:     {tuple(fused_eval_output.shape)}")
    print(f"Fused raw one2one:    {[tuple(t.shape) for t in fused_raw_output['one2one']]}")
    print(f"Decoded boxes:        {tuple(boxes.shape)}")
    print(f"Decoded scores:       {tuple(scores.shape)}")
    print(f"Decoded labels:       {tuple(labels.shape)}")
    print(f"Decoded detections:   {tuple(detections_from_decoder.shape)}")
    print(f"Loss value:           {float(loss_output['loss'].detach()):.6f}")
    print(f"Parameters:           {count_parameters(model):,}")
    print("=" * 70)

    assert one2many_shapes == expected_prediction_shapes
    assert one2one_shapes == expected_prediction_shapes
    assert tuple(eval_output.shape) == (1, 300, 6)
    assert [tuple(t.shape) for t in raw_eval_output["one2one"]] == expected_prediction_shapes
    assert tuple(fused_eval_output.shape) == (1, 300, 6)
    assert [tuple(t.shape) for t in fused_raw_output["one2one"]] == expected_prediction_shapes
    assert tuple(boxes.shape) == (1, 300, 4)
    assert tuple(scores.shape) == (1, 300)
    assert tuple(labels.shape) == (1, 300)
    assert tuple(detections_from_decoder.shape) == (1, 300, 6)
    assert torch.isfinite(loss_output["loss"])

    print("OK architecture forward, decode, loss, and fused inference work.")


if __name__ == "__main__":
    main()
