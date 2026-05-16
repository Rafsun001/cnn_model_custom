import torch

from EfficientNetV2SModel import EfficientNetV2SModel


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def capture_output_shapes(model, module_names, x):
    shapes = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            shapes[name] = tuple(output.shape)

        return hook

    final_block = model.model
    for name in module_names:
        hooks.append(getattr(final_block, name).register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            y = model(x)
    finally:
        for hook in hooks:
            hook.remove()

    return y, shapes


def main():
    model = EfficientNetV2SModel(
        num_classes=200,
        dropout_rate=0.2,
        drop_path_rate=0.1,
    )
    model.eval()

    x = torch.randn(2, 3, 64, 64)

    expected_shapes = {
        "input_block": (2, 3, 64, 64),
        "stem_block": (2, 24, 32, 32),
        "stage1": (2, 24, 32, 32),
        "stage2": (2, 48, 16, 16),
        "stage3": (2, 64, 8, 8),
        "stage4": (2, 128, 4, 4),
        "stage5": (2, 160, 4, 4),
        "stage6": (2, 256, 2, 2),
        "head_block": (2, 1280, 2, 2),
        "pooling_block": (2, 1280, 1, 1),
        "classifier_block": (2, 200),
    }

    y, shapes = capture_output_shapes(
        model=model,
        module_names=list(expected_shapes.keys()),
        x=x,
    )

    print("=" * 70)
    print("EfficientNetV2-S architecture verification")
    print("=" * 70)
    print(f"Input shape:  {tuple(x.shape)}")
    for name, shape in shapes.items():
        print(f"{name:16s}: {shape}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Parameters:   {count_parameters(model):,}")
    print("=" * 70)

    assert y.shape == (2, 200), "Output shape should be (batch_size, 200)."
    assert shapes == expected_shapes, "Intermediate shapes do not match EfficientNetV2-S."

    print("OK architecture forward pass works.")


if __name__ == "__main__":
    main()
