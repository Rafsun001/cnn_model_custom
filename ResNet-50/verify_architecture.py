import torch

from model import ResNet50


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def capture_output_shapes(model, module_names, x):
    shapes = {}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            shapes[name] = tuple(output.shape)

        return hook

    for name in module_names:
        hooks.append(getattr(model, name).register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            y = model(x)
    finally:
        for hook in hooks:
            hook.remove()

    return y, shapes


def main():
    model = ResNet50(num_classes=200)
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    expected_shapes = {
        "conv1": (2, 64, 112, 112),
        "maxpool": (2, 64, 56, 56),
        "stage1": (2, 256, 56, 56),
        "stage2": (2, 512, 28, 28),
        "stage3": (2, 1024, 14, 14),
        "stage4": (2, 2048, 7, 7),
        "avgpool": (2, 2048, 1, 1),
    }

    y, shapes = capture_output_shapes(
        model=model,
        module_names=list(expected_shapes.keys()),
        x=x,
    )

    print("=" * 70)
    print("Standard ResNet-50 architecture verification")
    print("=" * 70)
    print(f"Input shape:  {tuple(x.shape)}")
    for name, shape in shapes.items():
        print(f"{name:10s}: {shape}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Parameters:   {count_parameters(model):,}")
    print("=" * 70)

    assert y.shape == (2, 200), "Output shape should be (batch_size, 200)."
    assert shapes == expected_shapes, "Intermediate shapes do not match standard ResNet-50."

    print("OK architecture forward pass works.")


if __name__ == "__main__":
    main()
