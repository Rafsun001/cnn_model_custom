import torch

from ConvNeXtBaseModel import ConvNeXtBaseModel


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
    model = ConvNeXtBaseModel(
        num_classes=200,
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
    )
    model.eval()

    x = torch.randn(2, 3, 64, 64)

    expected_shapes = {
        "input_block": (2, 3, 64, 64),
        "stem": (2, 128, 16, 16),
        "stage1": (2, 128, 16, 16),
        "downsample1": (2, 256, 8, 8),
        "stage2": (2, 256, 8, 8),
        "downsample2": (2, 512, 4, 4),
        "stage3": (2, 512, 4, 4),
        "downsample3": (2, 1024, 2, 2),
        "stage4": (2, 1024, 2, 2),
        "global_pool": (2, 1024),
        "classifier": (2, 200),
    }

    y, shapes = capture_output_shapes(
        model=model,
        module_names=list(expected_shapes.keys()),
        x=x,
    )

    print("=" * 70)
    print("ConvNeXt-Base architecture verification")
    print("=" * 70)
    print(f"Input shape:  {tuple(x.shape)}")
    for name, shape in shapes.items():
        print(f"{name:12s}: {shape}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Parameters:   {count_parameters(model):,}")
    print("=" * 70)

    assert y.shape == (2, 200), "Output shape should be (batch_size, 200)."
    assert shapes == expected_shapes, "Intermediate shapes do not match ConvNeXt-Base."

    print("OK architecture forward pass works.")


if __name__ == "__main__":
    main()
