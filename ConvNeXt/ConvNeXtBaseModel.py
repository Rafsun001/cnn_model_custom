import torch.nn as nn

from ConvNeXtBaseFinalBlock import ConvNeXtBaseFinalBlock


class ConvNeXtBaseModel(nn.Module):
    def __init__(self, num_classes=1000, drop_path_rate=0.5, layer_scale_init_value=1e-6):
        super().__init__()

        self.model = ConvNeXtBaseFinalBlock(
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )

    def forward(self, x):
        x = self.model(x)
        return x
