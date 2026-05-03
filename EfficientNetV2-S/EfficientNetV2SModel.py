import torch.nn as nn

from EfficientNetV2SFinalBlock import EfficientNetV2SFinalBlock
class EfficientNetV2SModel(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.2, drop_path_rate=0.2):
        super().__init__()

        self.model = EfficientNetV2SFinalBlock(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.model(x)
        return x
