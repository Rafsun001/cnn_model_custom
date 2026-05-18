import torch.nn as nn
import torch.nn.functional as F
from .ASPP import ASPP
from .DeepLabV3PlusDecoder import DeepLabV3PlusDecoder
from .ResNet50BackboneForDeepLabV3Plus import ResNet50BackboneForDeepLabV3Plus


class DeepLabV3PlusResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = ResNet50BackboneForDeepLabV3Plus()
        self.aspp = ASPP()
        self.decoder = DeepLabV3PlusDecoder(num_classes=num_classes)

    def forward(self, images):
        input_spatial_size = images.shape[2:]

        low_level_feature, high_level_feature = self.backbone(images)

        encoder_output = self.aspp(high_level_feature)

        decoder_output = self.decoder(
            encoder_output=encoder_output,
            low_level_feature=low_level_feature,
        )

        final_output = F.interpolate(
            decoder_output,
            size=input_spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        return final_output
