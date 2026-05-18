import torch
import torch.nn as nn
import torch.nn.functional as F
from .DecoderClassifier import DecoderClassifier
from .DecoderConvBlock1 import DecoderConvBlock1
from .DecoderConvBlock2 import DecoderConvBlock2
from .DecoderLowLevelProjection import DecoderLowLevelProjection


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.low_level_projection = DecoderLowLevelProjection()
        self.decoder_conv_block1 = DecoderConvBlock1()
        self.decoder_conv_block2 = DecoderConvBlock2()
        self.classifier = DecoderClassifier(num_classes=num_classes)

    def forward(self, encoder_output, low_level_feature):
        processed_low_level_feature = self.low_level_projection(low_level_feature)

        upsampled_encoder_output = F.interpolate(
            encoder_output,
            size=processed_low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        concatenated_feature = torch.cat(
            [
                upsampled_encoder_output,
                processed_low_level_feature,
            ],
            dim=1,
        )

        x = self.decoder_conv_block1(concatenated_feature)
        x = self.decoder_conv_block2(x)
        x = self.classifier(x)

        return x
