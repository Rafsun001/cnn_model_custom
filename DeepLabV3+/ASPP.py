import torch
import torch.nn as nn
from .ASPPBranch1x1Conv import ASPPBranch1x1Conv
from .ASPPBranchAtrousRate12 import ASPPBranchAtrousRate12
from .ASPPBranchAtrousRate18 import ASPPBranchAtrousRate18
from .ASPPBranchAtrousRate6 import ASPPBranchAtrousRate6
from .ASPPImagePoolingBranch import ASPPImagePoolingBranch
from .ASPPProjectionConv import ASPPProjectionConv


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1_1x1 = ASPPBranch1x1Conv()
        self.branch2_rate6 = ASPPBranchAtrousRate6()
        self.branch3_rate12 = ASPPBranchAtrousRate12()
        self.branch4_rate18 = ASPPBranchAtrousRate18()
        self.branch5_image_pooling = ASPPImagePoolingBranch()

        self.projection = ASPPProjectionConv()

    def forward(self, high_level_feature):
        branch1_output = self.branch1_1x1(high_level_feature)
        branch2_output = self.branch2_rate6(high_level_feature)
        branch3_output = self.branch3_rate12(high_level_feature)
        branch4_output = self.branch4_rate18(high_level_feature)
        branch5_output = self.branch5_image_pooling(high_level_feature)

        concatenated_output = torch.cat(
            [
                branch1_output,
                branch2_output,
                branch3_output,
                branch4_output,
                branch5_output,
            ],
            dim=1,
        )

        encoder_output = self.projection(concatenated_output)

        return encoder_output
