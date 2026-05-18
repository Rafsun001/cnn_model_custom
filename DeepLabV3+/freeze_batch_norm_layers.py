import torch.nn as nn


def freeze_batch_norm_layers(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
