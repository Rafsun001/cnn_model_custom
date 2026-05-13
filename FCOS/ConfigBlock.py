import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfigBlock:

    LEVEL_NAMES = ["P3", "P4", "P5", "P6", "P7"]
    STRIDES = {
        "P3": 8,
        "P4": 16,
        "P5": 32,
        "P6": 64,
        "P7": 128,
    }
    REGRESSION_RANGES = {
        "P3": (0.0, 64.0),
        "P4": (64.0, 128.0),
        "P5": (128.0, 256.0),
        "P6": (256.0, 512.0),
        "P7": (512.0, float("inf")),
    }
