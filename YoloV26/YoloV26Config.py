NUM_CLASSES = 80

# YOLO26 is DFL-free, so reg_max = 1.
# Output channels = 4 box values + class logits.
REG_MAX = 1
NUM_OUTPUTS = NUM_CLASSES + (4 * REG_MAX)

# End-to-end / NMS-free style setting.
END2END = True

# P3, P4, P5 strides.
STRIDES = (8, 16, 32)

# Official-style max detections after one-to-one top-k postprocess.
MAX_DET = 300

# Input image size.
IMAGE_SIZE = 640

# Model scale: nano-style educational version.
MODEL_SCALE = "n"

# YOLO26n scale values.
DEPTH_MULTIPLE = 0.50
WIDTH_MULTIPLE = 0.25
MAX_CHANNELS = 1024