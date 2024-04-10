"""
YOLOPhiNet training configuration.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""

# Data configuration
batch_size = 8
data_cfg = "cfg/data/VOC.yaml"
data_dir = "datasets/coco"
epochs = 350
num_classes = 80

# Model configuration
input_shape = [3, 640, 640]
alpha = 1.1
num_layers = 8
beta = 0.75
t_zero = 5
divisor = 8
downsampling_layers = [5, 7]
heads = [True, True, True]

# just returning 2 intermediate layers (last is default)
return_layers = [6, 8]

# Placeholder for inference
ckpt_pretrained = ""
output_dir = "detection_output"
coco_names = "cfg/data/coco.names"
