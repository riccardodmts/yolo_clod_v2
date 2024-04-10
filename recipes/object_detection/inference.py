"""
YOLOv8 inference.

This code allows you to launch an object detection inference using a YOLO MicroMind.

To run this script, you should pass the checkpoint with the weights and the path to
an image, following this example:
    python inference.py cfg/yolo_phinet.py IMG_PATH --ckpt_pretrained CHECKPOINT_PATH

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023

"""

import sys
import time
from pathlib import Path

import torch
import torchvision

from micromind.utils import parse_configuration
from micromind.utils.yolo import (
    draw_bounding_boxes_and_save,
    postprocess,
    preprocess,
)
from train import YOLO
from micromind.utils.yolo import load_config


class Inference(YOLO):
    def __init__(self, m_cfg, hparams):
        super().__init__(m_cfg, hparams=hparams)

    def forward(self, batch):
        """Runs the forward method by calling every module."""
        backbone = self.modules["backbone"](batch)
        neck_input = backbone[1]
        neck_input.append(self.modules["sppf"](backbone[0]))
        neck = self.modules["neck"](*neck_input)
        head = self.modules["head"](neck)
        return head


if __name__ == "__main__":
    assert len(sys.argv) > 2, " ".join(
        "Something went wrong when launching the script. \
            Please check the arguments.".split(
            " "
        )
    )

    hparams = parse_configuration(sys.argv[1])
    if len(hparams.input_shape) != 3:
        hparams.input_shape = [
            int(x) for x in "".join(hparams.input_shape).split(",")
        ]  # temp solution
        print(f"Setting input shape to {hparams.input_shape}.")

    output_folder_path = Path(hparams.output_dir)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    img_paths = [sys.argv[2]]
    for img_path in img_paths:
        image = torchvision.io.read_image(img_path)
        if image.shape[0] == 4:
            image = image[:3, :, :]  # Mantieni solo i primi 3 canali (RGB)
        out_paths = [
            (
                output_folder_path
                / f"{Path(img_path).stem}_output{Path(img_path).suffix}"
            ).as_posix()
        ]
        if not isinstance(image, torch.Tensor):
            print("Error in image loading. Check your image file.")
            sys.exit(1)

        pre_processed_image = preprocess(image)

        m_cfg, data_cfg = load_config(hparams.data_cfg)
        model = Inference(m_cfg, hparams=hparams)
        # Load pretrained if passed.
        if hparams.ckpt_pretrained != "":
            model.load_modules(hparams.ckpt_pretrained)
            print(f"Pretrained model loaded from {hparams.ckpt_pretrained}.")
        else:
            print("Running inference with no weights.")

        model.eval()

        with torch.no_grad():
            st = time.time()
            predictions = model.forward(pre_processed_image)
            print(f"Inference took {int(round(((time.time() - st) * 1000)))}ms")
            breakpoint()
            post_predictions = postprocess(
                preds=predictions[0], img=pre_processed_image, orig_imgs=image
            )
            breakpoint()

        class_labels = [s.strip() for s in open(hparams.coco_names, "r").readlines()]
        draw_bounding_boxes_and_save(
            orig_img_paths=img_paths,
            output_img_paths=out_paths,
            all_predictions=post_predictions,
            class_labels=class_labels,
        )

        # Exporting onnx model.
