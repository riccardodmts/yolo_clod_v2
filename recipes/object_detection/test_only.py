"""
YOLO training.

This code allows you to train an object detection model with the YOLOv8 neck and loss.

To run this script, you can start it with:
    python train.py cfg/yolo_phinet.py

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""

import torch
from prepare_data import create_loaders
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from yolo_loss import Loss

import micromind as mm
from micromind.networks import PhiNet
from micromind.networks.yolo import SPPF, DetectionHead, Yolov8Neck, Yolov8NeckOpt
from micromind.utils import parse_configuration
from micromind.utils.yolo import (
    load_config,
    mean_average_precision,
    postprocess,
)
from micromind.networks.yolo import YOLOv8
import sys
import os
from micromind.utils.yolo import get_variant_multiples


class YOLO(mm.MicroMind):
    def __init__(self, m_cfg, hparams, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(*args, **kwargs)
        self.m_cfg = m_cfg
        w, r, d = get_variant_multiples("n")
        self.modules["yolov8"] = YOLOv8(w,r,d, 80)
        self.modules["yolov8"].load_state_dict(torch.load("usable_yolov8n.pt"))

        self.criterion = Loss(self.m_cfg, self.modules["yolov8"].head, self.device)

        print("Number of parameters for each module:")
        print(self.compute_params())

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        preprocessed_batch = {}
        preprocessed_batch["img"] = (
            batch["img"].to(self.device, non_blocking=True).float() / 255
            #batch["img"].to(self.device, non_blocking=True).float() *0 +1
        )
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)

        return preprocessed_batch

    def forward(self, batch):
        """Runs the forward method by calling every module."""
        preprocessed_batch = self.preprocess_batch(batch)

        return self.modules["yolov8"](preprocessed_batch["img"].to(self.device))

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
        )

        return lossi_sum

    def configure_optimizers(self):
        """Configures the optimizer and the scheduler."""
        opt = torch.optim.SGD(self.modules.parameters(), lr=1e-2, weight_decay=0.0005)
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt, T_max=14000, eta_min=1e-3
        # )
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, factor=0.7, patience=5, threshold=0.1
        # )
        return opt#, sched

    @torch.no_grad()
    def mAP(self, pred, batch):
        """Compute the mean average precision (mAP) for a batch of predictions.

        Arguments
        ---------
        pred : torch.Tensor
            Model predictions for the batch.
        batch : dict
            A dictionary containing batch information, including bounding boxes,
            classes and shapes.

        Returns
        -------
        torch.Tensor
            A tensor containing the computed mean average precision (mAP) for the batch.
        """
        preprocessed_batch = self.preprocess_batch(batch)
        post_predictions = postprocess(
            preds=pred[0], img=preprocessed_batch, orig_imgs=batch
        )

        batch_bboxes_xyxy = xywh2xyxy(batch["bboxes"])
        dim = batch["resized_shape"][0][0]
        batch_bboxes_xyxy[:, :4] *= dim

        batch_bboxes = []
        for i in range(len(batch["batch_idx"])):
            for b in range(len(batch_bboxes_xyxy[batch["batch_idx"] == i, :])):
                batch_bboxes.append(
                    scale_boxes(
                        batch["resized_shape"][i],
                        batch_bboxes_xyxy[batch["batch_idx"] == i, :][b],
                        batch["ori_shape"][i],
                    )
                )

        batch_bboxes = torch.stack(batch_bboxes).to(self.device)
        mmAP = mean_average_precision(
            post_predictions, batch, batch_bboxes, data_cfg["nc"]
        )

        return torch.Tensor([mmAP])


def replace_datafolder(hparams, data_cfg):
    """Replaces the data root folder, if told to do so from the configuration."""
    data_cfg["path"] = str(data_cfg["path"])
    data_cfg["path"] = (
        data_cfg["path"][:-1] if data_cfg["path"][-1] == "/" else data_cfg["path"]
    )
    for key in ["train", "val"]:
        if not isinstance(data_cfg[key], list):
            data_cfg[key] = [data_cfg[key]]
        new_list = []
        for tmp in data_cfg[key]:
            if hasattr(hparams, "data_dir"):
                if hparams.data_dir != data_cfg["path"]:
                    tmp = str(tmp).replace(data_cfg["path"], "")
                    tmp = tmp[1:] if tmp[0] == "/" else tmp
                    tmp = os.path.join(hparams.data_dir, tmp)
                    new_list.append(tmp)
        data_cfg[key] = new_list

    data_cfg["path"] = hparams.data_dir

    return data_cfg


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])
    if len(hparams.input_shape) != 3:
        hparams.input_shape = [
            int(x) for x in "".join(hparams.input_shape).split(",")
        ]  # temp solution
        print(f"Setting input shape to {hparams.input_shape}.")

    m_cfg, data_cfg = load_config(hparams.data_cfg)

    # check if specified path for images is different, correct it in case
    data_cfg = replace_datafolder(hparams, data_cfg)
    m_cfg.imgsz = hparams.input_shape[-1]  # temp solution

    train_loader, val_loader = create_loaders(m_cfg, data_cfg, hparams.batch_size)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(
        exp_folder, hparams=hparams, key="loss"
    )

    yolo_mind = YOLO(m_cfg, hparams=hparams)

    mAP = mm.Metric("mAP", yolo_mind.mAP, eval_only=True, eval_period=1)

    yolo_mind.train(
        epochs=hparams.epochs,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[mAP],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )
    # yolo_mind.test(
    #     datasets={"test": val_loader},
    #     metrics=[mAP],
    # )
