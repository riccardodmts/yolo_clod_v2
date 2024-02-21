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
import torch.nn as nn
import torch.optim as optim
from prepare_data import create_loaders
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from yolo_loss import Loss
import math

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

        self.modules["backbone"] = PhiNet(
            (3, 128, 128),
            alpha=1.1,
            beta=0.75,
            t_zero=5,
            num_layers=8,
            h_swish=False,
            squeeze_excite=True,
            include_top=False,
            num_classes=1000,
            divisor=8,
            compatibility=False,
            downsampling_layers=hparams.downsampling_layers,
            return_layers=hparams.return_layers,
        )

        # PhiNet(
            # input_shape=hparams.input_shape,
            # alpha=hparams.alpha,
            # num_layers=hparams.num_layers,
            # beta=hparams.beta,
            # t_zero=hparams.t_zero,
            # include_top=False,
            # compatibility=False,
            # divisor=hparams.divisor,
            # downsampling_layers=hparams.downsampling_layers,
            # return_layers=hparams.return_layers,
        # )

        sppf_ch, neck_filters, up, head_filters = self.get_parameters(
            heads=hparams.heads
        )

        self.modules["sppf"] = SPPF(*sppf_ch)
        self.modules["neck"] = Yolov8NeckOpt(
            filters=neck_filters, up=up, heads=hparams.heads
        )

        self.modules["head"] = DetectionHead(hparams.num_classes, filters=head_filters, heads=hparams.heads)
        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)

        print("Number of parameters for each module:")
        print(self.compute_params())

    def get_parameters(self, heads=[True, True, True]):
        """
        Gets the parameters with which to initialize the network detection part
        (SPPF block, Yolov8Neck, DetectionHead).

        Arguments
        ---------
        heads : Optional[List]
            List indicating whether each detection head is active.
            Default: [True, True, True].

        Returns
        -------
        Tuple containing the parameters for initializing the network detection part.
        Contains
            - Tuple (c1, c2): Tuple of input channel sizes for the SPPF block.
            - List neck_filters: List of filter sizes for Yolov8Neck.
            - List up: List of upsampling factors for Yolov8Neck.
            - List head_filters: List of filter sizes for DetectionHead. : Tuple
        """
        in_shape = self.modules["backbone"].input_shape
        x = torch.randn(1, *in_shape)
        y = self.modules["backbone"](x)

        c1 = c2 = y[0].shape[1]
        sppf = SPPF(c1, c2)
        out_sppf = sppf(y[0])

        neck_filters = [y[1][0].shape[1], y[1][1].shape[1], out_sppf.shape[1]]
        up = [2, 2]
        up[0] = y[1][1].shape[2] / out_sppf.shape[2]
        up[1] = y[1][0].shape[2] / (up[0] * out_sppf.shape[2])
        temp = """The layers you selected are not valid. \
            Please choose only layers between which the spatial resolution \
            doubles every time. Eventually, you can achieve this by \
            changing the downsampling layers. If you are trying to change \
            the input resolution, make sure you also change it in the \
            dataset configuration file and that it is a multiple of 4."""

        assert up == [2, 2], " ".join(temp.split())

        neck = Yolov8Neck(filters=neck_filters, up=up)
        out_neck = neck(y[1][0], y[1][1], out_sppf)

        head_filters = (
            out_neck[0].shape[1],
            out_neck[1].shape[1],
            out_neck[2].shape[1],
        )
        # keep only the heads we want
        head_filters = [head for heads, head in zip(heads, head_filters) if heads]

        return (c1, c2), neck_filters, up, head_filters

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
        backbone = self.modules["backbone"](preprocessed_batch["img"].to(self.device))
        neck_input = backbone[1]
        neck_input.append(self.modules["sppf"](backbone[0]))
        neck = self.modules["neck"](*neck_input)
        head = self.modules["head"](neck)
    
        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
        )

        return lossi_sum

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e6):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            print(
                f"optimizer: 'optimizer=auto' found, "
                f"ignoring 'lr0={lr}' and 'momentum={momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 80)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            # name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            name, lr, momentum = ("AdamW", lr_fit, 0.9)
            # self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        print(
            f"{optimizer:} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer, lr

    def _setup_scheduler(self, opt, lrf=0.01, lr0=0.01, cos_lr=True):
        """Initialize training learning rate scheduler."""
        def one_cycle(y1=0.0, y2=1.0, steps=100):
            """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
            return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

        lrf *= lr0

        if cos_lr:
            self.lf = one_cycle(1, lrf, 350)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - lrf) + lrf  # linear
        return optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lf)

    def configure_optimizers(self):
        """Configures the optimizer and the scheduler."""
        # opt = torch.optim.SGD(self.modules.parameters(), lr=1e-2, weight_decay=0.0005)
        # opt = torch.optim.AdamW(self.modules.parameters(), lr=0.000119, weight_decay=0.0)
        opt, lr = self.build_optimizer(
            self.modules, name="auto", lr=0.01, momentum=0.9
        )
        sched = self._setup_scheduler(opt, 0.01, lr)

        return opt, sched

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
        datasets={"train": train_loader}, # , "val": val_loader},
        metrics=[mAP],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )
    # yolo_mind.test(
    #     datasets={"test": val_loader},
    #     metrics=[mAP],
    # )
