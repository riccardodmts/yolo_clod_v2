from micromind.networks.yolo import Yolov8Neck, DetectionHead, SPPF
from micromind.networks import PhiNet
import torch
import micromind as mm

from micromind.utils import parse_configuration
from micromind.utils.yolo import (
    load_config,
)
import sys
from yolo_loss import Loss
from validation.validator import DetectionValidator


class YOLO(mm.MicroMind):
    def __init__(self, m_cfg, hparams, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(*args, **kwargs)
        self.m_cfg = m_cfg

        self.modules["backbone"] = PhiNet(
            input_shape=hparams.input_shape,
            alpha=hparams.alpha,
            num_layers=hparams.num_layers,
            beta=hparams.beta,
            t_zero=hparams.t_zero,
            include_top=False,
            compatibility=False,
            divisor=hparams.divisor,
            downsampling_layers=hparams.downsampling_layers,
            return_layers=hparams.return_layers,
        )

        sppf_ch, neck_filters, up, head_filters = self.get_parameters(
            heads=hparams.heads
        )

        self.modules["sppf"] = SPPF(*sppf_ch)
        self.modules["neck"] = Yolov8Neck(
            filters=neck_filters, up=up, heads=hparams.heads
        )
        self.modules["head"] = DetectionHead(filters=head_filters, heads=hparams.heads)

        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)

        print("Number of parameters for each module:")
        print(self.compute_params())

    def get_parameters(self, heads=[True, True, True]):
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
        )
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)

        return preprocessed_batch

    def compute_loss(self, pred, batch):
        pass

    def forward(self, img):
        """Runs the forward method by calling every module."""
        backbone = self.modules["backbone"](img)
        neck_input = backbone[1]
        neck_input.append(self.modules["sppf"](backbone[0]))
        neck = self.modules["neck"](*neck_input)
        head = self.modules["head"](neck)

        return head


if __name__ == "__main__":
    assertion_msg = "Usage: python validate.py <hparams_config> <model_weights>"
    assert len(sys.argv) >= 3, assertion_msg

    hparams = parse_configuration(sys.argv[1])
    m_cfg, data_cfg = load_config(hparams.data_cfg)
    model_weights_path = sys.argv[2]

    args = dict(model="yolov8n.pt", data="VOC.yaml")
    validator = DetectionValidator(args=args)

    model = YOLO(m_cfg, hparams)
    model.load_modules(model_weights_path)

    val = validator(model=model)

    print("METRICS:")
    print("Box map50-95:", round(validator.metrics.box.map * 100, 3), "%")
    print("Box map50:", round(validator.metrics.box.map50 * 100, 3), "%")
    print("Box map75:", round(validator.metrics.box.map75 * 100, 3), "%")
