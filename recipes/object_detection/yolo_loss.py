"""
Wrapper for the YOLO loss, from the ultralytics implementation.
For a reference on the parameters, please refer to https://shorturl.at/gkrAO


Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


class Loss(v8DetectionLoss):
    def __init__(self, h, m, device):  # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocesses the target counts and matches with the input batch size
        to output a tensor.
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and
        distribution.
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        # breakpoint()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # print("pred scores shape ", pred_scores.shape)    # x, 8400, 80
        # print("pred distri shape ", pred_distri.shape)    # x, 8400, 64 (reg_max * 4)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        # print(imgsz)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # print("pred bboxes")
        # print(pred_bboxes.shape)
        # print(pred_bboxes[0, 0])

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.clone().detach().sigmoid(),
            (pred_bboxes.clone().detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # print("target bboxes")
        # print(target_bboxes.shape)
        # print(target_bboxes[0, 0])

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # print("classification loss", loss[1])
        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # print("target bboxes w norm")
            # print(target_bboxes.shape)
            # print(target_bboxes[0, 0])

            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # with open("dump.txt", "a") as f:
        # f.write("bbox loss {}".format(loss[0].item()))
        # f.write("\n")
        # f.write("cls loss {}".format(loss[1].item()))
        # f.write("\n")
        # f.write("dfl loss {}".format(loss[2].item()))
        # f.write("\n")
        # f.write("total {}".format(loss.sum().item()))
        # f.write("\n")
        # f.write("total * batch_size {}".format(loss.sum().item() * batch_size))
        # f.write("\n")
        #
        # breakpoint()

        # print(torch.std_mean(batch["img"]))
        # breakpoint()

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
