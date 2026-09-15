"""K-FACE detector: backbone + neck + head, assembled for both training
(raw per-stride tensors) and inference (flattened, decoded, NMS'd boxes+kps).
"""
import torch
import torch.nn as nn

from .backbone import KFaceBackbone
from .neck import KFaceNeck
from .head import KFaceHead
from ..utils.box_utils import generate_anchors, decode_boxes, decode_kps, nms

STRIDES = (8, 16, 32)
DEFAULT_SCALES = ((16, 32), (64, 128), (256, 512))


class KFaceDetector(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, neck_channels=64,
                 num_anchors=2, strides=STRIDES):
        super().__init__()
        self.backbone = KFaceBackbone(width_mult=width_mult, depth_mult=depth_mult)
        self.neck = KFaceNeck(self.backbone.out_channels, out_channels=neck_channels)
        self.head = KFaceHead(neck_channels, num_anchors=num_anchors, strides=strides)
        self.strides = strides
        self.num_anchors = num_anchors

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        cls_outs, box_outs, kps_outs = self.head(feats)
        return cls_outs, box_outs, kps_outs

    @staticmethod
    def flatten_head_output(t, last_dim):
        """(B, A*last_dim, H, W) -> (B, H*W*A, last_dim), matching the
        anchor layout produced by generate_anchors (per-location, per-anchor).
        """
        b, c, h, w = t.shape
        num_anchors = c // last_dim
        t = t.view(b, num_anchors, last_dim, h, w)
        t = t.permute(0, 3, 4, 1, 2).contiguous()  # B,H,W,A,last_dim
        return t.view(b, h * w * num_anchors, last_dim)

    def flatten_all(self, cls_outs, box_outs, kps_outs):
        cls = torch.cat([self.flatten_head_output(c, 1) for c in cls_outs], dim=1)
        box = torch.cat([self.flatten_head_output(b, 4) for b in box_outs], dim=1)
        kps = torch.cat([self.flatten_head_output(k, 10) for k in kps_outs], dim=1)
        return cls, box, kps

    @torch.no_grad()
    def predict(self, x, anchors=None, conf_thresh=0.5, iou_thresh=0.4, scales=DEFAULT_SCALES):
        """Inference helper (Python-side; the exported ONNX graph only goes
        up to raw cls/box/kps outputs — decode+NMS run in tools/infer.py so
        NMS thresholds can be tuned without re-exporting).
        """
        self.eval()
        cls_outs, box_outs, kps_outs = self.forward(x)
        cls, box, kps = self.flatten_all(cls_outs, box_outs, kps_outs)
        cls = cls.sigmoid()

        if anchors is None:
            anchors = generate_anchors(x.shape[-2:], strides=self.strides, scales_per_stride=scales).to(x.device)

        results = []
        for b in range(x.shape[0]):
            scores = cls[b, :, 0]
            mask = scores >= conf_thresh
            if mask.sum() == 0:
                results.append((torch.zeros(0, 4), torch.zeros(0, 5, 2), torch.zeros(0)))
                continue
            a = anchors[mask]
            boxes = decode_boxes(box[b][mask], a)
            landm = decode_kps(kps[b][mask], a)
            sc = scores[mask]
            keep = nms(boxes, sc, iou_thresh=iou_thresh)
            results.append((boxes[keep], landm[keep], sc[keep]))
        return results
