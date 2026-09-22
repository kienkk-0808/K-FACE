"""Author: kienkk

Decoupled detection head (separate cls/reg stems), weights shared across
the FPN strides. Outputs per stride: cls (A x 1), box (A x 4), kps (A x 10).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DWBlock, conv_bn


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class KpsRefine(nn.Module):
    """Upsamples the reg feature 2x, refines it there, then downsamples
    back to the anchor grid resolution. Landmarks need sub-cell precision
    that a plain 1x1 conv on the same coarse grid as box/cls can't give —
    this lets the kps branch look at finer local detail while keeping its
    output on the same (H, W) grid the anchors are decoded against."""

    def __init__(self, channels):
        super().__init__()
        self.refine = DWBlock(channels, channels)
        self.down = conv_bn(channels, channels, k=3, s=2)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.down(self.refine(up))


class KFaceHead(nn.Module):
    def __init__(self, in_channels, num_anchors=2, stem_blocks=1, strides=(8, 16, 32)):
        super().__init__()
        self.num_anchors = num_anchors
        self.strides = strides

        self.cls_stem = nn.Sequential(*[DWBlock(in_channels, in_channels) for _ in range(stem_blocks)])
        self.reg_stem = nn.Sequential(*[DWBlock(in_channels, in_channels) for _ in range(stem_blocks)])
        self.kps_refine = KpsRefine(in_channels)

        self.cls_pred = nn.Conv2d(in_channels, num_anchors * 1, 1)
        self.box_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.kps_pred = nn.Conv2d(in_channels, num_anchors * 10, 1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])

        prior_prob = 0.01
        bias_value = -float(torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(self, feats):
        cls_outs, box_outs, kps_outs = [], [], []
        for feat, scale in zip(feats, self.scales):
            cls_feat = self.cls_stem(feat)
            reg_feat = self.reg_stem(feat)
            kps_feat = self.kps_refine(reg_feat)
            cls_outs.append(self.cls_pred(cls_feat))
            box_outs.append(scale(self.box_pred(reg_feat)))
            kps_outs.append(self.kps_pred(kps_feat))
        return cls_outs, box_outs, kps_outs
