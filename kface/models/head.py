"""Shared-stem detection head: cls (1 logit/anchor), box (4/anchor),
kps (10/anchor = 5 landmark points). Weights are shared across the 3 FPN
strides to cut parameter count; a per-stride learnable scalar rescales the
box regression so shared weights can still fit different stride magnitudes.
"""
import torch
import torch.nn as nn

from .backbone import DWSepBlock


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class KFaceHead(nn.Module):
    def __init__(self, in_channels, num_anchors=2, stem_blocks=2, strides=(8, 16, 32)):
        super().__init__()
        self.num_anchors = num_anchors
        self.strides = strides

        self.cls_stem = nn.Sequential(*[DWSepBlock(in_channels, in_channels) for _ in range(stem_blocks)])
        self.reg_stem = nn.Sequential(*[DWSepBlock(in_channels, in_channels) for _ in range(stem_blocks)])

        self.cls_pred = nn.Conv2d(in_channels, num_anchors * 1, 3, 1, 1)
        self.box_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, 1, 1)
        self.kps_pred = nn.Conv2d(in_channels, num_anchors * 10, 3, 1, 1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in strides])

        prior_prob = 0.01
        bias_value = -float(torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(self, feats):
        cls_outs, box_outs, kps_outs = [], [], []
        for feat, scale in zip(feats, self.scales):
            cls_feat = self.cls_stem(feat)
            reg_feat = self.reg_stem(feat)

            cls_out = self.cls_pred(cls_feat)
            box_out = scale(self.box_pred(reg_feat))
            kps_out = self.kps_pred(reg_feat)

            cls_outs.append(cls_out)
            box_outs.append(box_out)
            kps_outs.append(kps_out)
        return cls_outs, box_outs, kps_outs
