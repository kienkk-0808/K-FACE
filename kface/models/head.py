"""Author: kienkk

Decoupled detection head, weights shared across the 3 FPN strides.

Separate classification / regression stems: cls wants translation-invariant
features while box+kps regression wants precise localisation features, and
sharing one stem for both costs accuracy.

Cost note: the head runs on every level, so each of its parameters costs
3200 + 800 + 200 = 4200 FLOPs at 320 input. Stems are therefore
depthwise 3x3 -> pointwise 1x1 blocks, and the prediction layers are 1x1
(the stem already provides 3x3 context). A 3x3 kps_pred alone would cost
~50 MFLOPs — an eighth of the whole network's budget.

Outputs per stride: cls (A x 1 logit), box (A x 4), kps (A x 10).
A per-stride learnable Scale on box lets shared weights fit different
stride magnitudes.
"""
import torch
import torch.nn as nn

from .backbone import DWBlock


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class KFaceHead(nn.Module):
    def __init__(self, in_channels, num_anchors=2, stem_blocks=1, strides=(8, 16, 32)):
        super().__init__()
        self.num_anchors = num_anchors
        self.strides = strides

        self.cls_stem = nn.Sequential(*[DWBlock(in_channels, in_channels) for _ in range(stem_blocks)])
        self.reg_stem = nn.Sequential(*[DWBlock(in_channels, in_channels) for _ in range(stem_blocks)])

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
            cls_outs.append(self.cls_pred(cls_feat))
            box_outs.append(scale(self.box_pred(reg_feat)))
            kps_outs.append(self.kps_pred(reg_feat))
        return cls_outs, box_outs, kps_outs
