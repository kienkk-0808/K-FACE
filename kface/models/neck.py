"""Author: kienkk

Top-down FPN neck over 3 levels.

Smoothing after each fusion is configurable per level ("dw" = depthwise
3x3 -> pointwise 1x1, "dense" = plain 3x3). At stride 8 a dense 3x3 is
the most expensive layer in the whole network (3200 FLOPs per parameter),
so the default is dw there and dense at stride 16. Stride 32 is used
directly from its lateral projection.
"""
import torch.nn as nn
import torch.nn.functional as F

from .backbone import conv_bn, DWBlock


def _smooth(kind, ch):
    if kind == "dw":
        return DWBlock(ch, ch)
    if kind == "dense":
        return conv_bn(ch, ch, k=3)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


class KFaceNeck(nn.Module):
    def __init__(self, in_channels, out_channels=48, smooth=("dw", "dense")):
        super().__init__()
        c3, c4, c5 = in_channels
        self.lat3 = conv_bn(c3, out_channels, k=1)
        self.lat4 = conv_bn(c4, out_channels, k=1)
        self.lat5 = conv_bn(c5, out_channels, k=1)
        self.smooth3 = _smooth(smooth[0], out_channels)
        self.smooth4 = _smooth(smooth[1], out_channels)
        self.out_channels = out_channels

    def forward(self, feats):
        c3, c4, c5 = feats
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.smooth4(p4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        return p3, p4, p5  # stride 8, 16, 32
