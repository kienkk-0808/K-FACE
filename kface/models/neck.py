"""Author: kienkk

Top-down FPN neck over N levels (2 or 3). Smoothing per level is
configurable ("dw" = depthwise-separable, "dense" = plain 3x3, "none" = skip).
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
    def __init__(self, in_channels, out_channels=48, smooth=("dense",)):
        """in_channels: tuple of backbone output channels, finest stride
        first (e.g. (c16, c32) or (c8, c16, c32) — matches KFaceBackbone's
        forward() order). smooth: one entry per level below the coarsest
        (len(in_channels) - 1 entries)."""
        super().__init__()
        n = len(in_channels)
        assert len(smooth) == n - 1, f"need {n - 1} smooth entries for {n} levels, got {len(smooth)}"
        self.lats = nn.ModuleList([conv_bn(c, out_channels, k=1) for c in in_channels])
        self.smooths = nn.ModuleList([_smooth(s, out_channels) for s in smooth])
        self.out_channels = out_channels

    def forward(self, feats):
        # feats: finest stride first (e.g. c16, c32); last entry is coarsest
        laterals = [lat(f) for lat, f in zip(self.lats, feats)]
        outs = [None] * len(laterals)
        outs[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(outs[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            outs[i] = self.smooths[i](laterals[i] + up)
        return tuple(outs)  # same order as input: finest stride first
