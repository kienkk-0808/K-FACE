"""Lightweight PAFPN neck: top-down + bottom-up fusion across 3 levels.

Kept intentionally small (1x1 lateral + depthwise 3x3 smooth) since the
neck runs at full spatial resolution and is a common CPU-latency hotspot.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import conv_bn, DWSepBlock


class KFaceNeck(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        c3, c4, c5 = in_channels

        self.lat3 = conv_bn(c3, out_channels, k=1, s=1)
        self.lat4 = conv_bn(c4, out_channels, k=1, s=1)
        self.lat5 = conv_bn(c5, out_channels, k=1, s=1)

        self.td4 = DWSepBlock(out_channels, out_channels)
        self.td3 = DWSepBlock(out_channels, out_channels)

        # bottom-up second pass (PAN) for better small-face localization
        self.down3 = conv_bn(out_channels, out_channels, k=3, s=2)
        self.bu4 = DWSepBlock(out_channels, out_channels)
        self.down4 = conv_bn(out_channels, out_channels, k=3, s=2)
        self.bu5 = DWSepBlock(out_channels, out_channels)

        self.out_channels = out_channels

    def forward(self, feats):
        c3, c4, c5 = feats
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.td4(p4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.td3(p3)

        n3 = p3
        n4 = self.bu4(p4 + self.down3(n3))
        n5 = self.bu5(p5 + self.down4(n4))
        return n3, n4, n5  # stride 8, 16, 32
