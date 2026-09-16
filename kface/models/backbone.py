"""Author: kienkk

Config-driven backbone for K-FACE. Block types: "dw" (depthwise-separable),
"ir" (inverted residual), "dense" (plain residual). Returns features at
stride 8, 16, 32.
"""
import torch.nn as nn


def conv_bn(cin, cout, k=3, s=1, p=None, groups=1, act=True):
    if p is None:
        p = k // 2
    layers = [
        nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DWBlock(nn.Module):
    """depthwise 3x3 (optionally stride 2) -> pointwise 1x1."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = conv_bn(cin, cin, k=3, s=stride, groups=cin)
        self.pw = conv_bn(cin, cout, k=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class IRBlock(nn.Module):
    """1x1 expand -> depthwise 3x3 -> 1x1 project, residual (stride 1 only)."""

    def __init__(self, cin, cout, expand=2):
        super().__init__()
        mid = int(cin * expand)
        self.use_res = cin == cout
        self.pw1 = conv_bn(cin, mid, k=1)
        self.dw = conv_bn(mid, mid, k=3, groups=mid)
        self.pw2 = conv_bn(mid, cout, k=1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pw2(self.dw(self.pw1(x)))
        if self.use_res:
            out = out + x
        return self.act(out)


class DenseBlock(nn.Module):
    """residual 3x3 -> 3x3 (stride 1, cin == cout)."""

    def __init__(self, cin, cout):
        super().__init__()
        assert cin == cout
        self.conv1 = conv_bn(cin, cout, k=3)
        self.conv2 = conv_bn(cout, cout, k=3, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + x)


def make_stage(cin, spec):
    cout = spec["channels"]
    layers = [DWBlock(cin, cout, stride=2)]
    for _ in range(spec.get("blocks", 0)):
        t = spec["type"]
        if t == "dw":
            layers.append(DWBlock(cout, cout))
        elif t == "ir":
            layers.append(IRBlock(cout, cout, expand=spec.get("expand", 2)))
        elif t == "dense":
            layers.append(DenseBlock(cout, cout))
        else:
            raise ValueError(f"unknown block type {t}")
    return nn.Sequential(*layers)


class KFaceBackbone(nn.Module):
    def __init__(self, stem_channels=16, stages=None):
        super().__init__()
        assert stages is not None and len(stages) == 4
        self.stem = conv_bn(3, stem_channels, k=3, s=2)
        built = []
        cin = stem_channels
        for spec in stages:
            built.append(make_stage(cin, spec))
            cin = spec["channels"]
        self.stage2, self.stage3, self.stage4, self.stage5 = built
        self.out_channels = tuple(s["channels"] for s in stages[1:])

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5
