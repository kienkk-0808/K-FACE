"""Author: kienkk

Config-driven backbone for K-FACE. Block types: "dw" (depthwise-separable),
"ir" (inverted residual), "dense" (plain residual). Always builds 4 stages
(stride 4/8/16/32) but only returns the levels named in `out_stages` —
stride 4 and (by default) stride 8 stay internal, feeding forward into the
deeper stages without paying for a detection head on their own feature map.
Stride 8 is the most expensive level to run a head on (largest spatial
size) and only helps detect faces smaller than most deployments need, so
dropping it from the outputs frees that budget for stride 16/32 capacity.
"""
import torch.nn as nn


class SE(nn.Module):
    """Squeeze-Excitation channel attention — global-avg-pool -> tiny MLP ->
    per-channel gate. Cheap (operates on a 1x1 pooled vector) but a
    consistently reliable accuracy gain (SENet/MobileNetV3/EfficientNet),
    so it's worth its small cost on the stages we invest capacity in."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        s = self.act1(self.fc1(self.pool(x)))
        s = self.act2(self.fc2(s))
        return x * s


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
    """1x1 expand -> depthwise 3x3 -> [SE] -> 1x1 project, residual (stride 1 only)."""

    def __init__(self, cin, cout, expand=2, use_se=False):
        super().__init__()
        mid = int(cin * expand)
        self.use_res = cin == cout
        self.pw1 = conv_bn(cin, mid, k=1)
        self.dw = conv_bn(mid, mid, k=3, groups=mid)
        self.se = SE(mid) if use_se else nn.Identity()
        self.pw2 = conv_bn(mid, cout, k=1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pw2(self.se(self.dw(self.pw1(x))))
        if self.use_res:
            out = out + x
        return self.act(out)


class DenseBlock(nn.Module):
    """residual 3x3 -> 3x3 -> [SE] (stride 1, cin == cout)."""

    def __init__(self, cin, cout, use_se=False):
        super().__init__()
        assert cin == cout
        self.conv1 = conv_bn(cin, cout, k=3)
        self.conv2 = conv_bn(cout, cout, k=3, act=False)
        self.se = SE(cout) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.se(self.conv2(self.conv1(x))) + x)


def make_stage(cin, spec):
    cout = spec["channels"]
    use_se = spec.get("se", False)
    layers = [DWBlock(cin, cout, stride=2)]
    for _ in range(spec.get("blocks", 0)):
        t = spec["type"]
        if t == "dw":
            layers.append(DWBlock(cout, cout))
        elif t == "ir":
            layers.append(IRBlock(cout, cout, expand=spec.get("expand", 2), use_se=use_se))
        elif t == "dense":
            layers.append(DenseBlock(cout, cout, use_se=use_se))
        else:
            raise ValueError(f"unknown block type {t}")
    return nn.Sequential(*layers)


class KFaceBackbone(nn.Module):
    def __init__(self, stem_channels=16, stages=None, out_stages=(2, 3)):
        """stages: one spec per downsample stage, stride doubling each time
        starting from stride 4 (stem already halves once). E.g. 5 stages ->
        strides 4/8/16/32/64. out_stages: which stage INDICES to return,
        e.g. (2, 3) -> stride 16+32; (2, 3, 4) -> stride 16/32/64.
        """
        super().__init__()
        assert stages is not None and len(stages) >= 1
        self.stem = conv_bn(3, stem_channels, k=3, s=2)
        built = []
        cin = stem_channels
        for spec in stages:
            built.append(make_stage(cin, spec))
            cin = spec["channels"]
        self.stages = nn.ModuleList(built)
        self.out_stages = tuple(out_stages)
        self.out_channels = tuple(stages[i]["channels"] for i in out_stages)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_stages:
                outs.append(x)
        return tuple(outs)
