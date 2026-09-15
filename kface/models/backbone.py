"""Lightweight depthwise-separable backbone for K-FACE.

Design goals: run >60 FPS on CPU (ONNXRuntime, fp32) at 320x320 input while
keeping enough capacity for high recall/precision on small faces. Structure
follows a MobileNet/ShuffleNet-style stage layout (stem -> 4 stages) with
few channels and a light stem to stay fast on CPU.
"""
import torch
import torch.nn as nn


def conv_bn(cin, cout, k=3, s=1, p=None, groups=1):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DWSepBlock(nn.Module):
    """Depthwise 3x3 + Pointwise 1x1, with residual add when shapes match."""

    def __init__(self, cin, cout, stride=1, expand=2):
        super().__init__()
        mid = int(cin * expand)
        self.use_res = stride == 1 and cin == cout
        self.pw1 = conv_bn(cin, mid, k=1, s=1)
        self.dw = conv_bn(mid, mid, k=3, s=stride, groups=mid)
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid, cout, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pw1(x)
        out = self.dw(out)
        out = self.pw2(out)
        if self.use_res:
            out = out + x
        return self.act(out)


def make_stage(cin, cout, num_blocks, stride, expand=2):
    layers = [DWSepBlock(cin, cout, stride=stride, expand=expand)]
    for _ in range(num_blocks - 1):
        layers.append(DWSepBlock(cout, cout, stride=1, expand=expand))
    return nn.Sequential(*layers)


class KFaceBackbone(nn.Module):
    """Outputs feature maps at stride 8 (C3), 16 (C4), 32 (C5).

    width_mult scales all channel counts; depth_mult scales block counts.
    Defaults ("nano") target CPU real-time; use "small" for higher accuracy
    when GPU inference or a slower FPS budget is acceptable.
    """

    def __init__(self, width_mult=1.0, depth_mult=1.0):
        super().__init__()

        def c(ch):
            return max(8, int(round(ch * width_mult / 8)) * 8)

        def d(n):
            return max(1, int(round(n * depth_mult)))

        self.stem = nn.Sequential(
            conv_bn(3, c(16), k=3, s=2),          # stride 2
            DWSepBlock(c(16), c(16), stride=1),
        )
        self.stage2 = make_stage(c(16), c(32), d(2), stride=2)   # stride 4
        self.stage3 = make_stage(c(32), c(64), d(2), stride=2)   # stride 8  -> C3
        self.stage4 = make_stage(c(64), c(96), d(3), stride=2)   # stride 16 -> C4
        self.stage5 = make_stage(c(96), c(128), d(3), stride=2)  # stride 32 -> C5

        self.out_channels = (c(64), c(96), c(128))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


if __name__ == "__main__":
    net = KFaceBackbone()
    y = net(torch.randn(1, 3, 320, 320))
    for t in y:
        print(t.shape)
