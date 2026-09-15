"""Author: kienkk

Export a trained K-FACE checkpoint to ONNX (fixed input size, opset 12,
raw per-stride cls(sigmoid)/box/kps outputs — decode + NMS stay in Python
so thresholds are tunable without re-exporting). Uses the EMA weights when
the checkpoint has them.

Usage:
    python tools/export_onnx.py --ckpt runs/kface_n/epoch_299.pth --out kface_n.onnx --size 320
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kface.models.detector import build_model  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--no-ema", action="store_true", help="export raw weights instead of EMA")
    return ap.parse_args()


class ExportWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        cls_outs, box_outs, kps_outs = self.m(x)
        outs = []
        for c, b, k in zip(cls_outs, box_outs, kps_outs):
            outs += [c.sigmoid(), b, k]
        return tuple(outs)


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]

    model = build_model(cfg)
    key = "ema" if ("ema" in ckpt and not args.no_ema) else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

    output_names = []
    for s in cfg["model"]["strides"]:
        output_names += [f"s{s}_cls", f"s{s}_box", f"s{s}_kps"]

    torch.onnx.export(
        ExportWrapper(model), torch.randn(1, 3, args.size, args.size), args.out,
        input_names=["input"], output_names=output_names,
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    print(f"Exported ({key} weights): {args.out}")
    print(f"Output order: {output_names}")


if __name__ == "__main__":
    main()
