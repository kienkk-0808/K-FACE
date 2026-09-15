"""Export a trained K-FACE checkpoint to ONNX (fixed input size, opset 12,
raw per-stride cls/box/kps outputs — decode + NMS stay in Python at
inference time so thresholds are tunable without re-exporting).

Usage:
    python tools/export_onnx.py --ckpt runs/kface_n/epoch_299.pth \
        --out kface_n.onnx --size 320
"""
import argparse
import torch

from kface.models.detector import KFaceDetector


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--opset", type=int, default=12)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]

    model = KFaceDetector(
        width_mult=cfg["model"]["width_mult"],
        depth_mult=cfg["model"]["depth_mult"],
        neck_channels=cfg["model"]["neck_channels"],
        num_anchors=cfg["model"]["num_anchors"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    strides = cfg["model"]["strides"]

    output_names = []
    for s in strides:
        output_names += [f"s{s}_cls", f"s{s}_box", f"s{s}_kps"]

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            cls_outs, box_outs, kps_outs = self.m(x)
            outs = []
            for c, b, k in zip(cls_outs, box_outs, kps_outs):
                outs += [c.sigmoid(), b, k]
            return tuple(outs)

    torch.onnx.export(
        Wrapper(model), dummy, args.out,
        input_names=["input"], output_names=output_names,
        opset_version=args.opset, do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported: {args.out}")
    print(f"Output order: {output_names}")


if __name__ == "__main__":
    main()
