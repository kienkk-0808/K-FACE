"""Author: kienkk

Convert a K-FACE model to OpenVINO IR (.xml + .bin). Accepts either a
training checkpoint (--ckpt, exports to ONNX first) or an already-exported
ONNX file (--onnx).

Usage:
    python tools/export_openvino.py --ckpt runs/kface_n/best.pth --out kface_n_ov --size 320
    python tools/export_openvino.py --onnx kface_n.onnx --out kface_n_ov
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kface.models.detector import build_model  # noqa: E402
from tools.export_onnx import ExportWrapper  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="training checkpoint (.pth)")
    ap.add_argument("--onnx", default=None, help="already-exported ONNX file")
    ap.add_argument("--out", required=True, help="output path/prefix, e.g. kface_n_ov -> kface_n_ov.xml/.bin")
    ap.add_argument("--size", type=int, default=320, help="input size, only used with --ckpt")
    ap.add_argument("--opset", type=int, default=12, help="only used with --ckpt")
    ap.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA, only with --ckpt")
    ap.add_argument("--fp16", action="store_true", help="compress IR weights to fp16 (halves file size)")
    args = ap.parse_args()
    if not args.ckpt and not args.onnx:
        ap.error("give --ckpt or --onnx")
    if args.ckpt and args.onnx:
        ap.error("give only one of --ckpt or --onnx")
    return args


def ckpt_to_onnx_path(ckpt_path, size, opset, use_ema, tmp_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    key = "ema" if ("ema" in ckpt and use_ema) else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

    output_names = []
    for s in cfg["model"]["strides"]:
        output_names += [f"s{s}_cls", f"s{s}_box", f"s{s}_kps"]

    torch.onnx.export(
        ExportWrapper(model), torch.randn(1, 3, size, size), tmp_path,
        input_names=["input"], output_names=output_names,
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    print(f"(intermediate) exported {key} weights to ONNX: {tmp_path}")
    return output_names


def main():
    args = parse_args()

    try:
        import openvino as ov
    except ImportError:
        raise SystemExit(
            "OpenVINO is not installed. Install it with:\n"
            "    pip install openvino\n"
            "then re-run this command."
        )

    onnx_path = args.onnx
    tmp_onnx = None
    if args.ckpt:
        tmp_onnx = args.out + "._tmp.onnx"
        ckpt_to_onnx_path(args.ckpt, args.size, args.opset, not args.no_ema, tmp_onnx)
        onnx_path = tmp_onnx

    ov_model = ov.convert_model(onnx_path)
    xml_path = args.out if args.out.endswith(".xml") else args.out + ".xml"
    ov.save_model(ov_model, xml_path, compress_to_fp16=args.fp16)

    if tmp_onnx and os.path.exists(tmp_onnx):
        os.remove(tmp_onnx)

    print(f"Exported OpenVINO IR: {xml_path} (+ {os.path.splitext(xml_path)[0]}.bin), fp16={args.fp16}")


if __name__ == "__main__":
    main()
