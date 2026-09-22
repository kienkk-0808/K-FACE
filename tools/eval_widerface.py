"""Author: kienkk

Apples-to-apples mAP/AP@0.5/AP@0.75 (+ per-size AP@0.5) on WIDER FACE val
for K-FACE and/or a reference ONNX detector: same images, letterbox size,
score floor and NMS. Size buckets are the GT face height as a fraction of
its own image's height (small <10%, medium 10-30%, large >30%) — a proxy
for the official easy/medium/hard split, not the official leaderboard number.

Usage:
    python tools/eval_widerface.py --label data/widerface/val/label.txt \
        --images data/widerface/val/images --size 640 \
        --kface kface_n.onnx --ref path/to/reference_model.onnx
"""
import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kface.data.dataset import WiderFaceDataset  # noqa: E402
from kface.eval import IMAGENET_MEAN, IMAGENET_STD, VAR, evaluate_full, letterbox  # noqa: E402
from kface.models.detector import STRIDES as KFACE_STRIDES, DEFAULT_SCALES as KFACE_SCALES  # noqa: E402

# The reference detector is a separate, fixed architecture (unrelated to
# K-FACE's own strides/scales above) — don't reuse KFACE_STRIDES for it.
REF_STRIDES = (8, 16, 32)


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class KFaceRunner:
    def __init__(self, path, size, threads):
        self.sess = make_session(path, threads)
        self.size = size
        self.in_name = self.sess.get_inputs()[0].name
        self.anchors = self._anchors(size)

    @staticmethod
    def _anchors(size):
        out = []
        for stride, scales in zip(KFACE_STRIDES, KFACE_SCALES):
            f = size // stride
            cy, cx = np.mgrid[:f, :f].astype(np.float32)
            cx = (cx.reshape(-1) + 0.5) * stride
            cy = (cy.reshape(-1) + 0.5) * stride
            per_loc = np.stack([np.stack([cx, cy, np.full_like(cx, s), np.full_like(cy, s)], -1) for s in scales], 1)
            out.append(per_loc.reshape(-1, 4))
        return np.concatenate(out, 0)

    def __call__(self, img_bgr):
        net, scale = letterbox(img_bgr, self.size)
        rgb = cv2.cvtColor(net, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.in_name: inp})
        cls, box = [], []
        for i in range(len(KFACE_STRIDES)):
            c, b = outs[3 * i], outs[3 * i + 1]  # (1,A*1,H,W), (1,A*4,H,W)
            _, ch, h, w = b.shape
            a = ch // 4
            cls.append(c.reshape(a, 1, h, w).transpose(2, 3, 0, 1).reshape(-1))
            box.append(b.reshape(a, 4, h, w).transpose(2, 3, 0, 1).reshape(-1, 4))
        cls = np.concatenate(cls); box = np.concatenate(box)
        m = cls >= 0.02
        cls, box, an = cls[m], box[m], self.anchors[m]
        cx = box[:, 0] * VAR[0] * an[:, 2] + an[:, 0]
        cy = box[:, 1] * VAR[0] * an[:, 3] + an[:, 1]
        w = np.exp(np.clip(box[:, 2] * VAR[1], None, 8)) * an[:, 2]
        h = np.exp(np.clip(box[:, 3] * VAR[1], None, 8)) * an[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) / scale
        return boxes, cls


class RefDetRunner:
    """Anchor-point + ltrb-distance decoder for a reference ONNX detector:
    (img-127.5)/128 RGB input, sigmoid scores + ltrb distances per stride."""

    def __init__(self, path, size, threads):
        self.sess = make_session(path, threads)
        self.size = size
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, img_bgr):
        net, scale = letterbox(img_bgr, self.size)
        inp = cv2.dnn.blobFromImage(net, 1.0 / 128, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.in_name: inp})
        fmc = len(REF_STRIDES)
        all_boxes, all_scores = [], []
        for i, stride in enumerate(REF_STRIDES):
            scores = outs[i].reshape(-1)
            dist = outs[i + fmc] * stride
            f = self.size // stride
            centers = np.stack(np.mgrid[:f, :f][::-1], -1).astype(np.float32).reshape(-1, 2) * stride
            centers = np.stack([centers] * 2, 1).reshape(-1, 2)
            m = scores >= 0.02
            c, d = centers[m], dist[m]
            boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], -1)
            all_boxes.append(boxes); all_scores.append(scores[m])
        return np.concatenate(all_boxes) / scale, np.concatenate(all_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--kface", default=None)
    ap.add_argument("--ref", default=None, help="reference ONNX detector (ltrb-distance head format)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--score-floor", type=float, default=0.02)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ds = WiderFaceDataset(args.label, args.images)
    samples = ds.samples[: args.max_images] if args.max_images else ds.samples
    print(f"images={len(samples)} size={args.size} score_floor={args.score_floor} nms={args.nms}")

    if sum(len(s[1]) for s in samples) == 0:
        ap.error(
            f"'{args.label}' has no ground-truth boxes (looks like a plain "
            "image list, e.g. wider_val.txt) — AP can't be computed from it. "
            "Point --label at the annotated val label.txt (RetinaFace format, "
            "same as the train label.txt) instead."
        )

    runners = []
    if args.kface:
        runners.append(("K-FACE", KFaceRunner(args.kface, args.size, args.threads)))
    if args.ref:
        runners.append(("REF", RefDetRunner(args.ref, args.size, args.threads)))
    if not runners:
        ap.error("give --kface and/or --ref")

    order = ["mAP", "AP50", "AP75", "small<10%", "medium10-30%", "large>30%"]
    header = f"{'model':8s}" + "".join(f"{k:>14s}" for k in order) + f"{'ms/img':>9s}"
    print(header)
    for name, r in runners:
        res, n_gt, ms = evaluate_full(r, samples, args.score_floor, args.nms)
        print(f"{name:8s}" + "".join(f"{100 * res[k]:14.2f}" for k in order) + f"{ms:9.2f}")
    print("GT count per size bucket: " + ", ".join(f"{k}={n_gt[k]}" for k in n_gt))


if __name__ == "__main__":
    main()
