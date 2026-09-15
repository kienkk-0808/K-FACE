"""Author: kienkk

Apples-to-apples AP@0.5 on WIDER FACE val for K-FACE and/or a reference
ONNX detector: same images, same letterbox size, same score floor, same NMS.

Reports overall AP plus AP by GT face height in the ORIGINAL image
(small < 32px, medium 32-96px, large > 96px). These buckets are a proxy
for WIDER's official easy/medium/hard split (which needs the .mat ground
truth files); they are consistent across models so the comparison is fair,
but the absolute numbers are not the official leaderboard numbers.

Usage:
    python tools/eval_widerface.py --label data/widerface/val/label.txt \
        --images data/widerface/val/images --size 320 \
        --kface kface_n.onnx --ref path/to/reference_model.onnx
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kface.data.dataset import WiderFaceDataset  # noqa: E402

STRIDES = (8, 16, 32)
KFACE_SCALES = ((16, 32), (64, 128), (256, 512))
KFACE_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
KFACE_STD = np.array([0.229, 0.224, 0.225], np.float32)
VAR = (0.1, 0.2)


def letterbox(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return canvas, scale


def nms_np(boxes, scores, thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a_i + a_r - inter + 1e-9)
        order = rest[iou <= thr]
    return np.array(keep, dtype=np.int64)


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
        for stride, scales in zip(STRIDES, KFACE_SCALES):
            f = size // stride
            cy, cx = np.mgrid[:f, :f].astype(np.float32)
            cx = (cx.reshape(-1) + 0.5) * stride
            cy = (cy.reshape(-1) + 0.5) * stride
            per_loc = np.stack([np.stack([cx, cy, np.full_like(cx, s), np.full_like(cy, s)], -1) for s in scales], 1)
            out.append(per_loc.reshape(-1, 4))
        return np.concatenate(out, 0)

    def __call__(self, img_bgr, score_floor):
        net, scale = letterbox(img_bgr, self.size)
        rgb = cv2.cvtColor(net, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = ((rgb - KFACE_MEAN) / KFACE_STD).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.in_name: inp})
        cls, box = [], []
        for i in range(len(STRIDES)):
            c, b = outs[3 * i], outs[3 * i + 1]  # (1,A*1,H,W), (1,A*4,H,W)
            _, ch, h, w = b.shape
            a = ch // 4
            cls.append(c.reshape(a, 1, h, w).transpose(2, 3, 0, 1).reshape(-1))
            box.append(b.reshape(a, 4, h, w).transpose(2, 3, 0, 1).reshape(-1, 4))
        cls = np.concatenate(cls); box = np.concatenate(box)
        m = cls >= score_floor
        cls, box, an = cls[m], box[m], self.anchors[m]
        cx = box[:, 0] * VAR[0] * an[:, 2] + an[:, 0]
        cy = box[:, 1] * VAR[0] * an[:, 3] + an[:, 1]
        w = np.exp(np.clip(box[:, 2] * VAR[1], None, 8)) * an[:, 2]
        h = np.exp(np.clip(box[:, 3] * VAR[1], None, 8)) * an[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) / scale
        return boxes, cls


class RefDetRunner:
    """Generic anchor-point + ltrb-distance decoder for a reference ONNX
    detector: (img-127.5)/128 RGB input; per stride the graph emits sigmoid
    scores (N,1) and ltrb distances (N,4) in stride units, N = H*W*num_anchors
    anchors, location-major. Matches the common distance-based head format
    used by several lightweight face detectors."""

    def __init__(self, path, size, threads):
        self.sess = make_session(path, threads)
        self.size = size
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, img_bgr, score_floor):
        net, scale = letterbox(img_bgr, self.size)
        inp = cv2.dnn.blobFromImage(net, 1.0 / 128, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.in_name: inp})
        fmc = len(STRIDES)
        all_boxes, all_scores = [], []
        for i, stride in enumerate(STRIDES):
            scores = outs[i].reshape(-1)
            dist = outs[i + fmc] * stride
            f = self.size // stride
            centers = np.stack(np.mgrid[:f, :f][::-1], -1).astype(np.float32).reshape(-1, 2) * stride
            centers = np.stack([centers] * 2, 1).reshape(-1, 2)
            m = scores >= score_floor
            c, d = centers[m], dist[m]
            boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], -1)
            all_boxes.append(boxes); all_scores.append(scores[m])
        return np.concatenate(all_boxes) / scale, np.concatenate(all_scores)


def iou_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def voc_ap(tp, fp, n_gt):
    if n_gt == 0:
        return float("nan")
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


BUCKETS = {"all": (0, np.inf), "small<32": (0, 32), "medium32-96": (32, 96), "large>96": (96, np.inf)}


def evaluate(runner, samples, image_root_unused, score_floor, nms_thr, iou_thr):
    records = {k: [] for k in BUCKETS}  # (score, tp, fp)
    n_gt = {k: 0 for k in BUCKETS}
    t_infer = 0.0
    for img_path, gt, _, _ in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gt = gt[(gt[:, 2] - gt[:, 0] >= 2) & (gt[:, 3] - gt[:, 1] >= 2)]
        gt_h = gt[:, 3] - gt[:, 1]
        for k, (lo, hi) in BUCKETS.items():
            n_gt[k] += int(((gt_h >= lo) & (gt_h < hi)).sum())

        t0 = time.perf_counter()
        boxes, scores = runner(img, score_floor)
        t_infer += time.perf_counter() - t0
        if len(scores):
            keep = nms_np(boxes, scores, nms_thr)
            boxes, scores = boxes[keep], scores[keep]
        order = scores.argsort()[::-1]
        boxes, scores = boxes[order], scores[order]

        matched_gt = np.full(len(boxes), -1, dtype=np.int64)
        if len(gt) and len(boxes):
            ious = iou_matrix(boxes, gt)
            used = np.zeros(len(gt), bool)
            for i in range(len(boxes)):
                j = int(ious[i].argmax())
                if ious[i, j] >= iou_thr and not used[j]:
                    used[j] = True
                    matched_gt[i] = j
        for k, (lo, hi) in BUCKETS.items():
            for i in range(len(boxes)):
                j = matched_gt[i]
                if j < 0:
                    records[k].append((scores[i], 0, 1))
                elif lo <= gt_h[j] < hi:
                    records[k].append((scores[i], 1, 0))
                # matched a GT outside this bucket -> ignored for this bucket
    result = {}
    for k in BUCKETS:
        if records[k]:
            arr = np.array(records[k]); arr = arr[arr[:, 0].argsort()[::-1]]
            result[k] = voc_ap(arr[:, 1], arr[:, 2], n_gt[k])
        else:
            result[k] = 0.0 if n_gt[k] else float("nan")
    return result, n_gt, t_infer / max(len(samples), 1) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--kface", default=None)
    ap.add_argument("--ref", default=None, help="reference ONNX detector (ltrb-distance head format)")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--score-floor", type=float, default=0.02)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    ds = WiderFaceDataset(args.label, args.images)
    samples = ds.samples[: args.max_images] if args.max_images else ds.samples
    print(f"images={len(samples)} size={args.size} score_floor={args.score_floor} nms={args.nms} iou={args.iou}")

    runners = []
    if args.kface:
        runners.append(("K-FACE", KFaceRunner(args.kface, args.size, args.threads)))
    if args.ref:
        runners.append(("REF", RefDetRunner(args.ref, args.size, args.threads)))
    if not runners:
        ap.error("give --kface and/or --ref")

    header = f"{'model':8s}" + "".join(f"{k:>14s}" for k in BUCKETS) + f"{'ms/img':>9s}"
    print(header)
    for name, r in runners:
        res, n_gt, ms = evaluate(r, samples, args.images, args.score_floor, args.nms, args.iou)
        print(f"{name:8s}" + "".join(f"{100 * res[k]:14.2f}" for k in BUCKETS) + f"{ms:9.2f}")
    print("GT count per bucket: " + ", ".join(f"{k}={n_gt[k]}" for k in BUCKETS))


if __name__ == "__main__":
    main()
