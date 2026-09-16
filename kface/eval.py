"""Author: kienkk

Shared WIDER FACE detection metrics: AP@0.5 by GT face-size bucket
(small < 32px, medium 32-96px, large > 96px). Used by
tools/eval_widerface.py and kface/train.py.
"""
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

BUCKETS = {"all": (0, np.inf), "small<32": (0, 32), "medium32-96": (32, 96), "large>96": (96, np.inf)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
VAR = (0.1, 0.2)  # box/kps encode-decode variance, matches kface.utils.box_utils.VARIANCE


def letterbox(img, size, pad_value=114):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((size, size, 3), pad_value, np.uint8)
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


def collect_detections(predict_fn, samples, score_floor=0.02, nms_thr=0.4, progress=True):
    """Run predict_fn once per image; cache (gt_boxes, pred_boxes, pred_scores)
    so multiple AP metrics can be computed without re-running inference."""
    records = []
    t_infer = 0.0
    for img_path, gt, *_ in tqdm(samples, desc="eval", disable=not progress, leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        gt = gt[(gt[:, 2] - gt[:, 0] >= 2) & (gt[:, 3] - gt[:, 1] >= 2)]

        t0 = time.perf_counter()
        boxes, scores = predict_fn(img)
        t_infer += time.perf_counter() - t0
        mask = scores >= score_floor
        boxes, scores = boxes[mask], scores[mask]
        if len(scores):
            keep = nms_np(boxes, scores, nms_thr)
            boxes, scores = boxes[keep], scores[keep]
        order = scores.argsort()[::-1]
        records.append((gt, boxes[order], scores[order]))
    return records, t_infer / max(len(records), 1) * 1000


def collect_detections_batched(model, anchors, samples, size, device,
                                score_floor=0.02, nms_thr=0.4, batch_size=16, progress=True):
    """Same as collect_detections but runs the model forward pass on whole
    batches at once instead of one image per forward pass — the per-image
    Python/CPU overhead otherwise leaves the GPU idle between images and
    dominates wall-clock time. Used for in-training validation."""
    model.eval()
    records = []
    t_infer = 0.0
    n_images = 0

    batches = range(0, len(samples), batch_size)
    for start in tqdm(batches, desc="eval", disable=not progress, leave=False):
        chunk = samples[start:start + batch_size]
        imgs, scales, gts = [], [], []
        for img_path, gt, *_ in chunk:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gt = gt[(gt[:, 2] - gt[:, 0] >= 2) & (gt[:, 3] - gt[:, 1] >= 2)]
            net, scale = letterbox(img, size)
            rgb = cv2.cvtColor(net, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            imgs.append(inp.transpose(2, 0, 1))
            scales.append(scale)
            gts.append(gt)
        if not imgs:
            continue
        n_images += len(imgs)

        batch = torch.from_numpy(np.stack(imgs).astype(np.float32)).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            results = model.predict(batch, anchors=anchors, conf_thresh=score_floor, iou_thresh=nms_thr)
        t_infer += time.perf_counter() - t0

        for (boxes, _kps, scores), scale, gt in zip(results, scales, gts):
            boxes = (boxes / scale).cpu().numpy()
            scores = scores.cpu().numpy()
            order = scores.argsort()[::-1]
            records.append((gt, boxes[order], scores[order]))
    return records, t_infer / max(n_images, 1) * 1000


def ap_at_iou(records, iou_thr, size_range=None):
    """AP at a single IoU threshold, optionally restricted to GT boxes whose
    height falls in size_range=(lo, hi). Detections matched to a GT outside
    the bucket are ignored (neither TP nor FP) so buckets don't penalize
    each other."""
    tp_fp = []
    n_gt = 0
    for gt, boxes, scores in records:
        gt_h = gt[:, 3] - gt[:, 1]
        in_bucket = np.ones(len(gt), dtype=bool) if size_range is None else \
            (gt_h >= size_range[0]) & (gt_h < size_range[1])
        n_gt += int(in_bucket.sum())

        matched_gt = np.full(len(boxes), -1, dtype=np.int64)
        if len(gt) and len(boxes):
            ious = iou_matrix(boxes, gt)
            used = np.zeros(len(gt), dtype=bool)
            for i in range(len(boxes)):
                j = int(ious[i].argmax())
                if ious[i, j] >= iou_thr and not used[j]:
                    used[j] = True
                    matched_gt[i] = j
        for i in range(len(boxes)):
            j = matched_gt[i]
            if j < 0:
                tp_fp.append((scores[i], 0, 1))
            elif in_bucket[j]:
                tp_fp.append((scores[i], 1, 0))

    if not tp_fp:
        return 0.0 if n_gt else float("nan")
    arr = np.array(tp_fp)
    arr = arr[arr[:, 0].argsort()[::-1]]
    return voc_ap(arr[:, 1], arr[:, 2], n_gt)


def _bucket_ap(records, iou_thr):
    result, n_gt = {}, {}
    for k, (lo, hi) in BUCKETS.items():
        result[k] = ap_at_iou(records, iou_thr, None if k == "all" else (lo, hi))
        n_gt[k] = sum(int(((g[:, 3] - g[:, 1] >= lo) & (g[:, 3] - g[:, 1] < hi)).sum()) for g, _, _ in records)
    return result, n_gt


def _full_metrics(records, iou_thrs=None):
    if iou_thrs is None:
        iou_thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)
    ap_per_iou = np.array([ap_at_iou(records, t, None) for t in iou_thrs], dtype=float)
    metrics = {
        "mAP": float(np.nanmean(ap_per_iou)),
        "AP50": float(ap_per_iou[np.argmin(np.abs(iou_thrs - 0.5))]),
        "AP75": float(ap_per_iou[np.argmin(np.abs(iou_thrs - 0.75))]),
    }
    n_gt = {}
    for k, (lo, hi) in BUCKETS.items():
        if k == "all":
            continue
        metrics[k] = ap_at_iou(records, 0.5, (lo, hi))
        n_gt[k] = sum(int(((g[:, 3] - g[:, 1] >= lo) & (g[:, 3] - g[:, 1] < hi)).sum()) for g, _, _ in records)
    return metrics, n_gt


def evaluate(predict_fn, samples, score_floor=0.02, nms_thr=0.4, iou_thr=0.5, progress=True):
    """AP@iou_thr per GT face-size bucket. Returns (AP per bucket, GT count
    per bucket, ms/image)."""
    records, ms = collect_detections(predict_fn, samples, score_floor, nms_thr, progress=progress)
    result, n_gt = _bucket_ap(records, iou_thr)
    return result, n_gt, ms


def evaluate_full(predict_fn, samples, score_floor=0.02, nms_thr=0.4, iou_thrs=None, progress=True):
    """COCO-style report: mAP averaged over iou_thrs (default 0.5:0.95:0.05),
    AP50, AP75, plus per-size AP@0.5 (small/medium/large). Returns (metrics
    dict, GT count per size bucket, ms/image)."""
    records, ms = collect_detections(predict_fn, samples, score_floor, nms_thr, progress=progress)
    metrics, n_gt = _full_metrics(records, iou_thrs)
    return metrics, n_gt, ms


def evaluate_full_batched(model, anchors, samples, size, device, score_floor=0.02,
                          nms_thr=0.4, batch_size=16, iou_thrs=None, progress=True):
    """Same report as evaluate_full(), but batched (see collect_detections_batched)
    — much faster for in-training validation against the live PyTorch model."""
    records, ms = collect_detections_batched(model, anchors, samples, size, device,
                                             score_floor, nms_thr, batch_size, progress=progress)
    metrics, n_gt = _full_metrics(records, iou_thrs)
    return metrics, n_gt, ms
