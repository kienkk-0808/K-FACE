"""Anchor generation, box/landmark encode-decode, IoU, and NMS.

Encoding uses a center-offset + log-scale scheme for boxes, and center-offset
only for the 5 landmark points, with variance terms so the regression targets
stay in a well-conditioned range.
"""
import numpy as np
import torch

VARIANCE = (0.1, 0.2)


def generate_anchors(input_size, strides=(8, 16, 32), scales_per_stride=((16, 32), (64, 128), (256, 512))):
    """Return anchors as (cx, cy, w, h) in pixel units, one set per stride
    concatenated in row-major (y, x, anchor) order — matches how the head's
    conv output is reshaped/flattened.
    """
    if isinstance(input_size, int):
        in_h = in_w = input_size
    else:
        in_h, in_w = input_size

    all_anchors = []
    for stride, scales in zip(strides, scales_per_stride):
        fh, fw = in_h // stride, in_w // stride
        cy = (np.arange(fh, dtype=np.float32) + 0.5) * stride
        cx = (np.arange(fw, dtype=np.float32) + 0.5) * stride
        cx, cy = np.meshgrid(cx, cy)  # (fh, fw)
        for s in scales:
            a = np.stack([cx, cy, np.full_like(cx, s), np.full_like(cy, s)], axis=-1)
            all_anchors.append(a.reshape(-1, 4))
    # interleave per-location per-anchor order to match head output layout:
    # for each stride, output is [fh*fw, num_anchors, 4] flattened -> we build
    # that layout directly instead of concatenating per-scale blocks.
    anchors = []
    idx = 0
    for stride, scales in zip(strides, scales_per_stride):
        fh, fw = in_h // stride, in_w // stride
        cy = (np.arange(fh, dtype=np.float32) + 0.5) * stride
        cx = (np.arange(fw, dtype=np.float32) + 0.5) * stride
        gx, gy = np.meshgrid(cx, cy)
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)
        per_loc = np.stack(
            [np.stack([gx, gy, np.full_like(gx, s), np.full_like(gy, s)], axis=-1) for s in scales],
            axis=1,
        )  # (fh*fw, num_anchors, 4)
        anchors.append(per_loc.reshape(-1, 4))
    anchors = np.concatenate(anchors, axis=0)
    return torch.from_numpy(anchors.astype(np.float32))


def box_iou(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def anchors_to_xyxy(anchors):
    cx, cy, w, h = anchors.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def encode_boxes(gt_xyxy, anchors):
    """gt_xyxy, anchors matched 1:1 -> (dx, dy, dw, dh)."""
    a_cx, a_cy, a_w, a_h = anchors.unbind(-1)
    g_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2
    g_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
    g_w = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=1e-6)
    g_h = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=1e-6)

    dx = (g_cx - a_cx) / a_w / VARIANCE[0]
    dy = (g_cy - a_cy) / a_h / VARIANCE[0]
    dw = torch.log(g_w / a_w) / VARIANCE[1]
    dh = torch.log(g_h / a_h) / VARIANCE[1]
    return torch.stack([dx, dy, dw, dh], dim=-1)


def decode_boxes(deltas, anchors):
    a_cx, a_cy, a_w, a_h = anchors.unbind(-1)
    dx, dy, dw, dh = deltas.unbind(-1)
    cx = dx * VARIANCE[0] * a_w + a_cx
    cy = dy * VARIANCE[0] * a_h + a_cy
    w = torch.exp((dw * VARIANCE[1]).clamp(max=8.0)) * a_w
    h = torch.exp((dh * VARIANCE[1]).clamp(max=8.0)) * a_h
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def encode_kps(gt_kps, anchors):
    """gt_kps: (N,5,2) pixel coords -> (N,10) offsets, variance-normalized."""
    a_cx, a_cy, a_w, a_h = anchors.unbind(-1)
    dx = (gt_kps[..., 0] - a_cx[:, None]) / a_w[:, None] / VARIANCE[0]
    dy = (gt_kps[..., 1] - a_cy[:, None]) / a_h[:, None] / VARIANCE[0]
    out = torch.stack([dx, dy], dim=-1)  # (N,5,2)
    return out.reshape(gt_kps.shape[0], -1)


def decode_kps(deltas, anchors):
    a_cx, a_cy, a_w, a_h = anchors.unbind(-1)
    deltas = deltas.reshape(deltas.shape[0], 5, 2)
    x = deltas[..., 0] * VARIANCE[0] * a_w[:, None] + a_cx[:, None]
    y = deltas[..., 1] * VARIANCE[0] * a_h[:, None] + a_cy[:, None]
    return torch.stack([x, y], dim=-1)  # (N,5,2)


def nms(boxes, scores, iou_thresh=0.4):
    """Simple greedy NMS, numpy-friendly, used at inference/eval time."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long)
