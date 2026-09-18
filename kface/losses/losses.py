"""Author: kienkk

Loss functions + ATSS label assignment for K-FACE: Focal Loss (cls),
GIoU loss (box), Smooth-L1 (landmarks, positives with landmark labels only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.box_utils import anchors_to_xyxy, box_iou, decode_boxes, encode_kps


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * ce * (1 - p_t) ** self.gamma).sum()


def giou_loss(pred, target, eps=1e-7):
    """pred, target: (N,4) xyxy. Returns sum of (1 - GIoU)."""
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    lt_c = torch.min(pred[:, :2], target[:, :2])
    rb_c = torch.max(pred[:, 2:], target[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1] + eps
    giou = iou - (area_c - union) / area_c
    return (1 - giou).sum()


def atss_assign(anchors, level_sizes, gt_boxes, topk=9):
    """anchors: (A,4) cxcywh; gt_boxes: (G,4) xyxy -> pos_mask (A,), matched_gt (A,)."""
    device = anchors.device
    num_a, num_g = anchors.shape[0], gt_boxes.shape[0]
    a_xyxy = anchors_to_xyxy(anchors)
    ious = box_iou(a_xyxy, gt_boxes)  # (A,G)

    a_c = anchors[:, :2]
    gt_c = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    dist = torch.cdist(a_c, gt_c)  # (A,G)

    cand = []
    start = 0
    for n in level_sizes:
        k = min(topk, n)
        _, idx = dist[start:start + n].topk(k, dim=0, largest=False)
        cand.append(idx + start)
        start += n
    cand = torch.cat(cand, dim=0)  # (K,G)

    cand_ious = ious.gather(0, cand)
    thr = cand_ious.mean(0) + cand_ious.std(0, unbiased=False)
    is_pos = cand_ious >= thr[None]

    cand_c = a_c[cand]  # (K,G,2)
    inside = (
        (cand_c[..., 0] > gt_boxes[:, 0]) & (cand_c[..., 0] < gt_boxes[:, 2]) &
        (cand_c[..., 1] > gt_boxes[:, 1]) & (cand_c[..., 1] < gt_boxes[:, 3])
    )
    is_pos &= inside

    assigned = torch.full((num_a, num_g), -1.0, device=device)
    g_idx = torch.arange(num_g, device=device).expand_as(cand)
    assigned[cand[is_pos], g_idx[is_pos]] = ious[cand[is_pos], g_idx[is_pos]]

    # GTs with no positive: fall back to their best-IoU anchor
    has_pos = (assigned >= 0).any(0)
    if not has_pos.all():
        best_a = ious[:, ~has_pos].argmax(0)
        assigned[best_a, (~has_pos).nonzero(as_tuple=True)[0]] = ious[best_a, (~has_pos).nonzero(as_tuple=True)[0]]

    max_iou, matched_gt = assigned.max(1)
    pos_mask = max_iou >= 0
    return pos_mask, matched_gt


class KFaceLoss(nn.Module):
    def __init__(self, level_sizes, topk=9, box_weight=2.0, kps_weight=1.0):
        super().__init__()
        self.level_sizes = list(level_sizes)
        self.topk = topk
        self.cls_loss = FocalLoss()
        self.box_weight = box_weight
        self.kps_weight = kps_weight

    def forward(self, cls_pred, box_pred, kps_pred, anchors, targets):
        device = cls_pred.device
        zero = torch.zeros((), device=device)
        total_cls, total_box, total_kps = zero, zero, zero
        num_pos_total = 0

        for b, t in enumerate(targets):
            gt_boxes = t["boxes"].to(device)
            num_a = anchors.shape[0]

            if gt_boxes.shape[0] == 0:
                # all-negative image: normalize by anchor count like the
                # positive-image branch normalizes by num_pos below, so a
                # negative image can't dominate the batch loss with an
                # unnormalized sum over every anchor
                cls_t = torch.zeros(num_a, device=device)
                total_cls = total_cls + self.cls_loss(cls_pred[b, :, 0], cls_t) / num_a
                continue

            pos_mask, matched_gt = atss_assign(anchors, self.level_sizes, gt_boxes, self.topk)
            num_pos = int(pos_mask.sum().item())
            num_pos_total += num_pos
            norm = max(num_pos, 1)

            cls_t = pos_mask.float()
            total_cls = total_cls + self.cls_loss(cls_pred[b, :, 0], cls_t) / norm

            if num_pos == 0:
                continue

            pos_anchors = anchors[pos_mask]
            gt_pos = gt_boxes[matched_gt[pos_mask]]
            pred_boxes = decode_boxes(box_pred[b, pos_mask], pos_anchors)
            total_box = total_box + giou_loss(pred_boxes, gt_pos) / norm

            has_kps = t["has_kps"].to(device)[matched_gt[pos_mask]].bool()
            if has_kps.any():
                gt_kps = t["kps"].to(device)[matched_gt[pos_mask]][has_kps]
                kps_t = encode_kps(gt_kps, pos_anchors[has_kps])
                total_kps = total_kps + F.smooth_l1_loss(
                    kps_pred[b, pos_mask][has_kps], kps_t, reduction="sum"
                ) / max(int(has_kps.sum().item()), 1)

        bs = max(len(targets), 1)
        loss_cls = total_cls / bs
        loss_box = self.box_weight * total_box / bs
        loss_kps = self.kps_weight * total_kps / bs
        return {
            "loss": loss_cls + loss_box + loss_kps,
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "loss_kps": loss_kps.detach(),
            "num_pos": num_pos_total,
        }
