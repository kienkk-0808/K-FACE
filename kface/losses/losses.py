"""Loss functions + anchor-target matching for K-FACE.

- Classification: Focal Loss (handles the heavy anchor imbalance without
  needing hard-negative mining tricks).
- Box regression: Smooth L1 on positive anchors only.
- Landmark regression: Smooth L1 on positive anchors that have valid
  landmark annotations (WIDER FACE only labels ~half of the boxes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.box_utils import anchors_to_xyxy, box_iou, encode_boxes, encode_kps


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (loss * alpha_t).sum()


def match_anchors(anchors, gt_boxes, gt_kps, gt_has_kps, pos_iou=0.4, neg_iou=0.3):
    """anchors: (A,4) cxcywh. gt_boxes: (G,4) xyxy. gt_kps: (G,5,2).
    Returns per-anchor: cls target (A,), box target (A,4), kps target (A,10),
    kps valid mask (A,), pos mask (A,).
    """
    device = anchors.device
    a_xyxy = anchors_to_xyxy(anchors)
    num_a = anchors.shape[0]

    if gt_boxes.shape[0] == 0:
        cls_t = torch.zeros(num_a, device=device)
        box_t = torch.zeros(num_a, 4, device=device)
        kps_t = torch.zeros(num_a, 10, device=device)
        kps_valid = torch.zeros(num_a, device=device)
        pos_mask = torch.zeros(num_a, dtype=torch.bool, device=device)
        return cls_t, box_t, kps_t, kps_valid, pos_mask

    ious = box_iou(a_xyxy, gt_boxes)  # (A,G)
    best_iou, best_gt = ious.max(dim=1)

    # ensure every GT gets at least its best-matching anchor (recall for tiny/rare faces)
    best_anchor_per_gt = ious.argmax(dim=0)

    pos_mask = best_iou >= pos_iou
    pos_mask[best_anchor_per_gt] = True
    best_gt[best_anchor_per_gt] = torch.arange(gt_boxes.shape[0], device=device)

    ignore_mask = (best_iou >= neg_iou) & (best_iou < pos_iou)
    ignore_mask[best_anchor_per_gt] = False

    cls_t = torch.zeros(num_a, device=device)
    cls_t[pos_mask] = 1.0
    cls_t[ignore_mask] = -1.0  # sentinel: excluded from loss below

    matched_gt = best_gt
    box_t = torch.zeros(num_a, 4, device=device)
    box_t[pos_mask] = encode_boxes(gt_boxes[matched_gt[pos_mask]], anchors[pos_mask])

    kps_t = torch.zeros(num_a, 10, device=device)
    kps_valid = torch.zeros(num_a, device=device)
    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
    if pos_idx.numel() > 0:
        g_idx = matched_gt[pos_idx]
        has_kps = gt_has_kps[g_idx].bool()
        if has_kps.any():
            sub = pos_idx[has_kps]
            kps_t[sub] = encode_kps(gt_kps[g_idx[has_kps]], anchors[sub])
            kps_valid[sub] = 1.0

    return cls_t, box_t, kps_t, kps_valid, pos_mask


class KFaceLoss(nn.Module):
    def __init__(self, pos_iou=0.4, neg_iou=0.3, box_weight=2.0, kps_weight=1.0):
        super().__init__()
        self.cls_loss = FocalLoss()
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.box_weight = box_weight
        self.kps_weight = kps_weight

    def forward(self, cls_pred, box_pred, kps_pred, anchors, targets):
        """cls_pred:(B,A,1) box_pred:(B,A,4) kps_pred:(B,A,10) anchors:(A,4)
        targets: list of dicts {boxes:(G,4) xyxy, kps:(G,5,2), has_kps:(G,)}
        """
        device = cls_pred.device
        total_cls, total_box, total_kps = 0.0, 0.0, 0.0
        num_pos_total = 0

        for b, t in enumerate(targets):
            cls_t, box_t, kps_t, kps_valid, pos_mask = match_anchors(
                anchors, t["boxes"].to(device), t["kps"].to(device), t["has_kps"].to(device),
                pos_iou=self.pos_iou, neg_iou=self.neg_iou,
            )
            valid = cls_t >= 0  # drop ignored (grey-zone) anchors from cls loss
            num_pos = pos_mask.sum().clamp(min=1)
            num_pos_total += int(num_pos.item())

            total_cls = total_cls + self.cls_loss(cls_pred[b, valid, 0], cls_t[valid]) / num_pos

            if pos_mask.any():
                total_box = total_box + F.smooth_l1_loss(
                    box_pred[b, pos_mask], box_t[pos_mask], reduction="sum"
                ) / num_pos

                kps_mask = pos_mask & (kps_valid > 0)
                if kps_mask.any():
                    total_kps = total_kps + F.smooth_l1_loss(
                        kps_pred[b, kps_mask], kps_t[kps_mask], reduction="sum"
                    ) / kps_mask.sum().clamp(min=1)

        bs = max(len(targets), 1)
        loss_cls = total_cls / bs
        loss_box = self.box_weight * total_box / bs
        loss_kps = self.kps_weight * total_kps / bs
        loss = loss_cls + loss_box + loss_kps
        return {
            "loss": loss,
            "loss_cls": loss_cls.detach() if torch.is_tensor(loss_cls) else torch.tensor(loss_cls),
            "loss_box": loss_box.detach() if torch.is_tensor(loss_box) else torch.tensor(loss_box),
            "loss_kps": loss_kps.detach() if torch.is_tensor(loss_kps) else torch.tensor(loss_kps),
            "num_pos": num_pos_total,
        }
