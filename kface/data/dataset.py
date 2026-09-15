"""Author: kienkk

WIDER FACE dataset loader, RetinaFace-style label format:

    # 0--Parade/0_Parade_marchingband_1_849.jpg
    449 330 122 149 488.9 373.6 0.0 542.1 376.4 0.0 515.0 412.8 0.0 485.2 425.9 0.0 538.4 431.5 0.0 0.82
    <x> <y> <w> <h> <lm1x> <lm1y> <v1> ... <lm5x> <lm5y> <v5> <blur/score>

Faces without landmark annotations use -1 for all landmark fields; those
boxes still train cls+box but are masked out of the landmark loss.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class WiderFaceDataset(Dataset):
    def __init__(self, label_path, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []  # list of (img_path, boxes(N,4) xyxy, kps(N,5,2), has_kps(N,))
        self._parse(label_path)

    def _parse(self, label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        img_path = None
        boxes, kps, has_kps = [], [], []

        def flush():
            if img_path is not None:
                self.samples.append((
                    img_path,
                    np.array(boxes, dtype=np.float32).reshape(-1, 4),
                    np.array(kps, dtype=np.float32).reshape(-1, 5, 2),
                    np.array(has_kps, dtype=np.float32),
                ))

        for line in lines:
            if line.startswith("#"):
                flush()
                img_path = os.path.join(self.image_root, line[1:].strip())
                boxes, kps, has_kps = [], [], []
                continue
            vals = list(map(float, line.split()))
            x, y, w, h = vals[0:4]
            boxes.append([x, y, x + w, y + h])
            if len(vals) >= 19 and vals[4] >= 0:
                pts = np.array(vals[4:19], dtype=np.float32).reshape(5, 3)[:, :2]
                kps.append(pts)
                has_kps.append(1.0)
            else:
                kps.append(np.zeros((5, 2), dtype=np.float32))
                has_kps.append(0.0)
        flush()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes, kpts, has_kps = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = {
            "boxes": boxes.copy(),
            "kps": kpts.copy(),
            "has_kps": has_kps.copy(),
        }
        if self.transform is not None:
            img, target = self.transform(img, target)

        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        target = {
            "boxes": torch.from_numpy(target["boxes"].astype(np.float32)),
            "kps": torch.from_numpy(target["kps"].astype(np.float32)),
            "has_kps": torch.from_numpy(target["has_kps"].astype(np.float32)),
        }
        return img, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
