"""Author: kienkk

WIDER FACE dataset loader. Auto-detects three label formats:
1. RetinaFace (train, has landmarks): "# path" then "x y w h lm1x lm1y v1
   ... lm5x lm5y v5 score" rows; -1 landmarks mean "no landmark for this box".
2. Official bbx_gt (val, box-only): "path", then a face-count line, then
   that many "x y w h blur expr illum invalid occl pose" rows. Count 0 is
   followed by one dummy all-zero row, which is discarded.
3. Plain image list (e.g. wider_val.txt): one path per line, no boxes.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _try_floats(s):
    try:
        return [float(t) for t in s.split()]
    except ValueError:
        return None


class WiderFaceDataset(Dataset):
    def __init__(self, label_path, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []  # list of (img_path, boxes(N,4) xyxy, kps(N,5,2), has_kps(N,))
        self._parse(label_path)

    def _parse(self, label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        i, n = 0, len(lines)
        while i < n:
            body = lines[i][1:].strip() if lines[i].startswith("#") else lines[i]
            i += 1
            img_path = os.path.join(self.image_root, body)
            boxes, kps, has_kps = [], [], []

            # Format 2: bare integer count line right after the path.
            count_vals = _try_floats(lines[i]) if i < n else None
            if count_vals is not None and len(count_vals) == 1:
                count = int(count_vals[0])
                i += 1
                for _ in range(count if count > 0 else 1):  # 0-face dummy row quirk
                    if i >= n:
                        break
                    row = _try_floats(lines[i])
                    i += 1
                    if row is None:
                        break
                    if count > 0 and len(row) >= 4:
                        x, y, w, h = row[0:4]
                        boxes.append([x, y, x + w, y + h])
                        kps.append(np.zeros((5, 2), dtype=np.float32))
                        has_kps.append(0.0)
                self._append(img_path, boxes, kps, has_kps)
                continue

            # Format 1: box(+landmark) rows follow directly, each with >=4
            # fields, until the next path line (format 2's count line is
            # unambiguous vs. this since a box row always has >=4 fields).
            while i < n:
                row = _try_floats(lines[i])
                if row is None or len(row) < 4:
                    break
                i += 1
                x, y, w, h = row[0:4]
                boxes.append([x, y, x + w, y + h])
                if len(row) >= 19 and row[4] >= 0:
                    pts = np.array(row[4:19], dtype=np.float32).reshape(5, 3)[:, :2]
                    kps.append(pts)
                    has_kps.append(1.0)
                else:
                    kps.append(np.zeros((5, 2), dtype=np.float32))
                    has_kps.append(0.0)
            # Format 3 (plain list): the loop above never runs (next line
            # isn't a numeric row), so boxes/kps/has_kps stay empty.
            self._append(img_path, boxes, kps, has_kps)

    def _append(self, img_path, boxes, kps, has_kps):
        self.samples.append((
            img_path,
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(kps, dtype=np.float32).reshape(-1, 5, 2),
            np.array(has_kps, dtype=np.float32),
        ))

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

        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
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
