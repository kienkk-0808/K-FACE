"""Author: kienkk

Training-time augmentation: SSD-style random crop anchored on a face box,
random horizontal flip, photometric distortion, resize+pad to a square input.
Keeps boxes/landmarks consistent through every transform.
"""
import random
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _clip_boxes(boxes, w, h):
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, h)
    return boxes


def random_crop(img, target, min_scale=0.3, max_scale=1.0, tries=50):
    h, w = img.shape[:2]
    boxes = target["boxes"]
    if boxes.shape[0] == 0:
        return img, target

    for _ in range(tries):
        scale = random.uniform(min_scale, max_scale)
        crop_w, crop_h = int(w * scale), int(h * scale)
        if crop_w < 1 or crop_h < 1:
            continue
        x0 = random.randint(0, max(0, w - crop_w))
        y0 = random.randint(0, max(0, h - crop_h))
        crop_box = np.array([x0, y0, x0 + crop_w, y0 + crop_h], dtype=np.float32)

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        keep = (
            (centers[:, 0] >= crop_box[0]) & (centers[:, 0] <= crop_box[2]) &
            (centers[:, 1] >= crop_box[1]) & (centers[:, 1] <= crop_box[3])
        )
        if not keep.any():
            continue

        new_img = img[y0:y0 + crop_h, x0:x0 + crop_w]
        new_boxes = boxes[keep].copy()
        new_boxes[:, [0, 2]] -= x0
        new_boxes[:, [1, 3]] -= y0
        new_boxes = _clip_boxes(new_boxes, crop_w, crop_h)

        new_kps = target["kps"][keep].copy()
        new_kps[..., 0] -= x0
        new_kps[..., 1] -= y0
        new_has_kps = target["has_kps"][keep].copy()

        return new_img, {"boxes": new_boxes, "kps": new_kps, "has_kps": new_has_kps}

    return img, target


def random_flip(img, target, p=0.5):
    if random.random() >= p:
        return img, target
    w = img.shape[1]
    img = img[:, ::-1].copy()
    boxes = target["boxes"].copy()
    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    kps = target["kps"].copy()
    kps[..., 0] = w - kps[..., 0]
    # flipping swaps left/right eye & mouth corners: [leye,reye,nose,lmouth,rmouth] -> [reye,leye,nose,rmouth,lmouth]
    kps = kps[:, [1, 0, 2, 4, 3], :]

    target = {"boxes": boxes, "kps": kps, "has_kps": target["has_kps"]}
    return img, target


def photometric_distort(img):
    img = img.astype(np.float32)
    if random.random() < 0.5:
        img *= random.uniform(0.7, 1.3)  # brightness
    if random.random() < 0.5:
        img = (img - 127.5) * random.uniform(0.8, 1.2) + 127.5  # contrast
    return img.clip(0, 255)


def resize_pad(img, target, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    target["boxes"] = target["boxes"] * scale
    target["kps"] = target["kps"] * scale
    return canvas, target


class TrainTransform:
    def __init__(self, size=320):
        self.size = size

    def __call__(self, img, target):
        img, target = random_crop(img, target)
        img, target = random_flip(img, target)
        img, target = resize_pad(img, target, self.size)
        img = photometric_distort(img)
        img = (img / 255.0 - MEAN) / STD
        return img.astype(np.float32), target


class EvalTransform:
    def __init__(self, size=320):
        self.size = size

    def __call__(self, img, target):
        img, target = resize_pad(img, target, self.size)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return img.astype(np.float32), target
