"""Author: kienkk

Training-time augmentation: random crop anchored on a face box, flip,
photometric distortion, resize+pad to a square input.
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


def face_anchor_crop(img, target, margin_range=(0.1, 0.6)):
    """Crop tightly around one randomly chosen face (with a random margin),
    so that face ends up filling most of the training image. WIDER FACE is
    almost entirely mid/long-distance event photography — only ~0.1% of
    faces fill more than 70% of their image — so plain area-based random_crop
    essentially never produces a close-up example. This covers that end of
    the scale spectrum (e.g. webcam/selfie-distance use cases)."""
    boxes = target["boxes"]
    if boxes.shape[0] == 0:
        return img, target
    idx = random.randrange(boxes.shape[0])
    x1, y1, x2, y2 = boxes[idx]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return img, target

    margin = random.uniform(*margin_range)
    h, w = img.shape[:2]
    cx1 = max(0, int(x1 - margin * bw))
    cy1 = max(0, int(y1 - margin * bh))
    cx2 = min(w, int(x2 + margin * bw))
    cy2 = min(h, int(y2 + margin * bh))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return img, target

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep = (
        (centers[:, 0] >= cx1) & (centers[:, 0] <= cx2) &
        (centers[:, 1] >= cy1) & (centers[:, 1] <= cy2)
    )
    new_img = img[cy1:cy2, cx1:cx2]
    new_boxes = boxes[keep].copy()
    new_boxes[:, [0, 2]] -= cx1
    new_boxes[:, [1, 3]] -= cy1
    new_boxes = _clip_boxes(new_boxes, cx2 - cx1, cy2 - cy1)

    new_kps = target["kps"][keep].copy()
    new_kps[..., 0] -= cx1
    new_kps[..., 1] -= cy1

    return new_img, {"boxes": new_boxes, "kps": new_kps, "has_kps": target["has_kps"][keep]}


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
    img = img.clip(0, 255)

    if random.random() < 0.5:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-18, 18)) % 180  # hue
        hsv[..., 1] *= random.uniform(0.6, 1.4)  # saturation
        hsv[..., 1] = hsv[..., 1].clip(0, 255)
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if random.random() < 0.1:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    if random.random() < 0.3:
        img = img + np.random.normal(0, random.uniform(3, 15), img.shape).astype(np.float32)  # sensor/low-light noise

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
    """Returns a uint8 HWC image (0-255); normalization happens on-device
    in kface.train so DataLoader prefetch buffers stay small.
    """

    def __init__(self, size=320, anchor_crop_p=0.25):
        self.size = size
        self.anchor_crop_p = anchor_crop_p

    def __call__(self, img, target):
        if random.random() < self.anchor_crop_p:
            img, target = face_anchor_crop(img, target)
        else:
            img, target = random_crop(img, target)
        img, target = random_flip(img, target)
        img, target = resize_pad(img, target, self.size)
        img = photometric_distort(img)
        return img.astype(np.uint8), target


class EvalTransform:
    def __init__(self, size=320):
        self.size = size

    def __call__(self, img, target):
        img, target = resize_pad(img, target, self.size)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return img.astype(np.float32), target
