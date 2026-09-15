"""ONNXRuntime CPU inference for an exported K-FACE model: draws boxes +
5-point landmarks, using kface.utils.box_utils for anchor decoding and NMS.

Usage:
    python tools/infer.py --model kface_n.onnx --image test.jpg --size 320
"""
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch

from kface.utils.box_utils import generate_anchors, decode_boxes, decode_kps, nms

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STRIDES = (8, 16, 32)
SCALES = ((16, 32), (64, 128), (256, 512))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--out", default="result.jpg")
    return ap.parse_args()


def letterbox(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


def preprocess(img, size):
    net_img, scale = letterbox(img, size)
    rgb = cv2.cvtColor(net_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    inp = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    return inp, scale


def main():
    args = parse_args()
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    inp, scale = preprocess(img, args.size)
    out_names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(out_names, sess.run(out_names, {"input": inp})))

    anchors = generate_anchors(args.size, strides=STRIDES, scales_per_stride=SCALES)

    cls_list, box_list, kps_list = [], [], []
    for s in STRIDES:
        cls = outs[f"s{s}_cls"]   # (1, A*1, H, W)
        box = outs[f"s{s}_box"]   # (1, A*4, H, W)
        kps = outs[f"s{s}_kps"]   # (1, A*10, H, W)

        def flat(t, last_dim):
            t = torch.from_numpy(t)
            b, c, h, w = t.shape
            a = c // last_dim
            t = t.view(b, a, last_dim, h, w).permute(0, 3, 4, 1, 2).contiguous()
            return t.view(b, h * w * a, last_dim)

        cls_list.append(flat(cls, 1))
        box_list.append(flat(box, 4))
        kps_list.append(flat(kps, 10))

    cls = torch.cat(cls_list, dim=1)[0, :, 0]
    box = torch.cat(box_list, dim=1)[0]
    kps = torch.cat(kps_list, dim=1)[0]

    mask = cls >= args.conf
    if mask.sum() == 0:
        print("No face detected.")
        return

    a = anchors[mask]
    boxes = decode_boxes(box[mask], a)
    landm = decode_kps(kps[mask], a)
    scores = cls[mask]

    keep = nms(boxes, scores, iou_thresh=args.iou)
    boxes, landm, scores = boxes[keep], landm[keep], scores[keep]

    vis = img.copy()
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = (boxes[i] / scale).tolist()
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(vis, f"{scores[i]:.2f}", (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for px, py in (landm[i] / scale).tolist():
            cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)

    cv2.imwrite(args.out, vis)
    print(f"Detected {boxes.shape[0]} face(s). Saved: {args.out}")


if __name__ == "__main__":
    main()
