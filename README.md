# K-FACE

Thư viện huấn luyện model phát hiện khuôn mặt + 5 điểm landmark (kps), kiến
trúc nhẹ, tối ưu cho suy luận trên CPU với tốc độ thời gian thực và độ chính
xác cao.

**Tác giả:** kienkk

## Tính năng

- Kiến trúc anchor-based, backbone lai (depthwise-separable ở tầng phân
  giải cao + dense residual ở tầng sâu) để tối ưu tốc độ CPU thực tế, không
  chỉ tối ưu FLOPs trên giấy.
- Đầu ra: bounding box + 5 điểm landmark khuôn mặt (mắt trái/phải, mũi, khóe
  miệng trái/phải).
- Head dùng chung trọng số giữa 3 mức FPN (stride 8/16/32) để giảm tham số.
- Toàn bộ kiến trúc (số kênh, số block, loại block) cấu hình qua file YAML,
  không cần sửa code để đổi cấu hình nhẹ/nặng.
- Pipeline train đầy đủ: ATSS label assignment, Focal Loss + GIoU Loss,
  EMA weights, warm-up + cosine LR, augmentation (random crop theo khuôn
  mặt, flip, photometric).
- Export ONNX, benchmark tốc độ và đánh giá AP@0.5 trên WIDER FACE có sẵn.

## Cấu trúc thư viện

```
kface/
  models/backbone.py   # backbone cấu hình theo stage (dw / inverted-residual / dense block)
  models/neck.py        # FPN top-down
  models/head.py        # head tách cls/reg, share trọng số giữa các stride
  models/detector.py     # ghép model, flatten output, predict(), build_model(cfg)
  data/dataset.py       # WIDER FACE label.txt (định dạng RetinaFace, có landmark)
  data/augment.py       # augmentation lúc train
  losses/losses.py      # ATSS + Focal + GIoU + Smooth-L1 kps
  utils/box_utils.py    # anchor, encode/decode, IoU, NMS
  train.py              # training loop
tools/
  export_onnx.py        # checkpoint (EMA) → ONNX
  infer.py               # infer ảnh bằng ONNXRuntime, vẽ box + 5 landmark
  benchmark.py           # đo tốc độ suy luận trên CPU
  eval_widerface.py      # đánh giá AP@0.5 trên WIDER FACE
configs/
  kface_n.yaml           # cấu hình nano — ưu tiên tốc độ
  kface_s.yaml           # cấu hình small — ưu tiên độ chính xác
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Dữ liệu WIDER FACE với nhãn landmark theo định dạng RetinaFace (`label.txt`):

```
data/widerface/train/images/...
data/widerface/train/label.txt
data/widerface/val/images/...
data/widerface/val/label.txt
```

## Train

```bash
python -m kface.train --config configs/kface_n.yaml
python -m kface.train --config configs/kface_n.yaml --resume runs/kface_n/epoch_50.pth
```

## Export ONNX

```bash
python tools/export_onnx.py --ckpt runs/kface_n/epoch_299.pth --out kface_n.onnx --size 320
```

## Infer

```bash
python tools/infer.py --model kface_n.onnx --image test.jpg --size 320
```

## Đo tốc độ

```bash
python tools/benchmark.py --model kface_n.onnx --size 320 --threads 4
```

## Đánh giá độ chính xác (WIDER FACE)

```bash
python tools/eval_widerface.py --label data/widerface/val/label.txt \
    --images data/widerface/val/images --size 320 --kface kface_n.onnx
```

In AP@0.5 tổng và theo cỡ khuôn mặt (small <32px, medium 32–96px, large >96px).

## Cấu hình có sẵn

| Config | Đặc điểm |
|---|---|
| `kface_n.yaml` | Nhẹ, ưu tiên tốc độ suy luận trên CPU |
| `kface_s.yaml` | Nhiều tham số hơn, ưu tiên độ chính xác |

Đổi `stages`, `neck_channels`, `head_stem_blocks` trong YAML để tinh chỉnh
trade-off tốc độ/độ chính xác theo thiết bị triển khai — không cần sửa code.

## Ghi chú

- Nếu thiết bị đích yếu hơn: giảm số block ở stage cuối (stride 32) trước,
  sau đó đến `neck_channels`.
- Tăng độ chính xác mà không đổi tốc độ suy luận: train ở `input_size` lớn
  hơn, tăng số epoch, thêm augmentation mạnh hơn trong `augment.py`.
