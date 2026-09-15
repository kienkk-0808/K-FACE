# K-FACE

Thư viện train model face detection + 5-point landmark (kps), kiến trúc nhẹ,
mục tiêu **>60 FPS trên CPU** (ONNXRuntime, fp32, input 320x320) với độ chính
xác cao nhất có thể trong ngân sách tốc độ đó.

## Kiến trúc

- Backbone **depthwise-separable conv** (MobileNet-style), 3 stage output
  (stride 8/16/32) → FLOPs thấp trên CPU.
- Neck là **PAFPN rút gọn** (top-down + 1 bottom-up pass), giữ đủ ngữ cảnh
  đa tỉ lệ cho mặt nhỏ mà không tốn nhiều compute ở exact resolution cao.
- Head **shared-stem giữa 3 stride** để giảm tham số, có `Scale` học được
  theo từng stride bù cho việc share trọng số.
- Anchor-based, 2 anchor/vị trí, xuất cls + box + 5 điểm landmark (kps).
- Có 2 config sẵn: `kface_n` (nano, ưu tiên FPS) và `kface_s` (small, ưu tiên
  accuracy) — đổi `width_mult`/`depth_mult`/`input_size` để tinh chỉnh
  trade-off mà không cần sửa code.

## Cấu trúc thư viện

```
kface/
  models/
    backbone.py    # depthwise-separable backbone, 3 stage output (stride 8/16/32)
    neck.py         # PAFPN rút gọn
    head.py         # shared-stem head: cls + box + kps (5 điểm)
    detector.py     # ghép model, flatten output, predict() (decode+NMS) khi infer bằng Python
  data/
    dataset.py      # WIDER FACE (định dạng RetinaFace label.txt, có landmark)
    augment.py       # random crop theo box mặt, flip, color jitter, resize+pad
  losses/
    losses.py       # Focal loss (cls) + Smooth L1 (box, kps) + anchor matching
  utils/
    box_utils.py    # sinh anchor, encode/decode box+kps, IoU, NMS
  train.py           # training loop (SGD + cosine LR)
tools/
  export_onnx.py     # export checkpoint -> ONNX (raw cls/box/kps, decode+NMS ở Python)
  infer.py            # infer ảnh bằng ONNXRuntime, vẽ box + 5 điểm landmark
  benchmark.py        # đo FPS thực tế trên CPU (ONNXRuntime)
configs/
  kface_n.yaml         # nano — mục tiêu CPU >60 FPS
  kface_s.yaml         # small — ưu tiên accuracy hơn FPS
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Tải WIDER FACE + nhãn landmark theo định dạng RetinaFace
(https://github.com/biubug6/Pytorch_Retinaface — file `label.txt` chuẩn),
đặt vào:

```
data/widerface/train/images/...
data/widerface/train/label.txt
data/widerface/val/images/...
data/widerface/val/label.txt
```

## Train

```bash
python -m kface.train --config configs/kface_n.yaml
```

Checkpoint lưu ở `runs/kface_n/epoch_XXX.pth`. Resume: thêm `--resume runs/kface_n/epoch_50.pth`.

## Export ONNX

```bash
python tools/export_onnx.py --ckpt runs/kface_n/epoch_299.pth --out kface_n.onnx --size 320
```

## Infer thử

```bash
python tools/infer.py --model kface_n.onnx --image test.jpg --size 320
```

## Đo FPS thực tế trên CPU

```bash
python tools/benchmark.py --model kface_n.onnx --size 320 --threads 4
```

`configs/kface_n.yaml` (width_mult=0.75, neck=48 kênh, input 320) là điểm khởi
đầu để đạt >60 FPS trên CPU đa nhân hiện đại; nếu máy đích yếu hơn, giảm tiếp
`input_size` (VD 256) hoặc `width_mult` (VD 0.5) rồi train lại — kiến trúc
không đổi, chỉ scale nhẹ.

## Ghi chú độ chính xác vs tốc độ

- Tăng `neck_channels`, `width_mult`, `depth_mult`, hoặc `input_size` → tăng
  accuracy, giảm FPS. Dùng `kface_s.yaml` làm mốc tham chiếu accuracy cao hơn.
- Anchor scale trong config (`scales_per_stride`) nên chỉnh theo phân bố kích
  thước mặt của tập dữ liệu/triển khai thực tế (mặt nhỏ nhiều → thêm scale nhỏ
  ở stride 8).
- Loss dùng Focal Loss cho cls (không cần OHEM thủ công) + Smooth L1 cho box
  và kps; landmark chỉ tính loss trên các anchor dương có nhãn landmark hợp lệ
  (WIDER FACE không phải box nào cũng có landmark).
