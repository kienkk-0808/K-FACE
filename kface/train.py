"""Author: kienkk

Training entrypoint for K-FACE.

Usage:
    python -m kface.train --config configs/kface_n.yaml [--resume ckpt.pth]
"""
import argparse
import copy
import math
import os

import torch
import yaml
from torch.utils.data import DataLoader

from kface.models.detector import build_model
from kface.data.dataset import WiderFaceDataset, collate_fn
from kface.data.augment import TrainTransform
from kface.losses.losses import KFaceLoss
from kface.utils.box_utils import generate_anchors


class ModelEMA:
    """Exponential moving average of weights — evaluated/exported instead of
    the raw weights; a consistent +0.5-1 AP for detectors at no inference cost."""

    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))  # ramp-up early
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg, tcfg = cfg["model"], cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tcfg["input_size"]
    strides = tuple(mcfg["strides"])

    use_amp = device.type == "cuda"
    # bf16 has the same exponent range as fp32 (no gradient underflow/overflow),
    # so it needs no GradScaler and is what Ada-generation GPUs (e.g. L4) run
    # tensor-core matmuls/convs at natively — cheaper and simpler than fp16.
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    channels_last = tcfg.get("channels_last", True) and use_amp

    if use_amp:
        # TF32 speeds up the fp32 fallback paths (BN stats, loss) for free on
        # Ampere+/Ada GPUs; cudnn.benchmark autotunes conv algorithms since
        # every batch here has the same fixed input shape.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataset = WiderFaceDataset(cfg["data"]["train_label"], cfg["data"]["train_images"],
                               transform=TrainTransform(size=size))
    total_boxes = sum(len(s[1]) for s in dataset.samples)
    if total_boxes == 0:
        raise ValueError(
            f"'{cfg['data']['train_label']}' has zero ground-truth boxes across "
            f"{len(dataset.samples)} images — looks like a plain image list "
            "(e.g. wider_val.txt), not an annotated label.txt. Training on it "
            "would silently converge to 'always predict no face' with no error. "
            "Point train_label at the annotated RetinaFace-format label.txt instead."
        )
    print(f"train images={len(dataset.samples)} boxes={total_boxes} "
          f"(avg {total_boxes / len(dataset.samples):.1f}/image)")
    num_workers = tcfg["num_workers"]
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=num_workers, collate_fn=collate_fn,
                        drop_last=True, pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0,
                        prefetch_factor=tcfg.get("prefetch_factor", 4) if num_workers > 0 else None)

    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    ema = ModelEMA(model)

    anchors = generate_anchors(size, strides=strides,
                               scales_per_stride=tuple(tuple(s) for s in mcfg["scales_per_stride"])).to(device)
    level_sizes = [(size // s) ** 2 * mcfg["num_anchors"] for s in strides]
    criterion = KFaceLoss(level_sizes, topk=tcfg.get("atss_topk", 9),
                          box_weight=tcfg["box_weight"], kps_weight=tcfg["kps_weight"])

    optimizer = torch.optim.SGD(model.parameters(), lr=tcfg["lr"], momentum=0.9,
                                weight_decay=tcfg["weight_decay"], nesterov=True)
    epochs = tcfg["epochs"]
    iters_per_epoch = len(loader)
    warmup_iters = tcfg.get("warmup_iters", 500)
    total_iters = epochs * iters_per_epoch

    def lr_lambda(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        p = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
            ema.updates = ckpt.get("ema_updates", 0)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1

    os.makedirs(tcfg["output_dir"], exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "loss_kps": 0.0}
        for step, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                cls_outs, box_outs, kps_outs = model(imgs)
            cls, box, kps = model.flatten_all(cls_outs, box_outs, kps_outs)
            losses = criterion(cls.float(), box.float(), kps.float(), anchors, targets)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            scheduler.step()
            ema.update(model)

            for k in running:
                running[k] += float(losses[k].detach())
            if step % tcfg["log_interval"] == 0:
                n = step + 1
                msg = " ".join(f"{k}={running[k] / n:.4f}" for k in running)
                print(f"[epoch {epoch}][{step}/{iters_per_epoch}] {msg} "
                      f"num_pos={losses['num_pos']} lr={scheduler.get_last_lr()[0]:.5f}")

        ckpt_path = os.path.join(tcfg["output_dir"], f"epoch_{epoch}.pth")
        torch.save({
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "ema_updates": ema.updates,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
