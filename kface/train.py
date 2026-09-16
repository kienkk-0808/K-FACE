"""Author: kienkk

Training entrypoint for K-FACE.

Usage:
    python -m kface.train --config configs/kface_n.yaml [--resume ckpt.pth]

Validation / best-checkpoint / early stopping activate automatically when
data.val_label + data.val_images exist: every train.eval_interval epochs,
AP@0.5 is measured on the EMA weights (kface/eval.py) and best.pth is
(re)saved on improvement; training stops early after
train.early_stop_patience evaluations with no gain. last.pth is always
the most recent epoch, for --resume.
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
from kface.data.augment import TrainTransform, MEAN, STD
from kface.losses.losses import KFaceLoss
from kface.utils.box_utils import generate_anchors
from kface.eval import TorchValRunner, evaluate_full


class ModelEMA:
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


def make_checkpoint(model, ema, optimizer, scheduler, epoch, cfg, best_metric, best_res, no_improve):
    return {
        "model": model.state_dict(),
        "ema": ema.ema.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "cfg": cfg,
        "best_metric": best_metric,
        "best_res": best_res,
        "no_improve": no_improve,
    }


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg, tcfg, dcfg = cfg["model"], cfg["train"], cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tcfg["input_size"]
    strides = tuple(mcfg["strides"])

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    channels_last = tcfg.get("channels_last", True) and use_amp

    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataset = WiderFaceDataset(dcfg["train_label"], dcfg["train_images"], transform=TrainTransform(size=size))
    total_boxes = sum(len(s[1]) for s in dataset.samples)
    if total_boxes == 0:
        raise ValueError(
            f"'{dcfg['train_label']}' has zero ground-truth boxes across "
            f"{len(dataset.samples)} images — looks like a plain image list "
            "(e.g. wider_val.txt), not an annotated label.txt. Training on it "
            "would silently converge to 'always predict no face' with no error. "
            "Point train_label at the annotated RetinaFace-format label.txt instead."
        )
    print(f"train images={len(dataset.samples)} boxes={total_boxes} "
          f"(avg {total_boxes / len(dataset.samples):.1f}/image)")

    val_label, val_images = dcfg.get("val_label"), dcfg.get("val_images")
    val_samples = None
    if val_label and val_images and os.path.exists(val_label):
        val_ds = WiderFaceDataset(val_label, val_images)
        val_boxes = sum(len(s[1]) for s in val_ds.samples)
        if val_boxes == 0:
            print(f"WARNING: '{val_label}' has zero ground-truth boxes (plain image "
                  "list?) — skipping in-training validation / early stopping.")
        else:
            eval_max = tcfg.get("eval_max_images", 500)
            val_samples = val_ds.samples[:eval_max] if eval_max else val_ds.samples
            print(f"val images={len(val_samples)} boxes={sum(len(s[1]) for s in val_samples)} "
                  f"(evaluating every {tcfg.get('eval_interval', 10)} epochs)")
    else:
        print("No val_label/val_images configured or file missing — "
              "in-training validation, best-checkpoint tracking and early "
              "stopping are all disabled; every epoch is saved as-is.")

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

    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)

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
    best_metric = -1.0
    best_res = None
    no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.ema.load_state_dict(ckpt["ema"])
            ema.updates = ckpt.get("ema_updates", 0)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -1.0)
        best_res = ckpt.get("best_res")
        no_improve = ckpt.get("no_improve", 0)

    output_dir = tcfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    eval_interval = tcfg.get("eval_interval", 10)
    eval_metric_key = tcfg.get("early_stop_metric", "AP50")
    early_stop_patience = tcfg.get("early_stop_patience", 0)  # 0 = disabled
    save_every = tcfg.get("save_every", 1)

    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "loss_kps": 0.0}
        for step, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            imgs = imgs.float().div_(255.0).sub_(mean_t).div_(std_t)

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

        stop_early = False
        if val_samples is not None and (epoch + 1) % eval_interval == 0:
            runner = TorchValRunner(ema.ema, anchors, size, device,
                                    conf_thresh=tcfg.get("eval_conf", 0.02),
                                    nms_thresh=tcfg.get("eval_nms", 0.4))
            res, n_gt, ms = evaluate_full(runner, val_samples, tcfg.get("eval_conf", 0.02),
                                          tcfg.get("eval_nms", 0.4))
            metric = res[eval_metric_key]
            order = ["mAP", "AP50", "AP75", "small<32", "medium32-96", "large>96"]
            print(f"[epoch {epoch}] val: " +
                  " ".join(f"{k}={100 * res[k]:.2f}" for k in order) +
                  f"  ({ms:.1f} ms/img)")

            if metric > best_metric:
                best_metric = metric
                best_res = res
                no_improve = 0
                torch.save(make_checkpoint(model, ema, optimizer, scheduler, epoch, cfg,
                                           best_metric, best_res, no_improve),
                          os.path.join(output_dir, "best.pth"))
                print(f"  -> new best ({eval_metric_key}={100 * metric:.2f}), saved best.pth")
            else:
                no_improve += 1
                print(f"  -> no improvement ({no_improve}/{early_stop_patience or '∞'} since "
                      f"best {eval_metric_key}={100 * best_metric:.2f})")
                if early_stop_patience and no_improve >= early_stop_patience:
                    stop_early = True

        ckpt = make_checkpoint(model, ema, optimizer, scheduler, epoch, cfg, best_metric, best_res, no_improve)
        torch.save(ckpt, os.path.join(output_dir, "last.pth"))
        if save_every and (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if stop_early:
            print(f"Early stopping at epoch {epoch}: no {eval_metric_key} improvement for "
                  f"{no_improve} evaluations. Best weights are in "
                  f"{os.path.join(output_dir, 'best.pth')}.")
            break

    print(f"Done. Checkpoints are in {output_dir}.")
    if best_res is not None:
        order = ["mAP", "AP50", "AP75", "small<32", "medium32-96", "large>96"]
        print("Best model (best.pth): " + " ".join(f"{k}={100 * best_res[k]:.2f}" for k in order))


if __name__ == "__main__":
    main()
