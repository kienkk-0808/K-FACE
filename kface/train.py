"""Training entrypoint for K-FACE.

Usage:
    python -m kface.train --config configs/kface_n.yaml
"""
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from kface.models.detector import KFaceDetector
from kface.data.dataset import WiderFaceDataset, collate_fn
from kface.data.augment import TrainTransform
from kface.losses.losses import KFaceLoss
from kface.utils.box_utils import generate_anchors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def build_model(cfg):
    return KFaceDetector(
        width_mult=cfg["model"]["width_mult"],
        depth_mult=cfg["model"]["depth_mult"],
        neck_channels=cfg["model"]["neck_channels"],
        num_anchors=cfg["model"]["num_anchors"],
    )


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_size = cfg["train"]["input_size"]

    dataset = WiderFaceDataset(
        cfg["data"]["train_label"], cfg["data"]["train_images"],
        transform=TrainTransform(size=train_size),
    )
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn, drop_last=True,
    )

    model = build_model(cfg).to(device)
    anchors = generate_anchors(
        train_size, strides=tuple(cfg["model"]["strides"]),
        scales_per_stride=tuple(tuple(s) for s in cfg["model"]["scales_per_stride"]),
    ).to(device)

    criterion = KFaceLoss(
        pos_iou=cfg["train"]["pos_iou"], neg_iou=cfg["train"]["neg_iou"],
        box_weight=cfg["train"]["box_weight"], kps_weight=cfg["train"]["kps_weight"],
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["train"]["lr"], momentum=0.9,
        weight_decay=cfg["train"]["weight_decay"],
    )
    epochs = cfg["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "loss_kps": 0.0}
        for step, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["kps"] = t["kps"].to(device)
                t["has_kps"] = t["has_kps"].to(device)

            cls_outs, box_outs, kps_outs = model(imgs)
            cls, box, kps = model.flatten_all(cls_outs, box_outs, kps_outs)

            losses = criterion(cls, box, kps, anchors, targets)

            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in running:
                running[k] += float(losses[k])

            if step % cfg["train"]["log_interval"] == 0:
                n = step + 1
                msg = " ".join(f"{k}={running[k]/n:.4f}" for k in running)
                print(f"[epoch {epoch}][{step}/{len(loader)}] {msg} num_pos={losses['num_pos']}")

        scheduler.step()
        ckpt_path = os.path.join(cfg["train"]["output_dir"], f"epoch_{epoch}.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
