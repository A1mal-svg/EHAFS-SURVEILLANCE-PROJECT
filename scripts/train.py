"""Train EHAFS on RWF-2000."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehafs.dataset import RWF2000Dataset
from ehafs.model import build_model
from ehafs.utils import compute_metrics, get_device, load_config, set_seed


def make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    d = cfg["data"]
    train_ds = RWF2000Dataset(
        d["root"], split="train",
        num_frames=d["num_frames"], frame_size=d["frame_size"],
        classes=d["classes"], augment=True,
    )
    val_ds = RWF2000Dataset(
        d["root"], split="val",
        num_frames=d["num_frames"], frame_size=d["frame_size"],
        classes=d["classes"], augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=d["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=d["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool) -> tuple[float, list[int], list[int]]:
    model.train(train)
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    use_amp = scaler is not None
    with ctx:
        for clips, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * clips.size(0)
            n += clips.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    return total_loss / max(n, 1), y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"[EHAFS] Device: {device}")

    train_loader, val_loader = make_loaders(cfg)
    print(f"[EHAFS] Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["min_lr"]
    )
    scaler = torch.amp.GradScaler(device.type) if cfg["train"]["amp"] and device.type == "cuda" else None

    out_dir = Path(cfg["train"]["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["train"]["log_dir"])

    best_f1 = -1.0
    patience = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        va_loss, va_y, va_p = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False)
        scheduler.step()

        tr_m = compute_metrics(tr_y, tr_p)
        va_m = compute_metrics(va_y, va_p)
        msg = (f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} acc {tr_m['accuracy']:.4f} "
               f"| val_loss {va_loss:.4f} acc {va_m['accuracy']:.4f} f1 {va_m['f1']:.4f}")
        print(msg)
        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("acc/val", va_m["accuracy"], epoch)
        writer.add_scalar("f1/val", va_m["f1"], epoch)

        if va_m["f1"] > best_f1:
            best_f1 = va_m["f1"]
            patience = 0
            ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": va_m, "config": cfg}
            torch.save(ckpt, out_dir / "ehafs_best.pt")
            with open(out_dir / "best_metrics.json", "w") as f:
                json.dump(va_m, f, indent=2)
            print(f"  ↳ saved new best (F1={best_f1:.4f})")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"[EHAFS] Early stop at epoch {epoch} (no F1 improvement for {patience} epochs)")
                break

    writer.close()
    print(f"[EHAFS] Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
