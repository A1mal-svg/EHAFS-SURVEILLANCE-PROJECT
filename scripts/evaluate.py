"""Evaluate EHAFS on the RWF-2000 val split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehafs.dataset import RWF2000Dataset
from ehafs.model import build_model
from ehafs.utils import compute_metrics, get_device, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/ehafs_best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    d = cfg["data"]
    ds = RWF2000Dataset(d["root"], split="val", num_frames=d["num_frames"],
                        frame_size=d["frame_size"], classes=d["classes"], augment=False)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=d["num_workers"], pin_memory=True)

    y_true, y_pred = [], []
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="eval"):
            clips = clips.to(device); labels = labels.to(device)
            logits = model(clips)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
