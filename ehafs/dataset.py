"""RWF-2000 dataset loader.

Expected layout:

    data/RWF-2000/
        train/
            Fight/      *.avi or *.mp4
            NonFight/   *.avi or *.mp4
        val/
            Fight/      *.avi or *.mp4
            NonFight/   *.avi or *.mp4
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import (
    center_crop_and_resize,
    color_jitter,
    normalize_clip,
    random_crop_and_resize,
    random_horizontal_flip,
    uniform_sample_indices,
)

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def load_video_frames(path: str, indices: Sequence[int]) -> np.ndarray:
    """Read selected frame indices from a video file as RGB uint8 array [T,H,W,3]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(indices)
    indices = [min(int(i), total - 1) for i in indices]

    frames: list[np.ndarray] = []
    wanted = sorted(set(indices))
    cur = 0
    grabbed = {}
    for idx in wanted:
        # naive seek (works on most codecs); fall back to sequential read
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            # fallback: read sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for j in range(idx + 1):
                ok, frame = cap.read()
                if not ok:
                    break
        if not ok or frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        grabbed[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur = idx
    cap.release()

    for idx in indices:
        frames.append(grabbed.get(idx, np.zeros((224, 224, 3), dtype=np.uint8)))
    return np.stack(frames)


class RWF2000Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_frames: int = 16,
        frame_size: int = 224,
        classes: Sequence[str] = ("NonFight", "Fight"),
        augment: bool | None = None,
    ) -> None:
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(
                f"RWF-2000 split not found: {self.root}\n"
                f"Run: python scripts/download_rwf2000.py"
            )
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.augment = augment if augment is not None else (split == "train")

        self.samples: list[tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class dir: {cls_dir}")
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in VIDEO_EXTS:
                    self.samples.append((str(p), self.class_to_idx[cls]))
        if not self.samples:
            raise RuntimeError(f"No videos found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        indices = uniform_sample_indices(total, self.num_frames)
        clip = load_video_frames(path, indices)  # [T,H,W,3] uint8 RGB

        if self.augment:
            clip = random_horizontal_flip(clip, p=0.5)
            clip = color_jitter(clip, 0.2, 0.2)
            clip = random_crop_and_resize(clip, target=self.frame_size, scale_to=self.frame_size + 32)
        else:
            clip = center_crop_and_resize(clip, target=self.frame_size)

        return normalize_clip(clip), label
