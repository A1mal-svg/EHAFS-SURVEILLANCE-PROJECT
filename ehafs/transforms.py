"""Video transforms (uniform sampling + augmentation + normalization)."""
from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def uniform_sample_indices(num_frames: int, target: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if num_frames < target:
        # pad by repeating last frame
        idx = np.arange(num_frames)
        pad = np.full(target - num_frames, num_frames - 1, dtype=np.int64)
        return np.concatenate([idx, pad])
    return np.linspace(0, num_frames - 1, target).astype(np.int64)


def normalize_clip(clip: np.ndarray) -> torch.Tensor:
    """clip: [T, H, W, 3] uint8 RGB -> [T, 3, H, W] float tensor, ImageNet-normalized."""
    clip = clip.astype(np.float32) / 255.0
    clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
    clip = np.transpose(clip, (0, 3, 1, 2))  # T,C,H,W
    return torch.from_numpy(clip).float()


def random_horizontal_flip(clip: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() < p:
        return clip[:, :, ::-1, :].copy()
    return clip


def color_jitter(clip: np.ndarray, brightness: float = 0.2, contrast: float = 0.2) -> np.ndarray:
    b = 1.0 + np.random.uniform(-brightness, brightness)
    c = 1.0 + np.random.uniform(-contrast, contrast)
    out = clip.astype(np.float32) * b
    mean = out.mean()
    out = (out - mean) * c + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def random_crop_and_resize(clip: np.ndarray, target: int = 224, scale_to: int = 256) -> np.ndarray:
    import cv2
    t, h, w, _ = clip.shape
    resized = np.stack([cv2.resize(f, (scale_to, scale_to)) for f in clip])
    y = np.random.randint(0, scale_to - target + 1)
    x = np.random.randint(0, scale_to - target + 1)
    return resized[:, y : y + target, x : x + target, :]


def center_crop_and_resize(clip: np.ndarray, target: int = 224) -> np.ndarray:
    import cv2
    return np.stack([cv2.resize(f, (target, target)) for f in clip])
