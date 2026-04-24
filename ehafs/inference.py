"""Inference helper: load checkpoint, predict on a single video, return attention."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .model import EHAFS
from .transforms import center_crop_and_resize, normalize_clip, uniform_sample_indices
from .utils import get_device


class EHAFSPredictor:
    def __init__(
        self,
        checkpoint: str | Path | None,
        num_frames: int = 16,
        frame_size: int = 224,
        classes: tuple[str, ...] = ("NonFight", "Fight"),
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.classes = classes

        self.model = EHAFS(num_classes=len(classes), num_frames=num_frames, pretrained=False)
        self.has_weights = False
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
            self.has_weights = True
        self.model.to(self.device).eval()

    def _load_clip(self, video_path: str) -> tuple[torch.Tensor, np.ndarray, list[int]]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        indices = uniform_sample_indices(total, self.num_frames).tolist()

        cap = cv2.VideoCapture(video_path)
        grabbed: dict[int, np.ndarray] = {}
        for idx in sorted(set(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            grabbed[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        raw = np.stack([grabbed[i] for i in indices])
        resized = center_crop_and_resize(raw, target=self.frame_size)
        tensor = normalize_clip(resized).unsqueeze(0).to(self.device)  # [1,T,3,H,W]
        return tensor, resized, indices

    @torch.no_grad()
    def predict(self, video_path: str) -> dict:
        clip, frames_rgb, indices = self._load_clip(video_path)
        logits, attn = self.model(clip, return_attention=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        attn = attn.cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return {
            "label": self.classes[pred_idx],
            "label_index": pred_idx,
            "probs": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "attention": attn.tolist(),
            "frame_indices": indices,
            "frames": frames_rgb,  # uint8 RGB [T,H,W,3] for visualization
            "has_trained_weights": self.has_weights,
        }
