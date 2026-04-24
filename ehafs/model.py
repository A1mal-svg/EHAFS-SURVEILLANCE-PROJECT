"""EHAFS model: MobileNetV3-Small + TSM + Temporal Attention."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .attention import TemporalAttention
from .tsm import inject_tsm


class EHAFS(nn.Module):
    """Hybrid CNN + temporal attention for video action recognition."""

    def __init__(
        self,
        num_classes: int = 2,
        num_frames: int = 16,
        tsm_div: int = 8,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # features -> [B*T, 576, 7, 7]
        self.features = backbone.features
        inject_tsm(self.features, n_segment=num_frames, n_div=tsm_div)

        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 576  # MobileNetV3-Small last-channel before classifier
        self.attention = TemporalAttention(feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, 3, H, W]
        b, t, c, h, w = x.shape
        assert t == self.num_frames, f"Expected T={self.num_frames}, got {t}"
        x = x.view(b * t, c, h, w)

        feat = self.features(x)            # [B*T, 576, 7, 7]
        feat = self.pool(feat).flatten(1)  # [B*T, 576]
        feat = feat.view(b, t, -1)         # [B, T, 576]

        pooled, weights = self.attention(feat)  # [B, 576], [B, T]
        logits = self.classifier(pooled)        # [B, num_classes]

        if return_attention:
            return logits, weights
        return logits


def build_model(cfg: dict) -> EHAFS:
    m = cfg["model"]
    return EHAFS(
        num_classes=m["num_classes"],
        num_frames=cfg["data"]["num_frames"],
        tsm_div=m["tsm_div"],
        pretrained=m["pretrained"],
        dropout=m.get("dropout", 0.2),
    )
