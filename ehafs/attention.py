"""Frame-level temporal attention head (Meng et al., ICLR 2019).

Given per-frame features f_t in R^C, learn a vector w in R^C, compute scalar
scores s_t = w . f_t, take softmax over t -> alpha_t, return sum_t alpha_t f_t
plus the attention weights for visualization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # feats: [B, T, C]
        scores = self.score(feats).squeeze(-1)         # [B, T]
        weights = F.softmax(scores, dim=1)             # [B, T]
        pooled = torch.einsum("bt,btc->bc", weights, feats)  # [B, C]
        return pooled, weights
