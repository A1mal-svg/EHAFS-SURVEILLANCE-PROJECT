"""Temporal Shift Module (Lin et al., ICCV 2019).

Shifts a fraction (1/div) of channels forward and backward along the temporal
dimension, allowing a 2D CNN to model temporal information at zero FLOP cost.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    def __init__(self, n_segment: int = 16, n_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W]
        nt, c, h, w = x.shape
        t = self.n_segment
        b = nt // t
        x = x.view(b, t, c, h, w)

        fold = c // self.n_div
        out = torch.zeros_like(x)
        # shift left  (channels go from t -> t-1)  => future leaks into past
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # shift right (channels go from t -> t+1)
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        # not shifted
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)


def inject_tsm(module: nn.Module, n_segment: int, n_div: int = 8) -> nn.Module:
    """Wrap each InvertedResidual block in MobileNetV3 with TSM applied to its first conv input."""
    from torchvision.models.mobilenetv3 import InvertedResidual

    for name, child in module.named_children():
        if isinstance(child, InvertedResidual):
            # prepend a TSM to the residual block
            shifted = nn.Sequential(TemporalShift(n_segment, n_div), child)
            setattr(module, name, shifted)
        else:
            inject_tsm(child, n_segment, n_div)
    return module
