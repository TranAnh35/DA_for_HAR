"""Reference baselines for the DA-HAR benchmark.

These are *baselines*, not a contribution.  ``MaskedCNN1D`` deliberately
accepts a channel mask so that a task whose source and target expose different
sensor slots (e.g. cross-placement) still has a well-defined input shape.  The
mask is metadata that a future architecture is free to exploit better.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MaskedCNN1D(nn.Module):
    """Small 1D-CNN with an explicit channel-presence mask."""

    def __init__(
        self,
        in_slots: int,
        num_classes: int,
        hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if in_slots <= 0:
            raise ValueError("in_slots must be positive")
        self.in_slots = in_slots
        self.features = nn.Sequential(
            nn.Conv1d(in_slots, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Dropout(dropout),
        )
        self.feat_dim = hidden * 2
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            x = x * mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        h = self.features(x)
        return self.classifier(h), h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_backbone(name: str, in_slots: int, num_classes: int, **kwargs) -> nn.Module:
    if name in ("cnn1d", "masked_cnn1d"):
        return MaskedCNN1D(in_slots=in_slots, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown backbone {name!r}; available: ('cnn1d',)")


BACKBONES: Dict[str, str] = {"cnn1d": "MaskedCNN1D (masked channel input)"}
