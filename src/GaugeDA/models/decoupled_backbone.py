from typing import Optional, Tuple

import torch
import torch.nn as nn

from .canonical_projector import CanonicalOrientationProjector
from .st_lie_operator import STLieOperatorBlock


class DecoupledSTLieModel(nn.Module):
    """Decoupled spatio-temporal Lie architecture for wearable HAR.

    Decouples:
      1. Spatial axis: canonical orientation projector (gravity + optional yaw).
      2. Temporal axis: stacked continuous-time spatio-temporal Lie operators.
      3. Task head: linear classifier with gauge-invariant pooled features.

    ``delta_t`` is the physical sampling interval in seconds.  It is forwarded
    to every ST-LieOp layer; when omitted, ``1 / sampling_rate`` is used.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 6,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        filter_window: Optional[int] = None,
        use_projector: bool = True,
        sampling_rate: float = 50.0,
        heading_mode: str = "pca",
    ):
        super().__init__()
        self.use_projector = use_projector
        self.sampling_rate = float(sampling_rate)

        # 1. Spatial canonicalization
        self.projector = CanonicalOrientationProjector(
            filter_window=filter_window,
            sampling_rate=sampling_rate,
            heading_mode=heading_mode,
        )

        # 2. Stacked ST-LieOp layers
        self.layers = nn.ModuleList([
            STLieOperatorBlock(
                d_in=in_channels if i == 0 else d_model,
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # 3. Pooling and classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def extract_features(
        self,
        x: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        sampling_rate: Optional[float] = None,
        delta_t_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract pooled features z [B, d_model * 2]."""
        if delta_t is None:
            delta_t = delta_t_scale

        rate = self.sampling_rate if sampling_rate is None else float(sampling_rate)
        if delta_t is None:
            delta_t = 1.0 / rate

        x_canon = self.projector(x, sampling_rate=rate) if self.use_projector else x

        h = x_canon
        for layer in self.layers:
            h = layer(h, delta_t=delta_t)

        avg_pool = h.mean(dim=-1)
        max_pool = h.max(dim=-1)[0]
        return torch.cat([avg_pool, max_pool], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        sampling_rate: Optional[float] = None,
        delta_t_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, in_channels, T] -> (logits [B, C], features [B, 2*d_model])."""
        z = self.extract_features(
            x,
            delta_t=delta_t,
            sampling_rate=sampling_rate,
            delta_t_scale=delta_t_scale,
        )
        logits = self.classifier(z)
        return logits, z
