from typing import Optional, Tuple
import torch
import torch.nn as nn
from .canonical_projector import CanonicalOrientationProjector
from .st_lie_operator import STLieOperatorBlock

class DecoupledSTLieModel(nn.Module):
    """
    Decoupled Spatio-Temporal Lie Architecture for Wearable HAR.
    Decouples:
      1. Spatial axis: Canonical Orientation Projector on O_g(3).
      2. Temporal axis: Stack of Continuous-Time Spatio-Temporal Lie Operators (ST-LieOp).
      3. Task Head: Linear classifier with gauge-invariant feature pooling.
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 6,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        filter_window: int = 25,
        use_projector: bool = True
    ):
        super().__init__()
        self.use_projector = use_projector
        # 1. Spatial Canonicalization
        self.projector = CanonicalOrientationProjector(filter_window=filter_window)

        # 2. Stacked ST-LieOp layers
        self.layers = nn.ModuleList([
            STLieOperatorBlock(
                d_in=in_channels if i == 0 else d_model,
                d_model=d_model,
                d_state=d_state,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        # 3. Pooling and Classification Head
        # Concatenate Average Pooling and Max Pooling for rich temporal summary
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def extract_features(self, x: torch.Tensor, delta_t_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract pooled gauge-invariant latent features z [B, d_model * 2].
        """
        # Step 1: Spatial canonical projection
        x_canon = self.projector(x) if self.use_projector else x

        # Step 2: Temporal state-space evolution
        h = x_canon
        for layer in self.layers:
            h = layer(h, delta_t_scale=delta_t_scale)

        # Step 3: Dual pooling over sequence length
        avg_pool = h.mean(dim=-1) # [B, d_model]
        max_pool = h.max(dim=-1)[0] # [B, d_model]
        z = torch.cat([avg_pool, max_pool], dim=-1) # [B, d_model * 2]
        return z

    def forward(
        self,
        x: torch.Tensor,
        delta_t_scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, in_channels, T]
        Returns:
            logits: [B, num_classes]
            features: [B, d_model * 2]
        """
        z = self.extract_features(x, delta_t_scale=delta_t_scale)
        logits = self.classifier(z)
        return logits, z
