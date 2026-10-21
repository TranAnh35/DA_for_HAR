from typing import Tuple
import torch
import torch.nn as nn

class Standard1DCNN(nn.Module):
    """
    Standard 3-layer 1D-CNN backbone widely used in time-series UDA literature
    (e.g., AdaTime benchmark [TKDD 2023], CoDATS [KDD 2020], DANN, Deep-CORAL).
    
    Accepts input tensor x: [B, C, T] and projects to feature representation z: [B, d_model],
    followed by linear classifier head: [B, num_classes].
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 6,
        d_model: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model

        # 3-layer 1D Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=8, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, d_model, kernel_size=3, padding="same"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Classification Head
        self.classifier = nn.Linear(d_model, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T] -> z: [B, d_model]
        """
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B, num_classes]
            features: [B, d_model]
        """
        z = self.extract_features(x)
        logits = self.classifier(z)
        return logits, z
