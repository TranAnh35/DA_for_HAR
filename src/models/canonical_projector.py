import torch
import torch.nn as nn
import torch.nn.functional as F

class CanonicalOrientationProjector(nn.Module):
    """
    Canonical Orientation Projector on the Gravity-Preserving Subgroup O_g(3).
    Automatically maps arbitrary device orientations to a unified canonical frame
    where the local gravity vector aligns with the global z-axis e_z = [0, 0, 1]^T.
    """
    def __init__(self, filter_window: int = 25, eps: float = 1e-7):
        super().__init__()
        self.filter_window = filter_window
        self.eps = eps
        # Smoothing kernel to extract low-frequency gravity component from total acceleration
        kernel = torch.ones(1, 1, filter_window) / filter_window
        self.register_buffer("kernel", kernel)
        self.register_buffer("ez_ref", torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1))

    def _estimate_gravity(self, acc: torch.Tensor) -> torch.Tensor:
        """
        acc: [B, 3, T]
        Returns normalized gravity direction: [B, 3, T]
        """
        pad = self.filter_window // 2
        acc_padded = F.pad(acc, (pad, pad), mode="replicate")
        # Apply depthwise 1D conv for each axis (x, y, z)
        gravity = F.conv1d(acc_padded, self.kernel.repeat(3, 1, 1), groups=3)
        gravity_norm = F.normalize(gravity, p=2, dim=1, eps=self.eps)
        return gravity_norm

    def _rodrigues_rotate(self, v: torch.Tensor, g_norm: torch.Tensor) -> torch.Tensor:
        """
        Rotates vectors v [B, 3, T] so that g_norm [B, 3, T] aligns with e_z = [0, 0, 1]^T.
        Uses numerically stable Rodrigues rotation formula.
        """
        B, _, T = v.shape
        ez = self.ez_ref.to(dtype=v.dtype, device=v.device).expand(B, 3, T)

        # Cross product axis: k = g_norm x ez
        k = torch.cross(g_norm, ez, dim=1) # [B, 3, T]
        sin_theta = torch.norm(k, dim=1, keepdim=True) # [B, 1, T]
        cos_theta = (g_norm * ez).sum(dim=1, keepdim=True) # [B, 1, T]

        # Normalized rotation axis
        k_norm = k / (sin_theta + self.eps)

        # Rodrigues formula:
        # v_rot = v * cos + (k x v) * sin + k * (k . v) * (1 - cos)
        k_cross_v = torch.cross(k_norm, v, dim=1)
        k_dot_v = (k_norm * v).sum(dim=1, keepdim=True)

        v_rot = (
            v * cos_theta
            + k_cross_v * sin_theta
            + k_norm * k_dot_v * (1.0 - cos_theta)
        )

        # Handle degenerate case where g_norm is already nearly aligned with ez (sin_theta ~ 0)
        is_aligned = (sin_theta < 1e-4).float()
        return is_aligned * v + (1.0 - is_aligned) * v_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 6, T] where channels 0:3 are Accelerometer (ax, ay, az),
           and channels 3:6 are Gyroscope (gx, gy, gz).
        Returns:
           Canonicalized sensor tensor [B, 6, T].
        """
        assert x.size(1) >= 6, f"Expected at least 6 channels, got {x.size(1)}"
        acc = x[:, :3, :]
        gyro = x[:, 3:6, :]

        # 1. Estimate gravity orientation
        g_norm = self._estimate_gravity(acc)

        # 2. Canonical rotate both acceleration and gyroscope
        acc_canon = self._rodrigues_rotate(acc, g_norm)
        gyro_canon = self._rodrigues_rotate(gyro, g_norm)

        if x.size(1) > 6:
            # Pass through any auxiliary channels (e.g. heart rate, magnet) unchanged
            aux = x[:, 6:, :]
            return torch.cat([acc_canon, gyro_canon, aux], dim=1)
        
        return torch.cat([acc_canon, gyro_canon], dim=1)
