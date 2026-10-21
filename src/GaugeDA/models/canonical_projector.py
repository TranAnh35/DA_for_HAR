import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonicalOrientationProjector(nn.Module):
    """Canonical orientation projector on the gravity-preserving subgroup.

    Two stages are applied to every window:

    1. **Gravity alignment (tilt).** The local gravity direction is estimated
       with a low-pass filter whose length is derived from the physical sampling
       rate (``filter_seconds * sampling_rate``), then the whole sensor frame is
       rotated so that gravity maps to ``e_z = [0, 0, 1]``.  The antipodal case
       ``g ~= -e_z`` is handled explicitly with a 180-degree rotation; the
       previous implementation wrongly treated it as "already aligned".
    2. **Heading canonicalization (yaw), optional.** The horizontal principal
       axis of the gravity-aligned acceleration is aligned with ``e_x``.  A
       deterministic sign rule based on the third moment of the projected
       signal removes the 180-degree axis ambiguity.  This makes the projector
       invariant to arbitrary yaw rotations (up to noise), while a true sensor
       reflection (left-handed frame) is *not* correctable by an SO(3) rotation
       and is therefore documented as a limitation rather than silently
       "fixed".

    The module is parameter-free, so it does not change checkpoint contents.
    ``kernel`` is kept as a legacy buffer so old checkpoints still load.
    """

    def __init__(
        self,
        filter_window: Optional[int] = None,
        eps: float = 1e-7,
        sampling_rate: float = 50.0,
        filter_seconds: float = 0.5,
        heading_mode: str = "pca",
        heading_min_var: float = 1e-6,
    ):
        super().__init__()
        if heading_mode not in ("none", "pca"):
            raise ValueError(f"heading_mode must be 'none' or 'pca', got {heading_mode!r}")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        self.eps = float(eps)
        self.sampling_rate = float(sampling_rate)
        self.filter_seconds = float(filter_seconds)
        self.heading_mode = heading_mode
        self.heading_min_var = float(heading_min_var)

        # ``filter_window`` is kept for explicit/legacy configuration.  When it
        # is None the window is recomputed from the physical sampling rate.
        self._explicit_window = filter_window is not None
        self.filter_window = int(filter_window) if filter_window is not None else self._window_for(self.sampling_rate)

        # Legacy buffer (persistent so old checkpoints remain loadable).
        kernel = torch.ones(1, 1, self.filter_window) / self.filter_window
        self.register_buffer("kernel", kernel)
        self.register_buffer("ez_ref", torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1), persistent=False)

    def _window_for(self, sampling_rate: float) -> int:
        window = int(round(self.filter_seconds * float(sampling_rate)))
        if window % 2 == 0:
            window += 1
        return max(3, window)

    def _get_window(self, sampling_rate: Optional[float]) -> int:
        if self._explicit_window:
            return self.filter_window
        rate = self.sampling_rate if sampling_rate is None else float(sampling_rate)
        return self._window_for(rate)

    def _estimate_gravity(self, acc: torch.Tensor, window: int) -> torch.Tensor:
        """Estimate a per-timestep gravity direction from acceleration.

        acc: [B, 3, T] -> [B, 3, T]
        """
        pad = window // 2
        acc_padded = F.pad(acc, (pad, pad), mode="replicate")
        gravity = F.avg_pool1d(acc_padded, kernel_size=window, stride=1)
        return F.normalize(gravity, p=2, dim=1, eps=self.eps)

    def _rodrigues_rotate(self, v: torch.Tensor, g_norm: torch.Tensor) -> torch.Tensor:
        """Rotate vectors ``v`` so that ``g_norm`` aligns with ``e_z``.

        Handles the two degenerate cases correctly:
        * ``g ~= +e_z``  -> identity
        * ``g ~= -e_z``  -> 180-degree rotation about ``e_x``
        """
        B, _, T = v.shape
        ez = self.ez_ref.to(dtype=v.dtype, device=v.device).expand(B, 3, T)

        k = torch.cross(g_norm, ez, dim=1)  # [B, 3, T]
        sin_theta = torch.norm(k, dim=1, keepdim=True)  # [B, 1, T]
        cos_theta = (g_norm * ez).sum(dim=1, keepdim=True)  # [B, 1, T]

        k_norm = k / (sin_theta + self.eps)
        k_cross_v = torch.cross(k_norm, v, dim=1)
        k_dot_v = (k_norm * v).sum(dim=1, keepdim=True)

        v_rot = (
            v * cos_theta
            + k_cross_v * sin_theta
            + k_norm * k_dot_v * (1.0 - cos_theta)
        )

        is_aligned = sin_theta < 1e-4
        is_antipodal = is_aligned & (cos_theta < 0.0)
        is_identity = is_aligned & (~is_antipodal)

        # Rotate 180 degrees about x: (x, y, z) -> (x, -y, -z)
        flip = torch.tensor([1.0, -1.0, -1.0], dtype=v.dtype, device=v.device).view(1, 3, 1)
        v_antipodal = v * flip

        return torch.where(is_identity, v, torch.where(is_antipodal, v_antipodal, v_rot))

    def _canonicalize_heading(
        self, acc: torch.Tensor, gyro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align the horizontal principal axis with ``e_x`` and fix its sign.

        The principal axis is computed from the 2x2 horizontal covariance.  A
        deterministic sign is chosen by requiring the third moment of the
        projected horizontal acceleration to be non-negative.  Windows with
        negligible horizontal variance (e.g. static activities) are left
        unchanged because their heading is not identifiable.
        """
        B = acc.size(0)
        hx = acc[:, 0, :]
        hy = acc[:, 1, :]

        cxx = hx.pow(2).mean(dim=1)
        cyy = hy.pow(2).mean(dim=1)
        cxy = (hx * hy).mean(dim=1)

        # Orientation of the principal axis, in [-pi/2, pi/2].
        theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
        cos_t = torch.cos(theta).unsqueeze(1)
        sin_t = torch.sin(theta).unsqueeze(1)

        # Rotate by -theta (standard R(-theta)) so the axis maps to e_x.
        hx_rot = cos_t * hx + sin_t * hy
        m3 = hx_rot.pow(3).mean(dim=1)
        sign_flip = (m3 < 0.0).to(acc.dtype)

        angle = -theta + sign_flip * math.pi
        valid = ((cxx + cyy) > self.heading_min_var).to(acc.dtype)
        angle = angle * valid

        cos_a = torch.cos(angle).view(B, 1, 1)
        sin_a = torch.sin(angle).view(B, 1, 1)

        def rotate(v: torch.Tensor) -> torch.Tensor:
            vx = cos_a * v[:, 0:1, :] - sin_a * v[:, 1:2, :]
            vy = sin_a * v[:, 0:1, :] + cos_a * v[:, 1:2, :]
            return torch.cat([vx, vy, v[:, 2:3, :]], dim=1)

        return rotate(acc), rotate(gyro)

    def forward(self, x: torch.Tensor, sampling_rate: Optional[float] = None) -> torch.Tensor:
        """Canonicalize a sensor window.

        x: [B, 6, T] with channels 0:3 = accelerometer, 3:6 = gyroscope.
        sampling_rate: optional physical rate (Hz) overriding the constructor.
        """
        assert x.size(1) >= 6, f"Expected at least 6 channels, got {x.size(1)}"
        acc = x[:, :3, :]
        gyro = x[:, 3:6, :]

        window = self._get_window(sampling_rate)
        g_norm = self._estimate_gravity(acc, window)

        acc_canon = self._rodrigues_rotate(acc, g_norm)
        gyro_canon = self._rodrigues_rotate(gyro, g_norm)

        if self.heading_mode == "pca":
            acc_canon, gyro_canon = self._canonicalize_heading(acc_canon, gyro_canon)

        if x.size(1) > 6:
            aux = x[:, 6:, :]
            return torch.cat([acc_canon, gyro_canon, aux], dim=1)
        return torch.cat([acc_canon, gyro_canon], dim=1)
