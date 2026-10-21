import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_dt_scale(delta_t, u: torch.Tensor) -> torch.Tensor:
    """Normalize ``delta_t`` to a tensor broadcastable with [B, T, d_model]."""
    scale = torch.as_tensor(delta_t, device=u.device, dtype=u.dtype)
    if scale.ndim == 0:
        return scale.view(1, 1, 1)
    if scale.ndim == 1:
        return scale.view(-1, 1, 1)
    if scale.ndim == 2:
        return scale.view(scale.size(0), 1, 1)
    if scale.ndim == 3:
        return scale
    raise ValueError(f"Unsupported delta_t shape: {tuple(scale.shape)}")


def ssm_recurrent_scan(
    A_bar: torch.Tensor,
    deltaB_u: torch.Tensor,
    C_mat: torch.Tensor,
    D: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Recurrent scan of the discretized state-space model.

    A_bar:    [B, T, d_model, d_state]
    deltaB_u: [B, T, d_model, d_state] (exact ZOH input term)
    C_mat:    [B, T, d_state]
    D:        [d_model]
    u:        [B, T, d_model]
    Returns:  [B, T, d_model]
    """
    B = u.size(0)
    d_model = u.size(2)
    d_state = A_bar.size(3)

    h = torch.zeros(B, d_model, d_state, device=u.device, dtype=u.dtype)

    A_list = A_bar.unbind(1)
    dBu_list = deltaB_u.unbind(1)
    C_list = C_mat.unbind(1)
    u_list = u.unbind(1)

    y_list = []
    for At, dBu_t, Ct, ut in zip(A_list, dBu_list, C_list, u_list):
        h = At * h + dBu_t
        yt = (h * Ct.unsqueeze(1)).sum(dim=-1) + D * ut
        y_list.append(yt)

    return torch.stack(y_list, dim=1)


class ContinuousSSMCell(nn.Module):
    """Continuous-time diagonal state-space cell.

    The continuous dynamics ``dh/dt = A h + B u`` are discretized with a
    **zero-order hold** while ``u`` is constant over one sampling interval:

        A_bar     = exp(delta_t * A)
        B_bar_u   = (A_bar - I) A^{-1} B u

    ``delta_t`` is the *physical* sampling interval in seconds
    (``1 / sampling_rate``), not an abstract scale.  This keeps the effective
    memory of ``A`` tied to real time: with ``A in [1, d_state]`` and
    ``delta_t = 0.02 s`` the slowest mode decays over ~50 samples instead of
    one step.
    """

    def __init__(self, d_model: int = 128, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Hurwitz-stable continuous A (negative real part).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(d_model, 1)
        )

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(
        self,
        u: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        delta_t_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """u: [B, T, d_model]; delta_t: physical seconds (scalar, [B], or [B,1,1])."""
        if delta_t is None:
            delta_t = delta_t_scale

        A = -torch.exp(self.A_log)  # [d_model, d_state]

        # Dimensionless learned time constant, converted to physical seconds.
        dt = F.softplus(self.dt_proj(u))  # [B, T, d_model]
        if delta_t is not None:
            dt = dt * _as_dt_scale(delta_t, u)
        # Avoid underflow/overflow for pathological rates; HAR data are >= 50 Hz.
        dt = torch.clamp(dt, min=1e-5, max=0.5)

        B_mat = self.B_proj(u)  # [B, T, d_state]
        C_mat = self.C_proj(u)  # [B, T, d_state]

        dt_expanded = dt.unsqueeze(-1)  # [B, T, d_model, 1]
        A_expanded = A.view(1, 1, self.d_model, self.d_state)

        # Exact ZOH transition and input term.
        A_bar = torch.exp(dt_expanded * A_expanded)  # [B, T, d_model, d_state]
        phi = torch.expm1(dt_expanded * A_expanded) / A_expanded  # (A_bar - I) / A
        deltaB_u = phi * u.unsqueeze(-1) * B_mat.unsqueeze(2)  # [B, T, d_model, d_state]

        return ssm_recurrent_scan(A_bar, deltaB_u, C_mat, self.D, u)


class STLieOperatorBlock(nn.Module):
    """Spatio-temporal Lie operator block.

    Local temporal convolution + continuous-time SSM, with a residual path
    around the block.  The residual keeps gradients and identity information
    flowing when the learned dynamics are close to zero.
    """

    def __init__(self, d_in: int = 6, d_model: int = 128, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.in_proj = nn.Linear(d_in, d_model)

        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.act = nn.SiLU()

        self.ssm = ContinuousSSMCell(d_model=d_model, d_state=d_state)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        delta_t_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, C, T] -> [B, d_model, T]."""
        if delta_t is None:
            delta_t = delta_t_scale

        u = x.transpose(1, 2)  # [B, T, C]
        u_proj = self.in_proj(u)  # [B, T, d_model]

        u_conv = self.conv1d(u_proj.transpose(1, 2)).transpose(1, 2)
        u_conv = self.act(u_conv)

        # Local residual before the long-range operator.
        u_gated = u_proj + u_conv

        y = self.ssm(u_gated, delta_t=delta_t)
        y = self.norm(y)
        y = self.dropout(y)
        out = self.out_proj(y)

        return (out + u_proj).transpose(1, 2)
