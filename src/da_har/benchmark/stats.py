"""Small statistical helpers for paired seed-level comparisons."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = ["paired_bootstrap", "summarize"]


def paired_bootstrap(
    differences: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CI and two-sided p-value for the mean paired difference."""
    diff = np.asarray(differences, dtype=np.float64).reshape(-1)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n": 0}
    if diff.size == 1:
        return {
            "mean": float(diff[0]),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n": 1,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return {
        "mean": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "n": int(diff.size),
    }


def summarize(values: Sequence[float]) -> Tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size)
