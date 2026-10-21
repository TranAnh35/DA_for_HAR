"""Segmentation helpers shared by the continuous-signal loaders."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["contiguous_segments"]


def contiguous_segments(
    labels: np.ndarray,
    timestamps: np.ndarray,
    sampling_rate: float,
    max_gap_seconds: float = 0.25,
    min_samples: int = 1,
) -> List[Tuple[int, int]]:
    """Split a recording into ``[start, end)`` runs of one activity.

    A run ends when the activity id changes, when the timestamps jump by more
    than ``max_gap_seconds`` (sensor drop-out / removed transient activity), or
    when the caller's arrays end. Windowing inside a run is therefore immune to
    windows that silently bridge two activities or a recording gap.
    """
    labels = np.asarray(labels).reshape(-1)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if labels.size != timestamps.size:
        raise ValueError("labels and timestamps must have the same length")
    # ``timestamps`` must be in seconds. Callers convert ms/ns before calling.
    if labels.size == 0:
        return []

    dt = 1.0 / float(sampling_rate)
    gap = max_gap_seconds if max_gap_seconds is not None else 0.25
    max_delta = gap if gap > 0 else 4.0 * dt

    segments: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, labels.size):
        boundary = labels[i] != labels[start]
        if not boundary:
            delta = timestamps[i] - timestamps[i - 1]
            if not np.isfinite(delta):
                boundary = True
            elif delta > max_delta:
                boundary = True
        if boundary:
            if i - start >= min_samples:
                segments.append((start, i))
            start = i
    if labels.size - start >= min_samples:
        segments.append((start, labels.size))
    return segments


def windows_from_segments(
    signal: np.ndarray,
    labels: np.ndarray,
    segments: Sequence[Tuple[int, int]],
    sampling_rate: float,
    window_seconds: float,
    stride_seconds: float | None = None,
    min_label_fraction: float = 0.9,
    max_windows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply :func:`src.da_har.data.core.make_windows` inside each segment."""
    from .core import make_windows

    xs, ys = [], []
    remaining = max_windows
    for start, end in segments:
        if remaining is not None and remaining <= 0:
            break
        seg_signal = signal[start:end]
        seg_labels = labels[start:end]
        X, y, _ = make_windows(
            seg_signal,
            seg_labels,
            sampling_rate=sampling_rate,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_label_fraction=min_label_fraction,
            max_windows=remaining,
        )
        if X.size:
            xs.append(X)
            ys.append(y)
            if remaining is not None:
                remaining -= len(y)
    if not xs:
        width = signal.shape[1] if signal.ndim == 2 else 0
        win = int(round(window_seconds * sampling_rate))
        return np.empty((0, width, win), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
