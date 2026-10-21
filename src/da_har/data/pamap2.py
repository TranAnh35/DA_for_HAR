"""PAMAP2 loader (100 Hz, three IMUs: wrist / chest / ankle).

Why this loader exists in addition to the legacy one
----------------------------------------------------
The legacy PAMAP2 pipeline produced near-chance accuracy (12 classes).  The
root cause was segmentation, not the model: transient activity ``0`` rows were
removed *before* windowing but the remaining samples were treated as one long
recording, so windows could bridge activity changes and recording gaps.  Here
we (a) keep the canonical 12-class protocol, (b) split the recording into runs
of a single activity plus timestamp gaps, and (c) require 90 % label purity per
window.  Every window is also tagged with its subject.

The loader is placement-parametric: ``sensors=("wrist",)`` and
``sensors=("ankle",)`` both produce the slot layout ``imu0:{acc,gyro}:{x,y,z}``
but different ``placement`` metadata.  That makes a cross-placement task
well-defined instead of a channel-mismatch error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import ChannelSpec, DatasetBundle, DomainData, make_imu_channels
from .segments import contiguous_segments, windows_from_segments

# 12 standard protocol activities (transient 0 and optional activities dropped).
PAMAP2_ACTIVITY_MAP: Dict[int, int] = {
    1: 0,    # lying
    2: 1,    # sitting
    3: 2,    # standing
    4: 3,    # walking
    5: 4,    # running
    6: 5,    # cycling
    7: 6,    # Nordic walking
    12: 7,   # ascending stairs
    13: 8,   # descending stairs
    16: 9,   # vacuum cleaning
    17: 10,  # ironing
    24: 11,  # rope jumping
}

LABELS: Tuple[str, ...] = (
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "cycling",
    "nordic_walking",
    "ascending_stairs",
    "descending_stairs",
    "vacuum_cleaning",
    "ironing",
    "rope_jumping",
)

# 0-indexed columns: 3x acc16 (+-16 g) followed by 3x gyro.
POSITION_COLS: Dict[str, List[int]] = {
    "wrist": [4, 5, 6, 10, 11, 12],
    "chest": [21, 22, 23, 27, 28, 29],
    "ankle": [38, 39, 40, 44, 45, 46],
}

SAMPLING_RATE = 100.0
WINDOW_SECONDS = 1.28


def _fill_missing(signal: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN channels; all-NaN channels become zero."""
    signal = np.asarray(signal, dtype=np.float32).copy()
    n = signal.shape[0]
    idx = np.arange(n)
    for c in range(signal.shape[1]):
        col = signal[:, c]
        good = np.isfinite(col)
        if good.all():
            continue
        if not good.any():
            signal[:, c] = 0.0
            continue
        signal[:, c] = np.interp(idx, idx[good], col[good]).astype(np.float32)
    return signal


def load_bundle(
    root: str | Path = "Data",
    *,
    sensors: Sequence[str] = ("wrist",),
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: Optional[float] = None,
    min_label_fraction: float = 0.9,
    subjects: Optional[Sequence[int]] = None,
    max_windows_per_domain: Optional[int] = None,
    seed: int = 0,
) -> DatasetBundle:
    base = Path(root) / "PAMAP2_Dataset" / "Protocol"
    if not base.exists():
        raise FileNotFoundError(f"PAMAP2 not found at {base}")

    sensors = tuple(sensors)
    for sensor in sensors:
        if sensor not in POSITION_COLS:
            raise ValueError(f"unknown PAMAP2 sensor {sensor!r}; expected {tuple(POSITION_COLS)}")

    channels: List[ChannelSpec] = []
    for i, sensor in enumerate(sensors):
        channels.extend(make_imu_channels(role=f"imu{i}", placement=sensor, kinds=("acc", "gyro")))
    channels_t = tuple(channels)

    files = sorted(base.glob("subject*.dat"))
    if subjects is not None:
        wanted = {int(s) for s in subjects}
        files = [f for f in files if int(f.stem.replace("subject", "")) in wanted]
    if not files:
        raise FileNotFoundError(f"no PAMAP2 subject files under {base}")

    domains: Dict[str, DomainData] = {}
    for path in files:
        subject_id = int(path.stem.replace("subject", ""))
        sensor_cols = [c for sensor in sensors for c in POSITION_COLS[sensor]]
        usecols = sorted({0, 1, *sensor_cols})
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=usecols, dtype=np.float32)
        activity_raw = df[1].to_numpy()
        timestamps = df[0].to_numpy(dtype=np.float64)
        activity = np.full(activity_raw.shape, -1, dtype=np.int64)
        for raw, mapped in PAMAP2_ACTIVITY_MAP.items():
            activity[activity_raw.astype(np.int64) == raw] = mapped

        # Select columns in the caller's sensor order (not file order) so that
        # `sensors=("ankle", "wrist")` really puts ankle in imu0.
        signal = _fill_missing(df[sensor_cols].to_numpy(dtype=np.float32))

        keep = activity >= 0
        if not keep.any():
            continue
        signal = signal[keep]
        activity = activity[keep]
        timestamps = timestamps[keep]

        segments = contiguous_segments(
            activity, timestamps, sampling_rate=SAMPLING_RATE, max_gap_seconds=0.5
        )
        X, y = windows_from_segments(
            signal,
            activity,
            segments,
            sampling_rate=SAMPLING_RATE,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_label_fraction=min_label_fraction,
        )
        if X.size == 0:
            continue

        dom = DomainData(
            name=f"subject:{subject_id}",
            X=X,
            y=y,
            subjects=np.full(len(y), str(subject_id)),
            sessions=np.full(len(y), "protocol"),
            channels=channels_t,
            sampling_rate=SAMPLING_RATE,
            meta={"dataset": "pamap2", "subject": subject_id, "sensors": list(sensors)},
        )
        if max_windows_per_domain is not None:
            dom = dom.take(max_windows_per_domain, seed=seed)
        domains[dom.name] = dom

    if not domains:
        raise ValueError("PAMAP2 produced no windows; check the raw files and label mapping")

    return DatasetBundle(
        name="pamap2",
        domains=domains,
        label_names=LABELS,
        meta={
            "sampling_rate": SAMPLING_RATE,
            "window_seconds": window_seconds,
            "sensors": list(sensors),
            "min_label_fraction": min_label_fraction,
        },
    )
