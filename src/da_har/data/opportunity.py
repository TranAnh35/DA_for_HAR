"""Opportunity (UCI) loader: 30 Hz, 4 subjects, ADL1-5 + Drill sessions.

Only the on-body accelerometers and the ``Locomotion`` label track are used.
Subject and session are both exposed as domain factors, which gives two very
different shifts from the same recordings (cross-subject and cross-session).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import ChannelSpec, DatasetBundle, DomainData, make_imu_channels
from .segments import contiguous_segments, windows_from_segments

# 0-indexed accelerometer triples (column_names.txt: columns 2..19).
SENSOR_COLS: Dict[str, List[int]] = {
    "RKN": [1, 2, 3],
    "HIP": [4, 5, 6],
    "LUA": [7, 8, 9],
    "RUA": [10, 11, 12],
    "LH": [13, 14, 15],
    "BACK": [16, 17, 18],
}

# Locomotion track: 1 stand, 2 walk, 4 sit, 5 lie (0 = null).
LOCOMOTION_MAP: Dict[int, int] = {1: 0, 2: 1, 4: 2, 5: 3}
LABELS: Tuple[str, ...] = ("stand", "walk", "sit", "lie")

LABEL_COLUMN = 243  # 0-indexed -> "Column: 244 Locomotion"
TIMESTAMP_COLUMN = 0
SAMPLING_RATE = 30.0
WINDOW_SECONDS = 2.0


def load_bundle(
    root: str | Path = "Data",
    *,
    sensors: Sequence[str] = ("RKN", "HIP", "LUA", "RUA", "LH", "BACK"),
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: Optional[float] = None,
    min_label_fraction: float = 0.9,
    subjects: Optional[Sequence[int]] = None,
    sessions: Optional[Sequence[str]] = None,
    max_windows_per_domain: Optional[int] = None,
    seed: int = 0,
) -> DatasetBundle:
    base = Path(root) / "OpportunityUCIDataset" / "dataset"
    if not base.exists():
        raise FileNotFoundError(f"Opportunity not found at {base}")

    sensors = tuple(sensors)
    for sensor in sensors:
        if sensor not in SENSOR_COLS:
            raise ValueError(f"unknown Opportunity sensor {sensor!r}; expected {tuple(SENSOR_COLS)}")

    channels: List[ChannelSpec] = []
    for i, sensor in enumerate(sensors):
        channels.extend(make_imu_channels(role=f"imu{i}", placement=sensor.lower(), kinds=("acc",)))
    channels_t = tuple(channels)

    wanted_subjects = {int(s) for s in subjects} if subjects is not None else None
    wanted_sessions = {str(s) for s in sessions} if sessions is not None else None

    col_index: List[int] = []
    for sensor in sensors:
        col_index.extend(SENSOR_COLS[sensor])

    files = sorted(base.glob("S*-*.dat"))
    if not files:
        raise FileNotFoundError(f"no Opportunity .dat files under {base}")

    per_subject: Dict[int, List[DomainData]] = {}
    for path in files:
        stem = path.stem  # e.g. S1-ADL1
        try:
            subject = int(stem[1])
        except (IndexError, ValueError):
            continue
        session = stem.split("-", 1)[1]
        if wanted_subjects is not None and subject not in wanted_subjects:
            continue
        if wanted_sessions is not None and session not in wanted_sessions:
            continue

        usecols = sorted({TIMESTAMP_COLUMN, LABEL_COLUMN, *col_index})
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            usecols=usecols,
            dtype=np.float32,
        )
        raw_label = df[LABEL_COLUMN].to_numpy()
        activity = np.full(raw_label.shape, -1, dtype=np.int64)
        for raw, mapped in LOCOMOTION_MAP.items():
            activity[raw_label.astype(np.int64) == raw] = mapped

        signal = df[col_index].to_numpy(dtype=np.float32)
        timestamps = df[TIMESTAMP_COLUMN].to_numpy(dtype=np.float64) / 1000.0  # ms -> s

        keep = activity >= 0
        if not keep.any():
            continue
        signal = signal[keep]
        activity = activity[keep]
        timestamps = timestamps[keep]
        # Opportunity uses -1 for missing sensor readings.
        signal = np.where(signal <= -1e6, np.nan, signal)
        for c in range(signal.shape[1]):
            col = signal[:, c]
            good = np.isfinite(col)
            if not good.all():
                idx = np.arange(len(col))
                signal[:, c] = np.interp(idx, idx[good], col[good]) if good.any() else 0.0

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
            name=f"subject:S{subject}/{session}",
            X=X,
            y=y,
            subjects=np.full(len(y), str(subject)),
            sessions=np.full(len(y), session),
            channels=channels_t,
            sampling_rate=SAMPLING_RATE,
            meta={"dataset": "opportunity", "subject": subject, "session": session},
        )
        per_subject.setdefault(subject, []).append(dom)

    if not per_subject:
        raise ValueError("Opportunity produced no windows; check the raw files and label column")

    domains: Dict[str, DomainData] = {}
    for subject, parts in sorted(per_subject.items()):
        X = np.concatenate([p.X for p in parts], axis=0)
        y = np.concatenate([p.y for p in parts], axis=0)
        subjects_arr = np.concatenate([p.subjects for p in parts])
        sessions_arr = np.concatenate([p.sessions for p in parts])
        dom = DomainData(
            name=f"subject:S{subject}",
            X=X,
            y=y,
            subjects=subjects_arr,
            sessions=sessions_arr,
            channels=channels_t,
            sampling_rate=SAMPLING_RATE,
            meta={"dataset": "opportunity", "subject": subject},
        )
        if max_windows_per_domain is not None:
            dom = dom.take(max_windows_per_domain, seed=seed)
        domains[dom.name] = dom

    return DatasetBundle(
        name="opportunity",
        domains=domains,
        label_names=LABELS,
        meta={"sampling_rate": SAMPLING_RATE, "window_seconds": window_seconds, "sensors": list(sensors)},
    )
