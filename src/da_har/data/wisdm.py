"""WISDM 2.0 loader (phone + watch, ~20 Hz).

WISDM 2.0 stores one text file per ``(subject, sensor, device)``.  Accelerometer
and gyroscope are separate files, so they are resampled onto a shared uniform
grid before windowing.  Device is the natural domain, which makes WISDM the
cheapest cross-device task available in ``Data/`` (the phone and the watch are
worn at the same time and observe the same activities).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import ChannelSpec, DatasetBundle, DomainData, make_imu_channels
from .segments import contiguous_segments, windows_from_segments

ACTIVITY_KEY: Dict[str, str] = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}
LABELS: Tuple[str, ...] = tuple(ACTIVITY_KEY[k] for k in sorted(ACTIVITY_KEY))
ACTIVITY_INDEX: Dict[str, int] = {letter: i for i, letter in enumerate(sorted(ACTIVITY_KEY))}

SAMPLING_RATE = 20.0
WINDOW_SECONDS = 2.0

_PLACEMENT = {"phone": "pocket", "watch": "wrist"}

_COLUMNS = ["subject", "activity", "timestamp", "x", "y", "z"]
_DTYPES = {
    "subject": "int32",
    "activity": "string",
    "timestamp": "int64",
    "x": "float32",
    "y": "float32",
}


def _read_sensor_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=_COLUMNS,
        dtype=_DTYPES,
        converters={"z": lambda s: float(str(s).strip().rstrip(";"))},
    )
    df = df[df["activity"].isin(ACTIVITY_KEY.keys())]
    return df


def _grid_interp(t: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(grid, t, values[:, c]) for c in range(values.shape[1])], axis=1
    ).astype(np.float32)


def load_bundle(
    root: str | Path = "Data",
    *,
    devices: Sequence[str] = ("phone", "watch"),
    sensors: Sequence[str] = ("accel", "gyro"),
    target_hz: float = SAMPLING_RATE,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: Optional[float] = None,
    min_label_fraction: float = 0.9,
    subjects: Optional[Sequence[int]] = None,
    max_files: Optional[int] = None,
    max_windows_per_domain: Optional[int] = None,
    seed: int = 0,
) -> DatasetBundle:
    base = Path(root) / "wisdm-dataset" / "raw"
    if not base.exists():
        raise FileNotFoundError(f"WISDM not found at {base}")

    wanted_subjects = {int(s) for s in subjects} if subjects is not None else None
    domains: Dict[str, DomainData] = {}

    for device in devices:
        channels: Tuple[ChannelSpec, ...] = make_imu_channels(
            role="imu0", placement=_PLACEMENT.get(device, device), kinds=("acc", "gyro")
        )
        per_key: Dict[Tuple[int, str], Dict[str, pd.DataFrame]] = {}
        files_read = 0
        for sensor in sensors:
            folder = base / device / sensor
            if not folder.exists():
                continue
            for path in sorted(folder.glob("data_*.txt")):
                if max_files is not None and files_read >= max_files:
                    break
                df = _read_sensor_file(path)
                files_read += 1
                for (subject, activity), grp in df.groupby(["subject", "activity"], observed=True):
                    subject = int(subject)
                    if wanted_subjects is not None and subject not in wanted_subjects:
                        continue
                    per_key.setdefault((subject, str(activity)), {})[sensor] = grp

        xs, ys, subs, sess = [], [], [], []
        for (subject, activity), frames in sorted(per_key.items()):
            if not all(s in frames for s in sensors):
                continue
            acc = frames["accel"].sort_values("timestamp")
            gyro = frames["gyro"].sort_values("timestamp")
            if len(acc) < 4 or len(gyro) < 4:
                continue
            t_acc = acc["timestamp"].to_numpy(dtype=np.float64) / 1e9
            t_gyro = gyro["timestamp"].to_numpy(dtype=np.float64) / 1e9
            start = max(t_acc[0], t_gyro[0])
            stop = min(t_acc[-1], t_gyro[-1])
            if stop <= start:
                continue
            grid = np.arange(start, stop, 1.0 / target_hz)
            if grid.size < 4:
                continue
            acc_i = _grid_interp(t_acc, acc[["x", "y", "z"]].to_numpy(dtype=np.float64), grid)
            gyro_i = _grid_interp(t_gyro, gyro[["x", "y", "z"]].to_numpy(dtype=np.float64), grid)
            signal = np.concatenate([acc_i, gyro_i], axis=1)

            label = ACTIVITY_INDEX[activity]
            labels = np.full(signal.shape[0], label, dtype=np.int64)
            segments = contiguous_segments(
                labels, grid, sampling_rate=target_hz, max_gap_seconds=0.5
            )
            X, y = windows_from_segments(
                signal,
                labels,
                segments,
                sampling_rate=target_hz,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds,
                min_label_fraction=min_label_fraction,
            )
            if X.size == 0:
                continue
            xs.append(X)
            ys.append(y)
            subs.append(np.full(len(y), str(subject)))
            sess.append(np.full(len(y), device))

        if not xs:
            continue
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        subjects_arr = np.concatenate(subs)
        sessions_arr = np.concatenate(sess)
        dom = DomainData(
            name=f"device:{device}",
            X=X,
            y=y,
            subjects=subjects_arr,
            sessions=sessions_arr,
            channels=channels,
            sampling_rate=target_hz,
            meta={"dataset": "wisdm", "device": device},
        )
        if max_windows_per_domain is not None:
            dom = dom.take(max_windows_per_domain, seed=seed)
        domains[dom.name] = dom

    if not domains:
        raise ValueError("WISDM produced no windows; check the raw folder layout")

    return DatasetBundle(
        name="wisdm",
        domains=domains,
        label_names=LABELS,
        meta={"sampling_rate": target_hz, "window_seconds": window_seconds, "devices": list(devices)},
    )
