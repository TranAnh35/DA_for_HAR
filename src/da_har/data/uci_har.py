"""UCI-HAR loader (already windowed, 2.56 s @ 50 Hz).

The dataset ships pre-segmented windows, so this loader does **not** re-window.
It does, however, attach the subject id to every window, which the benchmark
needs in order to build subject-disjoint source/target domains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .core import ChannelSpec, DatasetBundle, DomainData, make_imu_channels

LABELS: Tuple[str, ...] = (
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "sitting",
    "standing",
    "laying",
)

SIGNAL_SETS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # (file stem, kind)
    "6ch": (
        ("total_acc", "total_acc"),
        ("body_gyro", "body_gyro"),
    ),
    "9ch": (
        ("total_acc", "total_acc"),
        ("body_acc", "body_acc"),
        ("body_gyro", "body_gyro"),
    ),
}

SAMPLING_RATE = 50.0
WINDOW_SECONDS = 2.56


def _channels(signal_sets: Sequence[Tuple[str, str]]) -> Tuple[ChannelSpec, ...]:
    specs = []
    for stem, kind in signal_sets:
        specs.extend(make_imu_channels(role="imu0", placement="waist", kinds=(kind,), unit="g"))
        # make_imu_channels cycles axes internally, so the result is x,y,z.
    return tuple(specs)


def _read_subset(base: Path, subset: str, signal_sets: Sequence[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    signals = []
    for stem, _ in signal_sets:
        for axis in ("x", "y", "z"):
            path = base / subset / "Inertial Signals" / f"{stem}_{axis}_{subset}.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            signals.append(np.loadtxt(path, dtype=np.float32))
    X = np.stack(signals, axis=1)  # (N, C, 128)
    y = np.loadtxt(base / subset / f"y_{subset}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(base / subset / f"subject_{subset}.txt", dtype=np.int64)
    return {"X": X, "y": y, "subjects": subjects}


def load_bundle(
    root: str | Path = "Data",
    *,
    channel_set: str = "9ch",
    max_windows_per_domain: Optional[int] = None,
    seed: int = 0,
) -> DatasetBundle:
    base = Path(root) / "UCI_HAR" / "UCI HAR Dataset"
    if not base.exists():
        raise FileNotFoundError(f"UCI-HAR not found at {base}")

    signal_sets = SIGNAL_SETS[channel_set]
    channels = _channels(signal_sets)

    parts = [_read_subset(base, subset, signal_sets) for subset in ("train", "test")]
    X = np.concatenate([p["X"] for p in parts], axis=0)
    y = np.concatenate([p["y"] for p in parts], axis=0)
    subjects = np.concatenate([p["subjects"] for p in parts], axis=0)

    domains: Dict[str, DomainData] = {}
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        dom = DomainData(
            name=f"subject:{int(subject):02d}",
            X=X[idx],
            y=y[idx],
            subjects=np.full(len(idx), f"{int(subject):02d}"),
            sessions=np.full(len(idx), "default"),
            channels=channels,
            sampling_rate=SAMPLING_RATE,
            meta={"dataset": "uci_har", "subject": int(subject)},
        )
        if max_windows_per_domain is not None:
            dom = dom.take(max_windows_per_domain, seed=seed)
        domains[dom.name] = dom

    return DatasetBundle(
        name="uci_har",
        domains=domains,
        label_names=LABELS,
        meta={
            "sampling_rate": SAMPLING_RATE,
            "window_seconds": WINDOW_SECONDS,
            "pre_windowed": True,
            "channel_set": channel_set,
        },
    )
