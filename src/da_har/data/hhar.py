"""HHAR loader with bounded-memory CSV streaming.

HHAR is the only dataset in ``Data/`` large enough to break a naive loader:
``Phones_accelerometer.csv`` is 1.23 GB and ``Phones_gyroscope.csv`` is
1.32 GB.  The legacy loader called ``pd.read_csv`` on the whole file, which
pulls ~2.5 GB of text plus object-dtype columns into RAM.

This loader reads the CSVs in chunks with explicit dtypes, keeps only numeric
columns plus category codes, and accumulates per ``(user, model, activity)``
segments.  Peak memory is therefore bounded by one chunk plus the numeric
samples we actually need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import ChannelSpec, DatasetBundle, DomainData, make_imu_channels
from .segments import contiguous_segments, windows_from_segments

LABELS: Tuple[str, ...] = ("bike", "sit", "stand", "walk", "stairsup", "stairsdown")
LABEL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(LABELS)}

SAMPLING_RATE = 100.0
WINDOW_SECONDS = 1.28
CACHE_VERSION = 1

_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("Phones_accelerometer.csv", "acc", "waist"),
    ("Phones_gyroscope.csv", "gyro", "waist"),
    ("Watch_accelerometer.csv", "acc", "wrist"),
    ("Watch_gyroscope.csv", "gyro", "wrist"),
)

_COLUMNS = ["Creation_Time", "x", "y", "z", "User", "Model", "gt"]
_DTYPES = {
    "Creation_Time": "int64",
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "User": "string",
    "Model": "string",
    "gt": "string",
}


def _stream_by_key(
    path: Path,
    models: Optional[Sequence[str]],
    chunk_size: int,
    max_rows: Optional[int] = None,
) -> Tuple[Dict[Tuple[str, str, str], List[np.ndarray]], Dict[str, str]]:
    """Read one CSV and group samples by ``(user, model, activity)``.

    Returns the grouped ``[timestamp, x, y, z]`` arrays and a mapping from model
    to the physical placement implied by the file it came from.
    """
    grouped: Dict[Tuple[str, str, str], List[np.ndarray]] = {}
    placement_by_model: Dict[str, str] = {}
    if not path.exists():
        return grouped, placement_by_model

    placement = next((p for name, _, p in _SOURCES if name == path.name), "waist")
    rows = 0
    reader = pd.read_csv(
        path,
        usecols=_COLUMNS,
        dtype=_DTYPES,
        chunksize=chunk_size,
        engine="c",
    )
    for chunk in reader:
        chunk = chunk[chunk["gt"].isin(LABELS)]
        if models is not None:
            chunk = chunk[chunk["Model"].isin(list(models))]
        if chunk.empty:
            continue
        for (user, model, gt), grp in chunk.groupby(["User", "Model", "gt"], observed=True):
            key = (str(user), str(model), str(gt))
            arr = grp[["Creation_Time", "x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)
            grouped.setdefault(key, []).append(arr)
            placement_by_model[str(model)] = placement
        rows += len(chunk)
        if max_rows is not None and rows >= max_rows:
            break
    return grouped, placement_by_model


def _concat_sorted(parts: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.concatenate(parts, axis=0)
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    # Drop duplicate timestamps (HHAR has a few).
    if arr.shape[0] > 1:
        keep = np.ones(arr.shape[0], dtype=bool)
        keep[1:] = arr[1:, 0] != arr[:-1, 0]
        arr = arr[keep]
    return arr


def _split_runs(timestamps: np.ndarray, max_gap_ns: float) -> List[Tuple[int, int]]:
    """Split a sorted timestamp array into gap-free ``[start, end)`` runs."""
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size == 0:
        return []
    cuts = np.where(np.diff(t) > max_gap_ns)[0]
    starts = np.concatenate([[0], cuts + 1]).astype(int)
    ends = np.concatenate([cuts + 1, [t.size]]).astype(int)
    return list(zip(starts.tolist(), ends.tolist()))


def _build_model_domain(
    model: str,
    acc: Dict[Tuple[str, str, str], List[np.ndarray]],
    gyro: Dict[Tuple[str, str, str], List[np.ndarray]],
    placement: str,
    target_hz: float,
    window_seconds: float,
    stride_seconds: Optional[float],
    min_label_fraction: float,
    max_windows: Optional[int],
    max_gap_seconds: float = 0.25,
) -> Optional[DomainData]:
    channels: Tuple[ChannelSpec, ...] = make_imu_channels(
        role="imu0", placement=placement, kinds=("acc", "gyro")
    )
    xs, ys, subs, sess = [], [], [], []
    max_gap_ns = max_gap_seconds * 1e9
    step_ns = 1e9 / target_hz
    min_run_ns = window_seconds * 1e9

    keys = sorted(k for k in acc if k[1] == model and k in gyro)
    for user, _, activity in keys:
        a = _concat_sorted(acc[(user, model, activity)])
        g = _concat_sorted(gyro[(user, model, activity)])
        if a.shape[0] < 2 or g.shape[0] < 2:
            continue

        # Resample only inside runs where *both* sensors are gap-free.  Doing
        # this on the whole (user, model, activity) group would interpolate
        # across hours-long recording gaps and manufacture millions of
        # duplicated windows (observed: 8M windows from 2M rows).
        a_runs = _split_runs(a[:, 0], max_gap_ns)
        g_runs = _split_runs(g[:, 0], max_gap_ns)
        produced = 0
        for a_start, a_end in a_runs:
            for g_start, g_end in g_runs:
                lo = max(a[a_start, 0], g[g_start, 0])
                hi = min(a[a_end - 1, 0], g[g_end - 1, 0])
                if hi - lo < min_run_ns:
                    continue
                n = int(np.floor((hi - lo) / step_ns)) + 1
                if n < 2:
                    continue
                grid = lo + np.arange(n, dtype=np.float64) * step_ns
                a_i = np.stack(
                    [np.interp(grid, a[a_start:a_end, 0], a[a_start:a_end, c]) for c in (1, 2, 3)],
                    axis=1,
                )
                g_i = np.stack(
                    [np.interp(grid, g[g_start:g_end, 0], g[g_start:g_end, c]) for c in (1, 2, 3)],
                    axis=1,
                )
                signal = np.concatenate([a_i, g_i], axis=1).astype(np.float32)

                label = LABEL_INDEX[activity]
                labels = np.full(signal.shape[0], label, dtype=np.int64)
                X, y = windows_from_segments(
                    signal,
                    labels,
                    [(0, signal.shape[0])],
                    sampling_rate=target_hz,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    min_label_fraction=min_label_fraction,
                )
                if X.size == 0:
                    continue
                xs.append(X)
                ys.append(y)
                subs.append(np.full(len(y), user))
                sess.append(np.full(len(y), "default"))
                produced += len(y)
        if max_windows is not None and produced >= max_windows:
            break

    if not xs:
        return None
    n = min(max_windows, sum(len(v) for v in ys)) if max_windows is not None else None
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subjects = np.concatenate(subs)
    sessions = np.concatenate(sess)
    if n is not None:
        X, y, subjects, sessions = X[:n], y[:n], subjects[:n], sessions[:n]
    return DomainData(
        name=f"device:{model}",
        X=X,
        y=y,
        subjects=subjects,
        sessions=sessions,
        channels=channels,
        sampling_rate=target_hz,
        meta={"dataset": "hhar", "device": model, "placement": placement},
    )


def load_bundle(
    root: str | Path = "Data",
    *,
    target_hz: float = SAMPLING_RATE,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: Optional[float] = None,
    min_label_fraction: float = 0.9,
    models: Optional[Sequence[str]] = None,
    chunk_size: int = 1_000_000,
    max_rows_per_file: Optional[int] = None,
    max_windows_per_domain: Optional[int] = None,
    use_cache: bool = True,
) -> DatasetBundle:
    base = Path(root) / "heterogeneity" / "Activity recognition exp"
    if not base.exists():
        raise FileNotFoundError(f"HHAR not found at {base}")

    cache_dir = base / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"da_hhar_fs{target_hz:g}_w{window_seconds:g}"
        f"_s{(stride_seconds if stride_seconds else window_seconds / 2.0):g}"
        f"_p{min_label_fraction:g}_v{CACHE_VERSION}"
    )
    if models is not None:
        tag += "_m" + "-".join(sorted(str(m) for m in models))
    if max_rows_per_file is not None:
        tag += f"_r{max_rows_per_file}"
    cache_file = cache_dir / f"{tag}.npz"

    if use_cache and cache_file.exists() and max_windows_per_domain is None:
        data = np.load(cache_file, allow_pickle=True)
        domains = {}
        for model in data["models"].tolist():
            X = data[f"X_{model}"]
            if X.size == 0:
                continue
            domains[f"device:{model}"] = DomainData(
                name=f"device:{model}",
                X=X,
                y=data[f"y_{model}"],
                subjects=data[f"subjects_{model}"].astype(str),
                sessions=np.full(len(X), "default"),
                channels=make_imu_channels(
                    role="imu0",
                    placement=str(data[f"placement_{model}"]),
                    kinds=("acc", "gyro"),
                ),
                sampling_rate=target_hz,
                meta={"dataset": "hhar", "device": model, "cached": True},
            )
    else:
        acc_all: Dict[Tuple[str, str, str], List[np.ndarray]] = {}
        gyro_all: Dict[Tuple[str, str, str], List[np.ndarray]] = {}
        placement_by_model: Dict[str, str] = {}
        for filename, kind, _ in _SOURCES:
            grouped, placements = _stream_by_key(
                base / filename, models, chunk_size, max_rows=max_rows_per_file
            )
            (acc_all if kind == "acc" else gyro_all).update(grouped)
            for model, place in placements.items():
                placement_by_model.setdefault(model, place)

        all_models = sorted({m for _, m, _ in acc_all} & {m for _, m, _ in gyro_all})
        if not all_models:
            raise ValueError("HHAR produced no overlapping accelerometer/gyroscope models")

        domains = {}
        for model in all_models:
            dom = _build_model_domain(
                model,
                acc_all,
                gyro_all,
                placement_by_model.get(model, "waist"),
                target_hz,
                window_seconds,
                stride_seconds,
                min_label_fraction,
                max_windows_per_domain,
            )
            if dom is not None:
                domains[dom.name] = dom

        if use_cache and max_windows_per_domain is None and domains:
            payload = {"models": np.asarray(sorted(d.name.split(":", 1)[1] for d in domains.values()))}
            for dom in domains.values():
                key = dom.name.split(":", 1)[1]
                payload[f"X_{key}"] = dom.X
                payload[f"y_{key}"] = dom.y
                payload[f"subjects_{key}"] = dom.subjects
                payload[f"placement_{key}"] = np.asarray(dom.channels[0].placement)
            np.savez_compressed(cache_file, **payload)

    if not domains:
        raise ValueError("HHAR produced no windows; check the raw CSVs and model names")

    return DatasetBundle(
        name="hhar",
        domains=domains,
        label_names=LABELS,
        meta={
            "sampling_rate": target_hz,
            "window_seconds": window_seconds,
            "streamed": max_rows_per_file,
        },
    )
