import os
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import torch

from .base_loader import BaseHARDataset

ACTIVITY_MAP: Dict[str, int] = {
    'bike': 0,
    'sit': 1,
    'stand': 2,
    'walk': 3,
    'stairsup': 4,
    'stairsdown': 5,
}

# Bump when the segmentation protocol changes so stale caches are not reused.
CACHE_VERSION = 2


def _resample_user(
    u_acc: pd.DataFrame,
    u_gyro: pd.DataFrame,
    target_hz: float,
    window_size: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resample one user's accelerometer/gyroscope onto a common uniform grid.

    HHAR devices sample at different rates (e.g. S3 ~100 Hz, Nexus 4 ~200 Hz).
    Windowing raw samples at a fixed length therefore mixes *different physical
    durations*.  Both signals are linearly interpolated onto the same
    ``target_hz`` grid before windowing.

    Returns ``(signals [6, N], labels [N])`` or ``None`` if the overlap is too
    short.
    """
    if len(u_acc) < 2 or len(u_gyro) < 2:
        return None

    # Work relative to the overlap start to avoid losing precision on ns int64.
    start_ns = max(int(u_acc['Creation_Time'].iloc[0]), int(u_gyro['Creation_Time'].iloc[0]))
    end_ns = min(int(u_acc['Creation_Time'].iloc[-1]), int(u_gyro['Creation_Time'].iloc[-1]))
    duration_s = (end_ns - start_ns) / 1e9
    if duration_s <= 0:
        return None

    n_samples = int(np.floor(duration_s * target_hz)) + 1
    if n_samples < window_size:
        return None

    grid = np.arange(n_samples, dtype=np.float64) / float(target_hz)

    # Relative times in seconds (keeps float precision high).
    t_acc = (u_acc['Creation_Time'].to_numpy(dtype=np.int64) - start_ns) / 1e9
    t_gyro = (u_gyro['Creation_Time'].to_numpy(dtype=np.int64) - start_ns) / 1e9

    axes = ['x', 'y', 'z']
    acc_vals = np.stack(
        [np.interp(grid, t_acc, u_acc[c].to_numpy(dtype=np.float64)) for c in axes],
        axis=0,
    )
    gyro_vals = np.stack(
        [np.interp(grid, t_gyro, u_gyro[c].to_numpy(dtype=np.float64)) for c in axes],
        axis=0,
    )

    # Nearest accelerometer row supplies the activity label.
    idx = np.searchsorted(t_acc, grid)
    idx = np.clip(idx, 0, len(t_acc) - 1)
    idx_prev = np.clip(idx - 1, 0, len(t_acc) - 1)
    use_prev = np.abs(t_acc[idx_prev] - grid) < np.abs(t_acc[idx] - grid)
    idx = np.where(use_prev, idx_prev, idx)
    label_names = u_acc['gt'].to_numpy()[idx]
    labels = np.asarray([ACTIVITY_MAP[name] for name in label_names], dtype=int)

    return np.concatenate([acc_vals, gyro_vals], axis=0).astype(np.float32), labels


def segment_device_data(
    acc_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    model_name: str,
    window_size: int = 128,
    step_size: int = 64,
    target_hz: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment one device model into windows [N, 6, window_size] at a common rate."""
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")

    acc_sub = acc_df[
        (acc_df['Model'] == model_name) & (acc_df['gt'].isin(ACTIVITY_MAP.keys()))
    ].copy()
    gyro_sub = gyro_df[
        (gyro_df['Model'] == model_name) & (gyro_df['gt'].isin(ACTIVITY_MAP.keys()))
    ].copy()

    if len(acc_sub) == 0 or len(gyro_sub) == 0:
        return np.empty((0, 6, window_size), dtype=np.float32), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    acc_sub.dropna(subset=['x', 'y', 'z'], inplace=True)
    gyro_sub.dropna(subset=['x', 'y', 'z'], inplace=True)
    acc_sub.sort_values(by=['User', 'Creation_Time'], inplace=True)
    gyro_sub.sort_values(by=['User', 'Creation_Time'], inplace=True)
    acc_sub.drop_duplicates(subset=['User', 'Creation_Time'], keep='last', inplace=True)
    gyro_sub.drop_duplicates(subset=['User', 'Creation_Time'], keep='last', inplace=True)

    windows: List[np.ndarray] = []
    labels: List[int] = []
    user_ids: List[int] = []

    user_map = {u: i for i, u in enumerate(sorted(acc_sub['User'].unique()))}

    for user in acc_sub['User'].unique():
        u_acc = acc_sub[acc_sub['User'] == user]
        u_gyro = gyro_sub[gyro_sub['User'] == user]
        if len(u_acc) < 2 or len(u_gyro) < 2:
            continue

        resampled = _resample_user(u_acc, u_gyro, target_hz=target_hz, window_size=window_size)
        if resampled is None:
            continue
        raw_signals, gt_series = resampled

        for start in range(0, len(gt_series) - window_size + 1, step_size):
            end = start + window_size
            win = raw_signals[:, start:end]
            lbl = gt_series[start:end]

            # Majority label; drop windows that straddle activity transitions.
            majority_lbl = np.bincount(lbl).argmax()
            if np.mean(lbl == majority_lbl) >= 0.75:
                windows.append(win)
                labels.append(int(majority_lbl))
                user_ids.append(user_map[user])

    if not windows:
        return np.empty((0, 6, window_size), dtype=np.float32), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    return (
        np.stack(windows, axis=0).astype(np.float32),
        np.asarray(labels, dtype=int),
        np.asarray(user_ids, dtype=int),
    )


def load_hhar(
    data_dir: str = "Data/heterogeneity/Activity recognition exp",
    source_model: str = "s3",
    target_model: str = "nexus4",
    window_size: int = 128,
    step_size: int = 64,
    target_hz: float = 100.0,
) -> Tuple[BaseHARDataset, BaseHARDataset]:
    """Load HHAR as a cross-device source/target pair at a common sampling rate.

    Both devices are resampled to ``target_hz`` before windowing, so a window
    of 128 samples has the same physical duration (1.28 s at 100 Hz) in both
    domains.  Source statistics are used for z-score normalization of both.
    """
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"hhar_{source_model}_to_{target_model}_w{window_size}_s{step_size}"
        f"_fs{target_hz:g}_v{CACHE_VERSION}.npz",
    )

    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        x_src, y_src, u_src = cached["x_src"], cached["y_src"], cached["u_src"]
        x_tgt, y_tgt, u_tgt = cached["x_tgt"], cached["y_tgt"], cached["u_tgt"]
    else:
        acc_file = os.path.join(data_dir, "Phones_accelerometer.csv")
        gyro_file = os.path.join(data_dir, "Phones_gyroscope.csv")

        acc_df = pd.read_csv(acc_file, usecols=['Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'gt'])
        gyro_df = pd.read_csv(gyro_file, usecols=['Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'gt'])

        x_src, y_src, u_src = segment_device_data(
            acc_df, gyro_df, source_model, window_size, step_size, target_hz=target_hz
        )
        x_tgt, y_tgt, u_tgt = segment_device_data(
            acc_df, gyro_df, target_model, window_size, step_size, target_hz=target_hz
        )

        np.savez_compressed(
            cache_file,
            x_src=x_src, y_src=y_src, u_src=u_src,
            x_tgt=x_tgt, y_tgt=y_tgt, u_tgt=u_tgt,
        )

    if len(x_src) == 0 or len(x_tgt) == 0:
        raise ValueError(
            f"HHAR produced empty data for {source_model}->{target_model} "
            f"at {target_hz:g} Hz. Check the raw CSV paths and device names."
        )

    # Z-score normalization based strictly on source statistics.
    mean_src = np.mean(x_src, axis=(0, 2), keepdims=True)
    std_src = np.std(x_src, axis=(0, 2), keepdims=True) + 1e-8
    x_src = (x_src - mean_src) / std_src
    x_tgt = (x_tgt - mean_src) / std_src

    source_dataset = BaseHARDataset(
        data=torch.from_numpy(x_src),
        labels=torch.from_numpy(y_src),
        domain_ids=torch.from_numpy(u_src),
        sampling_rate=target_hz,
    )
    target_dataset = BaseHARDataset(
        data=torch.from_numpy(x_tgt),
        labels=torch.from_numpy(y_tgt),
        domain_ids=torch.from_numpy(u_tgt),
        sampling_rate=target_hz,
    )
    return source_dataset, target_dataset
