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
    'stairsdown': 5
}

def segment_device_data(
    acc_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    model_name: str,
    window_size: int = 128,
    step_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segments accelerometer and gyroscope readings for a specific device model into windows [N, 6, window_size].
    """
    # Filter by model and valid activity labels
    acc_sub = acc_df[(acc_df['Model'] == model_name) & (acc_df['gt'].isin(ACTIVITY_MAP.keys()))].copy()
    gyro_sub = gyro_df[(gyro_df['Model'] == model_name) & (gyro_df['gt'].isin(ACTIVITY_MAP.keys()))].copy()

    # Sort by creation time
    acc_sub.sort_values(by=['User', 'Creation_Time'], inplace=True)
    gyro_sub.sort_values(by=['User', 'Creation_Time'], inplace=True)

    windows = []
    labels = []
    user_ids = []

    # Map user strings 'a'..'i' to 0..8
    user_map = {u: i for i, u in enumerate(sorted(acc_sub['User'].unique()))}

    # Segment per user to avoid mixing different subjects across window boundaries
    for user in acc_sub['User'].unique():
        u_acc = acc_sub[acc_sub['User'] == user]
        u_gyro = gyro_sub[gyro_sub['User'] == user]

        # Interpolate gyro onto acc time grid or merge asof
        merged = pd.merge_asof(
            u_acc[['Creation_Time', 'x', 'y', 'z', 'gt']],
            u_gyro[['Creation_Time', 'x', 'y', 'z']],
            on='Creation_Time',
            suffixes=('_acc', '_gyro'),
            direction='nearest',
            tolerance=50000000 # 50 ms tolerance
        ).dropna()

        if len(merged) < window_size:
            continue

        raw_signals = merged[['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']].values
        gt_labels = merged['gt'].map(ACTIVITY_MAP).values

        for start in range(0, len(merged) - window_size + 1, step_size):
            end = start + window_size
            win = raw_signals[start:end].T # [6, window_size]
            lbl = gt_labels[start:end]

            # Label of window is the majority label; drop if mixed activities in transition
            majority_lbl = np.bincount(lbl).argmax()
            if np.mean(lbl == majority_lbl) >= 0.75:
                windows.append(win)
                labels.append(majority_lbl)
                user_ids.append(user_map[user])

    if len(windows) == 0:
        return np.empty((0, 6, window_size)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=int), np.array(user_ids, dtype=int)

def load_hhar(
    data_dir: str = "Data/heterogeneity/Activity recognition exp",
    source_model: str = "s3",
    target_model: str = "nexus4",
    window_size: int = 128,
    step_size: int = 64
) -> Tuple[BaseHARDataset, BaseHARDataset]:
    """
    Loads HHAR dataset and splits by device model into Source and Target domains.
    e.g. source_model='s3' (~100Hz) -> target_model='nexus4' (~200Hz).
    """
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"hhar_{source_model}_to_{target_model}_w{window_size}.npz")

    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        x_src, y_src, u_src = cached["x_src"], cached["y_src"], cached["u_src"]
        x_tgt, y_tgt, u_tgt = cached["x_tgt"], cached["y_tgt"], cached["u_tgt"]
    else:
        acc_file = os.path.join(data_dir, "Phones_accelerometer.csv")
        gyro_file = os.path.join(data_dir, "Phones_gyroscope.csv")

        acc_df = pd.read_csv(acc_file, usecols=['Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'gt'])
        gyro_df = pd.read_csv(gyro_file, usecols=['Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'gt'])

        x_src, y_src, u_src = segment_device_data(acc_df, gyro_df, source_model, window_size, step_size)
        x_tgt, y_tgt, u_tgt = segment_device_data(acc_df, gyro_df, target_model, window_size, step_size)

        np.savez_compressed(
            cache_file,
            x_src=x_src, y_src=y_src, u_src=u_src,
            x_tgt=x_tgt, y_tgt=y_tgt, u_tgt=u_tgt
        )

    # Z-score normalization based strictly on Source statistics
    mean_src = np.mean(x_src, axis=(0, 2), keepdims=True)
    std_src = np.std(x_src, axis=(0, 2), keepdims=True) + 1e-8

    x_src = (x_src - mean_src) / std_src
    x_tgt = (x_tgt - mean_src) / std_src

    # Approximate sampling rates based on hardware specifications
    sr_map = {"nexus4": 200.0, "s3": 100.0, "s3mini": 100.0, "samsungold": 50.0, "gear": 50.0, "lgwatch": 50.0}

    source_dataset = BaseHARDataset(
        data=torch.from_numpy(x_src),
        labels=torch.from_numpy(y_src),
        domain_ids=torch.from_numpy(u_src),
        sampling_rate=sr_map.get(source_model, 100.0)
    )

    target_dataset = BaseHARDataset(
        data=torch.from_numpy(x_tgt),
        labels=torch.from_numpy(y_tgt),
        domain_ids=torch.from_numpy(u_tgt),
        sampling_rate=sr_map.get(target_model, 200.0)
    )

    return source_dataset, target_dataset
