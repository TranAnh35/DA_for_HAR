import os
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
import torch
from .base_loader import BaseHARDataset

# 12 standard protocol activities in PAMAP2
PAMAP2_ACTIVITY_MAP: Dict[int, int] = {
    1: 0,   # lying
    2: 1,   # sitting
    3: 2,   # standing
    4: 3,   # walking
    5: 4,   # running
    6: 5,   # cycling
    7: 6,   # Nordic walking
    12: 7,  # ascending stairs
    13: 8,  # descending stairs
    16: 9,  # vacuum cleaning
    17: 10, # ironing
    24: 11  # rope jumping
}

POSITION_COLS: Dict[str, List[int]] = {
    # 3D accel (+-16g) + 3D gyro
    "wrist": [4, 5, 6, 10, 11, 12],
    "chest": [21, 22, 23, 27, 28, 29],
    "ankle": [38, 39, 40, 44, 45, 46]
}

def process_subject_file(
    file_path: str,
    subject_id: int,
    source_cols: List[int],
    target_cols: List[int],
    window_size: int = 128,
    step_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a single PAMAP2 subject .dat file and extracts synchronized windows for source & target positions.
    """
    # Load file using pandas with space delimiter
    usecols = [1] + source_cols + target_cols
    # Remove duplicates if source_cols == target_cols
    unique_usecols = sorted(list(set(usecols)))
    
    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=None,
        usecols=unique_usecols,
        dtype=np.float32
    )

    # Filter out transient activities (activity 0) and non-protocol activities
    act_col = df[1].astype(int)
    valid_mask = act_col.isin(PAMAP2_ACTIVITY_MAP.keys())
    df = df[valid_mask].copy()

    if len(df) < window_size:
        return np.empty((0, 6, window_size)), np.empty((0, 6, window_size)), np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    # Interpolate NaNs
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(0.0, inplace=True)

    labels = df[1].astype(int).map(PAMAP2_ACTIVITY_MAP).values
    src_data = df[source_cols].values # [N, 6]
    tgt_data = df[target_cols].values # [N, 6]

    src_windows = []
    tgt_windows = []
    win_labels = []
    win_subjs = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        lbl_chunk = labels[start:end]

        # Majority label rule
        majority_lbl = np.bincount(lbl_chunk).argmax()
        if np.mean(lbl_chunk == majority_lbl) >= 0.8:
            src_windows.append(src_data[start:end].T) # [6, window_size]
            tgt_windows.append(tgt_data[start:end].T) # [6, window_size]
            win_labels.append(majority_lbl)
            win_subjs.append(subject_id)

    if len(src_windows) == 0:
        return np.empty((0, 6, window_size)), np.empty((0, 6, window_size)), np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    return (
        np.array(src_windows, dtype=np.float32),
        np.array(tgt_windows, dtype=np.float32),
        np.array(win_labels, dtype=int),
        np.array(win_subjs, dtype=int)
    )

def load_pamap2(
    data_dir: str = "Data/PAMAP2_Dataset",
    source_position: str = "chest",
    target_position: str = "wrist",
    window_size: int = 128,
    step_size: int = 64
) -> Tuple[BaseHARDataset, BaseHARDataset]:
    """
    Loads PAMAP2 dataset and partitions by on-body sensor location (Cross-Placement).
    e.g. source_position='chest' -> target_position='wrist'.
    """
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"pamap2_{source_position}_to_{target_position}_w{window_size}.npz")

    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        x_src, x_tgt = cached["x_src"], cached["x_tgt"]
        y_all, subj_all = cached["y_all"], cached["subj_all"]
    else:
        protocol_dir = os.path.join(data_dir, "Protocol")
        src_cols = POSITION_COLS[source_position.lower()]
        tgt_cols = POSITION_COLS[target_position.lower()]

        all_src_wins = []
        all_tgt_wins = []
        all_labels = []
        all_subjs = []

        subject_files = [f"subject10{i}.dat" for i in range(1, 9)] # subjects 101 to 108

        for subj_idx, s_file in enumerate(subject_files):
            f_path = os.path.join(protocol_dir, s_file)
            if not os.path.exists(f_path):
                continue
            src_w, tgt_w, lbls, subjs = process_subject_file(
                f_path, subj_idx, src_cols, tgt_cols, window_size, step_size
            )
            if len(src_w) > 0:
                all_src_wins.append(src_w)
                all_tgt_wins.append(tgt_w)
                all_labels.append(lbls)
                all_subjs.append(subjs)

        x_src = np.concatenate(all_src_wins, axis=0)
        x_tgt = np.concatenate(all_tgt_wins, axis=0)
        y_all = np.concatenate(all_labels, axis=0)
        subj_all = np.concatenate(all_subjs, axis=0)

        np.savez_compressed(
            cache_file,
            x_src=x_src, x_tgt=x_tgt, y_all=y_all, subj_all=subj_all
        )

    # Z-score normalization computed strictly on Source statistics
    mean_src = np.mean(x_src, axis=(0, 2), keepdims=True)
    std_src = np.std(x_src, axis=(0, 2), keepdims=True) + 1e-8

    x_src = (x_src - mean_src) / std_src
    x_tgt = (x_tgt - mean_src) / std_src

    source_dataset = BaseHARDataset(
        data=torch.from_numpy(x_src),
        labels=torch.from_numpy(y_all),
        domain_ids=torch.from_numpy(subj_all),
        sampling_rate=100.0
    )

    target_dataset = BaseHARDataset(
        data=torch.from_numpy(x_tgt),
        labels=torch.from_numpy(y_all),
        domain_ids=torch.from_numpy(subj_all),
        sampling_rate=100.0
    )

    return source_dataset, target_dataset
