import os
from typing import Tuple, List, Optional
import numpy as np
import torch
from .base_loader import BaseHARDataset

def read_signals(data_dir: str, subset: str, signal_types: List[str]) -> np.ndarray:
    """
    Read raw inertial signals from UCI HAR text files.
    Returns: numpy array of shape [N, num_channels, 128]
    """
    signals = []
    signals_dir = os.path.join(data_dir, subset, "Inertial Signals")
    for st in signal_types:
        filename = os.path.join(signals_dir, f"{st}_{subset}.txt")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Signal file not found: {filename}")
        # Each row has 128 float values
        data = np.loadtxt(filename)
        signals.append(data)
    # Stack along channel axis: [num_channels, N, 128] -> transpose to [N, num_channels, 128]
    stacked = np.stack(signals, axis=1)
    return stacked

def load_uci_har(
    data_dir: str = "Data/UCI_HAR/UCI HAR Dataset",
    source_subjects: Optional[List[int]] = None,
    target_subjects: Optional[List[int]] = None,
    channels: str = "6ch"
) -> Tuple[BaseHARDataset, BaseHARDataset]:
    """
    Loads UCI HAR dataset and splits strictly by subject ID into Source and Target domains.
    
    channels:
        - "6ch": [total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z]
        - "9ch": including body_acc
    """
    if source_subjects is None:
        # Default cross-subject split: Subjects 1..20 for Source, 21..30 for Target
        source_subjects = list(range(1, 21))
    if target_subjects is None:
        target_subjects = list(range(21, 31))

    if channels == "6ch":
        signal_types = [
            "total_acc_x", "total_acc_y", "total_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z"
        ]
    else:
        signal_types = [
            "total_acc_x", "total_acc_y", "total_acc_z",
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z"
        ]

    cache_file = os.path.join(data_dir, f"cache_{channels}.npz")
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        x_all = cached["x_all"]
        y_all = cached["y_all"]
        subj_all = cached["subj_all"]
    else:
        # UCI HAR comes partitioned into train and test folders.
        # To perform proper Cross-Subject Domain Adaptation without leakage,
        # we merge all samples, then split strictly by subject_id.
        x_train = read_signals(data_dir, "train", signal_types)
        y_train = np.loadtxt(os.path.join(data_dir, "train", "y_train.txt"), dtype=int) - 1 # 0-indexed [0..5]
        subj_train = np.loadtxt(os.path.join(data_dir, "train", "subject_train.txt"), dtype=int)

        x_test = read_signals(data_dir, "test", signal_types)
        y_test = np.loadtxt(os.path.join(data_dir, "test", "y_test.txt"), dtype=int) - 1
        subj_test = np.loadtxt(os.path.join(data_dir, "test", "subject_test.txt"), dtype=int)

        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        subj_all = np.concatenate([subj_train, subj_test], axis=0)

        # Save cache for ultra-fast subsequent loads
        np.savez_compressed(cache_file, x_all=x_all, y_all=y_all, subj_all=subj_all)

    # Filter source and target
    src_mask = np.isin(subj_all, source_subjects)
    tgt_mask = np.isin(subj_all, target_subjects)

    x_src, y_src, subj_src = x_all[src_mask], y_all[src_mask], subj_all[src_mask]
    x_tgt, y_tgt, subj_tgt = x_all[tgt_mask], y_all[tgt_mask], subj_all[tgt_mask]

    # Z-score normalization computed strictly on source domain (prevent target leakage)
    mean_src = np.mean(x_src, axis=(0, 2), keepdims=True)
    std_src = np.std(x_src, axis=(0, 2), keepdims=True) + 1e-8

    x_src = (x_src - mean_src) / std_src
    x_tgt = (x_tgt - mean_src) / std_src

    source_dataset = BaseHARDataset(
        data=torch.from_numpy(x_src),
        labels=torch.from_numpy(y_src),
        domain_ids=torch.from_numpy(subj_src),
        sampling_rate=50.0
    )

    target_dataset = BaseHARDataset(
        data=torch.from_numpy(x_tgt),
        labels=torch.from_numpy(y_tgt),
        domain_ids=torch.from_numpy(subj_tgt),
        sampling_rate=50.0
    )

    return source_dataset, target_dataset
