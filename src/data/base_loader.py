import abc
from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader

class BaseHARDataset(Dataset):
    """
    Abstract Base Dataset class for Human Activity Recognition (HAR).
    Guarantees consistent data shapes and metadata across all benchmark datasets.
    """
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, domain_ids: torch.Tensor, sampling_rate: float):
        """
        data: [N, Channels, Seq_Len]
        labels: [N]
        domain_ids: [N] (e.g. subject_id or device_id)
        sampling_rate: Sampling frequency in Hz (e.g. 50.0, 100.0, 200.0)
        """
        assert data.ndim == 3, f"Expected data of shape [N, C, T], got {data.shape}"
        assert len(data) == len(labels) == len(domain_ids), "Length mismatch across tensors"
        self.data = data.float()
        self.labels = labels.long()
        self.domain_ids = domain_ids.long()
        self.sampling_rate = float(sampling_rate)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        return self.data[idx], self.labels[idx], self.domain_ids[idx], self.sampling_rate
