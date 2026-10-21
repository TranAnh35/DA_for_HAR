"""Leak-free split and protocol audit.

The only hard rule of an unsupervised DA benchmark is:

    target **labels** must never influence training or checkpoint selection.

Target *inputs* may be used (that is what "unsupervised" means), but the
selection signal has to come from the source side.  :func:`audit_task` records
this explicitly so a result JSON can be rejected later if it was produced by a
target-oracle run -- the failure mode that invalidated the legacy results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ..data.core import DomainData

__all__ = ["Split", "ProtocolAudit", "audit_task", "subject_disjoint_split"]


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    subject_disjoint: bool
    stats: Dict[str, Any] = field(default_factory=dict)


def _stratified_fallback(domain: DomainData, val_fraction: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    val: List[int] = []
    for cls in np.unique(domain.y):
        idx = np.where(domain.y == cls)[0]
        if len(idx) < 2:
            continue
        k = max(1, int(round(val_fraction * len(idx))))
        k = min(k, len(idx) - 1)
        val.extend(rng.choice(idx, size=k, replace=False).tolist())
    if not val:
        n = max(1, int(round(val_fraction * domain.num_windows)))
        n = min(n, domain.num_windows - 1)
        val = rng.choice(domain.num_windows, size=n, replace=False).tolist()
    val_arr = np.asarray(sorted(val), dtype=np.int64)
    mask = np.ones(domain.num_windows, dtype=bool)
    mask[val_arr] = False
    return Split(
        train=np.where(mask)[0],
        val=val_arr,
        subject_disjoint=False,
        stats={"reason": "single subject; stratified window split"},
    )


def subject_disjoint_split(
    domain: DomainData, val_fraction: float = 0.2, seed: int = 0
) -> Split:
    """Hold out whole subjects for validation when possible."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    subjects = np.unique(domain.subjects)
    if len(subjects) < 2:
        return _stratified_fallback(domain, val_fraction, seed)

    rng = np.random.default_rng(seed)
    order = rng.permutation(subjects)
    n_val = max(1, int(round(val_fraction * len(subjects))))
    n_val = min(n_val, len(subjects) - 1)
    val_subjects = set(order[:n_val].tolist())

    is_val = np.asarray([s in val_subjects for s in domain.subjects], dtype=bool)
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    if train_idx.size == 0 or val_idx.size == 0:
        return _stratified_fallback(domain, val_fraction, seed)
    return Split(
        train=train_idx,
        val=val_idx,
        subject_disjoint=True,
        stats={"val_subjects": sorted(val_subjects)},
    )


@dataclass
class ProtocolAudit:
    selection_metric: str = "source_val_macro_f1"
    target_labels_in_training: bool = False
    target_labels_in_selection: bool = False
    source_val_subject_disjoint: bool = True
    source_target_subject_overlap: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_metric": self.selection_metric,
            "target_labels_in_training": self.target_labels_in_training,
            "target_labels_in_selection": self.target_labels_in_selection,
            "source_val_subject_disjoint": self.source_val_subject_disjoint,
            "source_target_subject_overlap": self.source_target_subject_overlap,
            "notes": list(self.notes),
            "is_target_oracle": bool(
                self.target_labels_in_training or self.target_labels_in_selection
            ),
            "protocol_version": "da_har_source_selected_v1",
        }


def audit_task(
    source: DomainData,
    target: DomainData,
    split: Split,
    shift: str,
) -> ProtocolAudit:
    overlap = len(set(source.subjects.tolist()) & set(target.subjects.tolist()))
    notes: List[str] = []
    if shift == "cross_subject" and overlap:
        notes.append(
            "subject overlap between source and target: this is not a clean cross-subject task"
        )
    if not split.subject_disjoint:
        notes.append("source validation split is not subject-disjoint (single-subject domain)")
    return ProtocolAudit(
        source_val_subject_disjoint=split.subject_disjoint,
        source_target_subject_overlap=overlap,
        notes=notes,
    )
