"""Shared helpers for label-free model selection and evaluation.

The benchmark protocol must never use target labels for checkpoint selection.
This module provides:

* ``split_source_train_val``: subject/domain-held-out split of the *source*
  domain only (target data is never touched).
* ``evaluate``: generic evaluation returning metrics and optional loss.
* ``is_improvement`` / ``selection_score``: consistent selection semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.GaugeDA.utils.metrics import compute_metrics


def split_source_train_val(
    dataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset, Dict[str, Any]]:
    """Split a source dataset into train/validation without touching target data.

    When the dataset exposes ``domain_ids`` (subject or device id), the split is
    made at the domain level so validation subjects are unseen during training.
    Otherwise it falls back to a random sample split.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n = len(dataset)
    if n < 2:
        raise ValueError("Source dataset is too small to create a validation split")

    rng = np.random.default_rng(seed)
    domain_ids = getattr(dataset, "domain_ids", None)

    if domain_ids is not None and len(domain_ids) == n:
        domains = np.asarray(domain_ids.cpu() if isinstance(domain_ids, torch.Tensor) else domain_ids)
        unique = np.unique(domains)
        labels = getattr(dataset, "labels", None)
        labels_np = None
        if labels is not None and len(labels) == n:
            labels_np = np.asarray(labels.cpu() if isinstance(labels, torch.Tensor) else labels)

        if len(unique) >= 5:
            # Identify classes that occur in only one domain (e.g. PAMAP2
            # class 11 occurs only in subject109).  Those domains must stay in
            # training, otherwise the model never sees the class.
            forced_train = set()
            rare_classes = []
            class_domains = {}
            if labels_np is not None:
                for c in np.unique(labels_np):
                    doms = set(np.unique(domains[labels_np == c]).tolist())
                    class_domains[int(c)] = doms
                    if len(doms) == 1:
                        forced_train.update(doms)
                        rare_classes.append(int(c))

            candidates = [int(d) for d in unique if int(d) not in forced_train]
            n_val_domains = max(1, int(round(len(unique) * val_fraction)))

            if len(candidates) >= 2:
                n_val_domains = min(n_val_domains, len(candidates) - 1)
                val_domains = rng.choice(np.asarray(candidates), size=n_val_domains, replace=False).tolist()
                val_set = set(int(v) for v in val_domains)
                train_set = set(int(v) for v in unique) - val_set

                # Guarantee that every class is present in training.  If a
                # missing class lives in a validation domain, move that domain
                # back to training (validation remains subject-disjoint).
                if labels_np is not None:
                    for c, doms in class_domains.items():
                        if not (doms & train_set):
                            moved = sorted(doms & val_set)
                            if moved:
                                dom = moved[0]
                                val_set.remove(dom)
                                train_set.add(dom)

                if val_set:
                    val_mask = np.isin(domains, list(val_set))
                    train_idx = np.where(~val_mask)[0]
                    val_idx = np.where(val_mask)[0]
                    info = {
                        "split": "subject_held_out",
                        "val_domains": sorted(int(v) for v in val_set),
                        "forced_train_domains": sorted(int(v) for v in forced_train),
                        "rare_classes": rare_classes,
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                    }
                    if labels_np is not None:
                        info["train_classes"] = sorted(int(c) for c in np.unique(labels_np[train_idx]))
                else:
                    train_idx, val_idx = _random_split(n, val_fraction, rng)
                    info = {"split": "random", "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
            else:
                train_idx, val_idx = _random_split(n, val_fraction, rng)
                info = {"split": "random", "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
        else:
            train_idx, val_idx = _random_split(n, val_fraction, rng)
            info = {"split": "random", "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
    else:
        train_idx, val_idx = _random_split(n, val_fraction, rng)
        info = {"split": "random", "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Validation split produced an empty train or validation set")

    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist()), info


def _random_split(n: int, val_fraction: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)
    return perm[n_val:], perm[:n_val]


def forward_model(
    model: nn.Module,
    x: torch.Tensor,
    delta_t: Optional[float] = None,
    sampling_rate: Optional[float] = None,
):
    """Call a backbone with an optional physical sampling interval/rate.

    Only the continuous-time model accepts ``delta_t``/``sampling_rate``;
    baseline 1D-CNNs keep their original single-argument interface.
    """
    kwargs = {}
    if delta_t is not None:
        kwargs["delta_t"] = delta_t
    if sampling_rate is not None:
        kwargs["sampling_rate"] = sampling_rate
    if not kwargs:
        return model(x)
    try:
        return model(x, **kwargs)
    except TypeError:
        return model(x)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    delta_t: Optional[float] = None,
    sampling_rate: Optional[float] = None,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Evaluate a model and return metrics plus mean loss (if criterion given)."""
    model.eval()
    all_preds = []
    all_targets = []
    losses = []

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            for batch in data_loader:
                x, y = batch[0], batch[1]
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits, _ = forward_model(
                    model, x, delta_t=delta_t, sampling_rate=sampling_rate
                )
                if criterion is not None:
                    losses.append(float(criterion(logits.float(), y).item()))
                all_preds.append(torch.argmax(logits, dim=-1).cpu())
                all_targets.append(y.cpu())

    if not all_preds:
        raise ValueError("Evaluation loader is empty")

    metrics = compute_metrics(torch.cat(all_targets), torch.cat(all_preds))
    if losses:
        metrics["loss"] = float(np.mean(losses))
    return metrics


def selection_score(selection_metric: str, metrics: Dict[str, Any]) -> float:
    """Return a scalar where larger is always better."""
    if selection_metric == "source_loss":
        return -float(metrics.get("loss", float("inf")))
    if selection_metric == "source_macro_f1":
        return float(metrics.get("macro_f1", float("-inf")))
    if selection_metric == "source_accuracy":
        return float(metrics.get("accuracy", float("-inf")))
    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def is_improvement(selection_metric: str, metrics: Dict[str, Any], best: float) -> bool:
    return selection_score(selection_metric, metrics) > best
