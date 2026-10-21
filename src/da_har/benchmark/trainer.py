"""Source-selected training loop.

The loop is deliberately boring.  What matters is the *selection rule*:

1. Fit the normaliser on **source train** only.
2. Split source into train/val with whole subjects held out when possible.
3. Train with source labels + unlabeled target inputs.
4. Select the checkpoint by **source validation macro-F1**.
5. Only then read target labels, once, to report the final metric.

Step 4 is the whole point.  The legacy pipeline selected on
``best_target_f1``, which is target-oracle and made every reported number
unusable.  The audit block written into each result makes that mistake
detectable after the fact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data.core import (
    DomainData,
    align_channels,
    fit_normalizer,
    macro_f1,
    pad_to_slots,
)
from .domains import TaskData
from .methods import build_method
from .models import build_backbone
from .protocol import audit_task, subject_disjoint_split

__all__ = ["TrainConfig", "train_task", "evaluate"]


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 0
    device: str = "cpu"
    da_weight: float = 1.0
    backbone: str = "cnn1d"
    hidden: int = 64
    dropout: float = 0.2
    deterministic: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


def _set_seed(seed: int, deterministic: bool = True) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_tensor(X: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)


def _mask_tensor(mask: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask, dtype=bool)).to(device)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int = 256,
) -> Dict[str, Any]:
    model.eval()
    preds: List[np.ndarray] = []
    for start in range(0, X.size(0), batch_size):
        xb = X[start : start + batch_size]
        logits, _ = model(xb, mask)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
    truth = y.cpu().numpy()
    acc = float((pred == truth).mean()) if len(truth) else float("nan")
    num_classes = int(max(truth.max(initial=0), pred.max(initial=0))) + 1 if len(truth) else 0
    f1 = macro_f1(truth, pred, num_classes=num_classes) if len(truth) else float("nan")

    per_class: Dict[str, float] = {}
    for cls in np.unique(truth):
        sel = truth == cls
        per_class[str(int(cls))] = float((pred[sel] == cls).mean())
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "per_class_recall": per_class,
        "num_windows": int(len(truth)),
    }


def _iterate_batches(
    n: int, batch_size: int, rng: np.random.Generator
) -> List[np.ndarray]:
    order = rng.permutation(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def train_task(
    task: TaskData,
    method_name: str = "source_only",
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    config = config or TrainConfig()
    _set_seed(config.seed, config.deterministic)
    device = config.device

    src: DomainData = task.source
    tgt: DomainData = task.target
    if src.window_size != tgt.window_size:
        raise ValueError(
            f"{task.name}: source/target window length mismatch "
            f"({src.window_size} vs {tgt.window_size})"
        )

    slots, masks = align_channels([src, tgt])
    src_X, src_mask = pad_to_slots(src, slots)
    tgt_X, tgt_mask = pad_to_slots(tgt, slots)

    split = subject_disjoint_split(src, val_fraction=config.val_fraction, seed=config.seed)
    normalizer = fit_normalizer(src_X[split.train])
    src_X = normalizer.transform(src_X)
    tgt_X = normalizer.transform(tgt_X)

    src_train_X = _to_tensor(src_X[split.train], device)
    src_train_y = torch.from_numpy(src.y[split.train]).long().to(device)
    src_val_X = _to_tensor(src_X[split.val], device)
    src_val_y = torch.from_numpy(src.y[split.val]).long().to(device)
    tgt_X_t = _to_tensor(tgt_X, device)
    tgt_y_t = torch.from_numpy(tgt.y).long().to(device)
    src_mask_t = _mask_tensor(src_mask, device)
    tgt_mask_t = _mask_tensor(tgt_mask, device)

    model = build_backbone(
        config.backbone,
        in_slots=len(slots),
        num_classes=task.num_classes,
        hidden=config.hidden,
        dropout=config.dropout,
    ).to(device)
    method = build_method(
        method_name, feat_dim=model.feat_dim, num_classes=task.num_classes
    ).to(device)
    method.da_weight = config.da_weight  # type: ignore[attr-defined]

    params = list(model.parameters()) + list(method.parameters())
    optim = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    n_train = src_train_X.size(0)
    if n_train < 2:
        raise ValueError(f"{task.name}: need at least 2 source-training windows")
    batch_size = max(1, min(config.batch_size, n_train))
    drop_last = batch_size < n_train

    rng = np.random.default_rng(config.seed + 7919)
    history: List[Dict[str, Any]] = []
    best = {"score": -np.inf, "epoch": -1, "state": None}

    for epoch in range(config.epochs):
        model.train()
        method.train()
        total, n_batches = 0.0, 0
        src_batches = _iterate_batches(n_train, batch_size, rng)
        tgt_batches = _iterate_batches(tgt_X_t.size(0), batch_size, rng)
        for i, s_idx in enumerate(src_batches):
            if drop_last and len(s_idx) < 2:
                continue
            s_x = src_train_X[s_idx]
            s_y = src_train_y[s_idx]
            t_idx = tgt_batches[i % len(tgt_batches)]
            if drop_last and len(t_idx) < 2:
                continue
            t_x = tgt_X_t[t_idx]

            s_logits, s_feat = model(s_x, src_mask_t)
            t_logits, t_feat = model(t_x, tgt_mask_t)
            ce = F.cross_entropy(s_logits, s_y)
            adapt, _ = method.adapt_loss(s_feat, t_feat, s_logits, t_logits)
            loss = ce + config.da_weight * adapt

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach())
            n_batches += 1

        val = evaluate(model, src_val_X, src_val_y, src_mask_t, batch_size=256)
        record = {
            "epoch": epoch,
            "train_loss": total / max(n_batches, 1),
            "source_val_accuracy": val["accuracy"],
            "source_val_macro_f1": val["macro_f1"],
        }
        history.append(record)

        score = val["macro_f1"]
        if score > best["score"]:
            best = {
                "score": float(score),
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    if best["state"] is None:
        raise RuntimeError(f"{task.name}: training never produced a checkpoint")
    model.load_state_dict(best["state"])

    src_train_eval = evaluate(model, src_train_X, src_train_y, src_mask_t)
    src_val_eval = evaluate(model, src_val_X, src_val_y, src_mask_t)
    tgt_eval = evaluate(model, tgt_X_t, tgt_y_t, tgt_mask_t)

    audit = audit_task(src, tgt, split, task.shift)
    return {
        "task": task.name,
        "dataset": task.spec.dataset,
        "shift": task.shift,
        "method": method_name,
        "backbone": config.backbone,
        "seed": int(config.seed),
        "is_target_oracle": False,
        "selection": {
            "metric": "source_val_macro_f1",
            "best_epoch": int(best["epoch"]),
            "best_score": float(best["score"]),
            "epochs": int(config.epochs),
        },
        "metrics": {
            "target_accuracy": tgt_eval["accuracy"],
            "target_macro_f1": tgt_eval["macro_f1"],
            "source_val_accuracy": src_val_eval["accuracy"],
            "source_val_macro_f1": src_val_eval["macro_f1"],
            "source_train_accuracy": src_train_eval["accuracy"],
            "source_train_macro_f1": src_train_eval["macro_f1"],
        },
        "per_class_recall": tgt_eval["per_class_recall"],
        "data": {
            **task.summary(),
            "source_train_windows": int(n_train),
            "source_val_windows": int(len(split.val)),
            "channel_slots": list(slots),
            "source_channel_mask": [bool(v) for v in src_mask],
            "target_channel_mask": [bool(v) for v in tgt_mask],
            "normalizer": normalizer.to_dict(),
        },
        "protocol": audit.to_dict(),
        "split": {
            "subject_disjoint": bool(split.subject_disjoint),
            "val_subjects": split.stats.get("val_subjects", []),
        },
        "config": asdict(config),
        "history": history,
        "num_parameters": int(model.num_parameters()),
        "label_names": list(task.label_names),
    }
