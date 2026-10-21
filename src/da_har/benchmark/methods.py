"""Adaptation objectives used by the benchmark baselines.

Only three classical objectives are shipped here (CORAL, MMD, DANN) plus
``source_only``.  They exist to answer a single question: *does a new method
beat ERM and the standard alignment baselines on the same task, with a
source-selected checkpoint?*  No claim is made that any of them is novel.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ADAPTATION_METHODS", "build_method"]


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


def _cov(features: torch.Tensor) -> torch.Tensor:
    n = features.size(0)
    if n < 2:
        return torch.zeros((features.size(1), features.size(1)), device=features.device)
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.t().matmul(centered) / (n - 1)


class AdaptationMethod(nn.Module):
    name = "base"

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def adapt_loss(
        self,
        src_feat: torch.Tensor,
        tgt_feat: torch.Tensor,
        src_logits: torch.Tensor,
        tgt_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def describe(self) -> Dict[str, object]:
        return {"name": self.name}


class SourceOnly(AdaptationMethod):
    name = "source_only"

    def adapt_loss(self, src_feat, tgt_feat, src_logits, tgt_logits):
        return src_feat.sum() * 0.0, {"adapt": 0.0}


class CORAL(AdaptationMethod):
    name = "coral"

    def adapt_loss(self, src_feat, tgt_feat, src_logits, tgt_logits):
        cs = _cov(src_feat)
        ct = _cov(tgt_feat)
        d = cs.size(0)
        loss = (cs - ct).pow(2).sum() / (4.0 * d * d)
        return loss, {"adapt": float(loss.detach())}


class MMD(AdaptationMethod):
    name = "mmd"

    def __init__(self, feat_dim: int, num_classes: int, bandwidths=(1.0, 2.0, 4.0, 8.0)) -> None:
        super().__init__(feat_dim, num_classes)
        self.bandwidths = tuple(bandwidths)

    @staticmethod
    def _rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        x2 = x.pow(2).sum(dim=1, keepdim=True)
        y2 = y.pow(2).sum(dim=1, keepdim=True)
        d2 = x2 + y2.t() - 2.0 * x.matmul(y.t())
        return torch.exp(-d2 / (2.0 * sigma * sigma + 1e-8))

    def adapt_loss(self, src_feat, tgt_feat, src_logits, tgt_logits):
        loss = src_feat.new_zeros(())
        for sigma in self.bandwidths:
            k_ss = self._rbf(src_feat, src_feat, sigma).mean()
            k_tt = self._rbf(tgt_feat, tgt_feat, sigma).mean()
            k_st = self._rbf(src_feat, tgt_feat, sigma).mean()
            loss = loss + (k_ss + k_tt - 2.0 * k_st)
        loss = loss / len(self.bandwidths)
        return loss, {"adapt": float(loss.detach())}


class DANN(AdaptationMethod):
    name = "dann"

    def __init__(self, feat_dim: int, num_classes: int, lambd: float = 1.0) -> None:
        super().__init__(feat_dim, num_classes)
        self.lambd = lambd
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, max(feat_dim // 2, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(feat_dim // 2, 16), 2),
        )

    def adapt_loss(self, src_feat, tgt_feat, src_logits, tgt_logits):
        feats = torch.cat([src_feat, tgt_feat], dim=0)
        reversed_feats = _GradientReversal.apply(feats, self.lambd)
        logits = self.domain_head(reversed_feats)
        labels = torch.cat(
            [
                torch.zeros(src_feat.size(0), dtype=torch.long, device=src_feat.device),
                torch.ones(tgt_feat.size(0), dtype=torch.long, device=src_feat.device),
            ]
        )
        loss = F.cross_entropy(logits, labels)
        acc = float((logits.argmax(dim=1) == labels).float().mean().detach())
        return loss, {"adapt": float(loss.detach()), "domain_acc": acc}


ADAPTATION_METHODS = {
    "source_only": SourceOnly,
    "coral": CORAL,
    "mmd": MMD,
    "dann": DANN,
}


def build_method(
    name: str, feat_dim: int, num_classes: int, da_weight: float = 1.0
) -> AdaptationMethod:
    key = name.lower()
    if key not in ADAPTATION_METHODS:
        raise ValueError(f"Unknown method {name!r}. Available: {tuple(ADAPTATION_METHODS)}")
    method = ADAPTATION_METHODS[key](feat_dim=feat_dim, num_classes=num_classes)
    method.da_weight = float(da_weight)  # type: ignore[attr-defined]
    return method
