from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaugeInvariantUDALoss(nn.Module):
    """Unsupervised domain-adaptation objective for the proposed model.

    Combines:
      1. Supervised cross-entropy on the labeled source domain.
      2. **Class-conditional prototype alignment** between source class
         prototypes and confident target pseudo-class prototypes.  This
         replaces the previous Gram-matrix penalty, which was only a
         reformulation of CORAL and ignored class structure.
      3. Target predictive entropy minimization.

    Target labels are never used: the target side is built from
    ``argmax`` pseudo-labels and a confidence threshold.
    """

    def __init__(
        self,
        lambda_gauge: float = 0.5,
        lambda_ent: float = 0.1,
        pseudo_threshold: float = 0.0,
    ):
        super().__init__()
        # ``lambda_gauge`` is kept as the public name for backward
        # compatibility; it now weights prototype alignment.
        self.lambda_proto = float(lambda_gauge)
        self.lambda_ent = float(lambda_ent)
        self.pseudo_threshold = float(pseudo_threshold)
        self.ce_loss = nn.CrossEntropyLoss()

    def _prototype_alignment(
        self,
        z_s: torch.Tensor,
        y_s: torch.Tensor,
        z_t: torch.Tensor,
        logits_t: torch.Tensor,
        num_classes: int,
    ) -> Tuple[torch.Tensor, int, float, int]:
        z_s_norm = F.normalize(z_s, p=2, dim=-1)
        z_t_norm = F.normalize(z_t, p=2, dim=-1)

        # Source class prototypes (classes present in the batch).
        onehot_s = F.one_hot(y_s, num_classes=num_classes).to(z_s.dtype)
        count_s = onehot_s.sum(dim=0)
        proto_s = torch.matmul(onehot_s.t(), z_s_norm) / count_s.clamp(min=1.0).unsqueeze(1)
        proto_s = F.normalize(proto_s, p=2, dim=-1)

        # Confident target pseudo-class prototypes.
        probs_t = F.softmax(logits_t, dim=-1)
        confidence, pseudo = probs_t.max(dim=-1)
        mask = confidence >= self.pseudo_threshold
        onehot_t = F.one_hot(pseudo, num_classes=num_classes).to(z_t.dtype)
        onehot_t = onehot_t * mask.unsqueeze(1).to(z_t.dtype)
        count_t = onehot_t.sum(dim=0)
        proto_t = torch.matmul(onehot_t.t(), z_t_norm) / count_t.clamp(min=1.0).unsqueeze(1)
        proto_t = F.normalize(proto_t, p=2, dim=-1)

        valid = (count_s > 0) & (count_t > 0)
        if valid.any():
            similarity = (proto_s * proto_t).sum(dim=-1)
            loss_proto = (1.0 - similarity)[valid].mean()
            fallback = 0
        else:
            # Early in training (or in a small batch) the source labels and
            # target pseudo-labels may have no class in common.  Fall back to
            # a global mean-prototype alignment so the target branch still
            # receives a gradient; the primary objective remains conditional.
            global_s = F.normalize(z_s.mean(dim=0, keepdim=True), p=2, dim=-1)
            global_t = F.normalize(z_t.mean(dim=0, keepdim=True), p=2, dim=-1)
            loss_proto = (1.0 - (global_s * global_t).sum()).squeeze()
            fallback = 1

        coverage = float(mask.float().mean().item()) if mask.numel() > 0 else 0.0
        return loss_proto, int(valid.sum().item()), coverage, fallback

    def forward(
        self,
        logits_s: torch.Tensor,
        y_s: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
        logits_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """logits_s [B_s,C], y_s [B_s], z_s [B_s,D], z_t [B_t,D], logits_t [B_t,C]."""
        loss_src = self.ce_loss(logits_s, y_s)
        num_classes = logits_s.size(1)

        if z_t is None or logits_t is None or (self.lambda_proto == 0.0 and self.lambda_ent == 0.0):
            zero = torch.zeros((), device=logits_s.device, dtype=logits_s.dtype)
            loss_proto = zero
            loss_ent = zero
            valid_classes = 0
            coverage = 0.0
            fallback = 0
        else:
            loss_proto, valid_classes, coverage, fallback = self._prototype_alignment(
                z_s, y_s, z_t, logits_t, num_classes
            )
            probs_t = F.softmax(logits_t, dim=-1)
            loss_ent = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=-1))

        total_loss = loss_src + self.lambda_proto * loss_proto + self.lambda_ent * loss_ent

        metrics = {
            "loss_total": total_loss.item(),
            "loss_src": loss_src.item(),
            "loss_da": loss_proto.item(),
            "loss_proto": loss_proto.item(),
            # Legacy key kept for existing loggers/checkpoints.
            "loss_gauge": loss_proto.item(),
            "loss_ent": loss_ent.item(),
            "proto_classes": float(valid_classes),
            "pseudo_coverage": coverage,
            "proto_fallback": float(fallback),
        }
        return total_loss, metrics
