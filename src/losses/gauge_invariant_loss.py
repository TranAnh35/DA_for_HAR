from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaugeInvariantUDALoss(nn.Module):
    """
    Gauge-Invariant Unsupervised Domain Adaptation Loss.
    Combines:
      1. Source supervised cross-entropy loss.
      2. Gauge-invariant manifold discrepancy (Gram matrix distance).
      3. Target predictive entropy minimization for boundary refinement.
    """
    def __init__(self, lambda_gauge: float = 0.5, lambda_ent: float = 0.1):
        super().__init__()
        self.lambda_gauge = lambda_gauge
        self.lambda_ent = lambda_ent
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits_s: torch.Tensor,
        y_s: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
        logits_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        logits_s: [B_s, num_classes]
        y_s: [B_s]
        z_s: [B_s, d_feature]
        z_t: [B_t, d_feature]
        logits_t: [B_t, num_classes]
        """
        # 1. Supervised loss on labeled source domain
        loss_src = self.ce_loss(logits_s, y_s)

        if z_t is None or logits_t is None or (self.lambda_gauge == 0.0 and self.lambda_ent == 0.0):
            loss_gauge = torch.tensor(0.0, device=logits_s.device)
            loss_ent = torch.tensor(0.0, device=logits_s.device)
            total_loss = loss_src
        else:
            # 2. Gauge-invariant Gram alignment
            z_s_norm = F.normalize(z_s, p=2, dim=-1)
            z_t_norm = F.normalize(z_t, p=2, dim=-1)

            # Inner product Gram matrices [d_feature, d_feature]
            gram_s = torch.matmul(z_s_norm.T, z_s_norm) / float(z_s.size(0))
            gram_t = torch.matmul(z_t_norm.T, z_t_norm) / float(z_t.size(0))
            loss_gauge = (gram_s - gram_t).pow(2).sum()

            # 3. Target conditional entropy minimization
            probs_t = F.softmax(logits_t, dim=-1)
            loss_ent = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=-1))

            # Total objective
            total_loss = loss_src + self.lambda_gauge * loss_gauge + self.lambda_ent * loss_ent

        metrics = {
            "loss_total": total_loss.item(),
            "loss_src": loss_src.item(),
            "loss_gauge": loss_gauge.item(),
            "loss_ent": loss_ent.item()
        }

        return total_loss, metrics
