import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseLayerF(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from Ganin et al. (ICML 2016).
    Multiplies gradient by -alpha during backward pass.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    """
    Standard MLP Domain Discriminator for DANN / CDAN.
    Distinguishes whether feature vectors belong to Source (0) or Target (1).
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CORALLoss(nn.Module):
    """
    Deep CORAL (Correlation Alignment) Loss from Sun & Saenko (ECCV 2016).
    Minimizes Frobenius norm distance between feature covariance matrices.
    """
    def __init__(self, lambda_coral: float = 0.5, lambda_ent: float = 0.05):
        super().__init__()
        self.lambda_coral = lambda_coral
        self.lambda_ent = lambda_ent
        self.ce_loss = nn.CrossEntropyLoss()

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        mean = torch.mean(x, dim=0, keepdim=True)
        xm = x - mean
        cov = torch.matmul(xm.t(), xm) / (n - 1 + 1e-8)
        return cov

    def forward(
        self,
        logits_s: torch.Tensor,
        y_s: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
        logits_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_src = self.ce_loss(logits_s, y_s)

        if z_t is None or self.lambda_coral == 0.0:
            return loss_src, {"loss_total": loss_src.item(), "loss_src": loss_src.item(), "loss_da": 0.0}

        d = z_s.size(1)
        cov_s = self._covariance(z_s)
        cov_t = self._covariance(z_t)

        loss_coral = torch.sum((cov_s - cov_t) ** 2) / (4.0 * d * d)

        loss_ent = torch.tensor(0.0, device=logits_s.device)
        if logits_t is not None and self.lambda_ent > 0.0:
            probs_t = F.softmax(logits_t, dim=-1)
            loss_ent = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=-1))

        total_loss = loss_src + self.lambda_coral * loss_coral + self.lambda_ent * loss_ent

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_src": loss_src.item(),
            "loss_da": loss_coral.item(),
            "loss_ent": loss_ent.item()
        }

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) with Multi-Kernel Gaussian RBF.
    From Long et al. (DAN, ICML 2015 / Gretton et al., JMLR 2012).
    """
    def __init__(self, lambda_mmd: float = 0.5, lambda_ent: float = 0.05, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        self.lambda_ent = lambda_ent
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.ce_loss = nn.CrossEntropyLoss()

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples + 1e-8)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-l2_distance / (bw + 1e-8)) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(
        self,
        logits_s: torch.Tensor,
        y_s: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
        logits_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_src = self.ce_loss(logits_s, y_s)

        if z_t is None or self.lambda_mmd == 0.0:
            return loss_src, {"loss_total": loss_src.item(), "loss_src": loss_src.item(), "loss_da": 0.0}

        batch_size = int(z_s.size(0))
        kernels = self._gaussian_kernel(z_s, z_t)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss_mmd = torch.mean(XX + YY - XY - YX)

        loss_ent = torch.tensor(0.0, device=logits_s.device)
        if logits_t is not None and self.lambda_ent > 0.0:
            probs_t = F.softmax(logits_t, dim=-1)
            loss_ent = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=-1))

        total_loss = loss_src + self.lambda_mmd * loss_mmd + self.lambda_ent * loss_ent

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_src": loss_src.item(),
            "loss_da": loss_mmd.item(),
            "loss_ent": loss_ent.item()
        }

class DANNLoss(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) Loss from Ganin et al. (ICML 2016).
    Uses Gradient Reversal Layer (GRL) + Domain Discriminator.
    """
    def __init__(self, discriminator: DomainDiscriminator, lambda_dann: float = 0.5, lambda_ent: float = 0.05):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_dann = lambda_dann
        self.lambda_ent = lambda_ent
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits_s: torch.Tensor,
        y_s: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
        logits_t: Optional[torch.Tensor] = None,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_src = self.ce_loss(logits_s, y_s)

        if z_t is None or self.lambda_dann == 0.0:
            return loss_src, {"loss_total": loss_src.item(), "loss_src": loss_src.item(), "loss_da": 0.0}

        # Apply GRL
        z_s_rev = ReverseLayerF.apply(z_s, alpha)
        z_t_rev = ReverseLayerF.apply(z_t, alpha)

        d_s = self.discriminator(z_s_rev)
        d_t = self.discriminator(z_t_rev)

        label_s = torch.zeros_like(d_s) # Source = 0
        label_t = torch.ones_like(d_t)  # Target = 1

        loss_d = 0.5 * (self.bce_loss(d_s, label_s) + self.bce_loss(d_t, label_t))

        loss_ent = torch.tensor(0.0, device=logits_s.device)
        if logits_t is not None and self.lambda_ent > 0.0:
            probs_t = F.softmax(logits_t, dim=-1)
            loss_ent = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=-1))

        total_loss = loss_src + self.lambda_dann * loss_d + self.lambda_ent * loss_ent

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_src": loss_src.item(),
            "loss_da": loss_d.item(),
            "loss_ent": loss_ent.item()
        }

class CDANLoss(nn.Module):
    """
    Conditional Domain Adversarial Network (CDAN) Loss from Long et al. (NeurIPS 2018).
    Conditions domain discrimination on multilinear feature-classifier predictions: h = f (x) g.
    Applies entropy weighting: w(H) = 1 + exp(-H(g)).
    """
    def __init__(self, discriminator: DomainDiscriminator, lambda_cdan: float = 0.5, lambda_ent: float = 0.05):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_cdan = lambda_cdan
        self.lambda_ent = lambda_ent
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        logits_s: torch.Tensor,
        y_s: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
        logits_t: Optional[torch.Tensor] = None,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_src = self.ce_loss(logits_s, y_s)

        if z_t is None or logits_t is None or self.lambda_cdan == 0.0:
            return loss_src, {"loss_total": loss_src.item(), "loss_src": loss_src.item(), "loss_da": 0.0}

        probs_s = F.softmax(logits_s, dim=-1)
        probs_t = F.softmax(logits_t, dim=-1)

        # Multilinear map: outer product between normalized feature and predicted class probability
        # h = f \otimes g: [B, d_feature, num_classes] -> flattened [B, d_feature * num_classes]
        h_s = torch.bmm(z_s.unsqueeze(2), probs_s.unsqueeze(1)).view(z_s.size(0), -1)
        h_t = torch.bmm(z_t.unsqueeze(2), probs_t.unsqueeze(1)).view(z_t.size(0), -1)

        # Entropy conditioning weights
        ent_s = -torch.sum(probs_s * torch.log(probs_s + 1e-8), dim=-1)
        ent_t = -torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=-1)
        w_s = (1.0 + torch.exp(-ent_s)).unsqueeze(1).detach()
        w_t = (1.0 + torch.exp(-ent_t)).unsqueeze(1).detach()

        # Reverse layer
        h_s_rev = ReverseLayerF.apply(h_s, alpha)
        h_t_rev = ReverseLayerF.apply(h_t, alpha)

        d_s = self.discriminator(h_s_rev)
        d_t = self.discriminator(h_t_rev)

        label_s = torch.zeros_like(d_s)
        label_t = torch.ones_like(d_t)

        loss_d_s = torch.mean(w_s * self.bce_loss(d_s, label_s))
        loss_d_t = torch.mean(w_t * self.bce_loss(d_t, label_t))
        loss_cdan = 0.5 * (loss_d_s + loss_d_t)

        loss_ent = -torch.mean(torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=-1))
        total_loss = loss_src + self.lambda_cdan * loss_cdan + self.lambda_ent * loss_ent

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_src": loss_src.item(),
            "loss_da": loss_cdan.item(),
            "loss_ent": loss_ent.item()
        }
