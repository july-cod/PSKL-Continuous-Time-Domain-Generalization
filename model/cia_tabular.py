# -*- coding: utf-8 -*-
"""
Tabular adaptation of CIA (Causal Invariance-aware Augmentation) ideas for vector inputs.

Brain-graph CIA operates on edges; here each feature dimension is scored (analogous to
edge attention), split into causal / spurious masks, and mixed across samples like the
cross-subgraph mixing in the original paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _concrete_bernoulli(logits: torch.Tensor, temp: float, training: bool) -> torch.Tensor:
    """Differentiable relaxation similar to CIA concrete_sample on sigmoid logits."""
    eps = 1e-9
    p = torch.sigmoid(logits)
    if not training:
        return p
    u1 = torch.empty_like(p).uniform_(eps, 1.0 - eps)
    u2 = torch.empty_like(p).uniform_(eps, 1.0 - eps)
    g1 = -torch.log(-torch.log(u1))
    g2 = -torch.log(-torch.log(u2))
    z1 = (torch.log(p + eps) + g1) / temp
    z2 = (torch.log(1.0 - p + eps) + g2) / temp
    return torch.exp(z1) / (torch.exp(z1) + torch.exp(z2))


class CIAFeatureAugment(nn.Module):
    """
    Per-dimension scores -> causal mask (top-k by relaxed weights) -> causal / spurious / mix views.

    task_fusion: 将因果子空间以可学习强度注入主路径（供 Koodos 的 predictive / generalized 使用），
    避免 CIA 仅通过 aux loss 间接作用、主表示仍为原始 x。
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        reserve_ratio: float = 0.35,
        min_keep_dims: int = 1,
        concrete_temp: float = 1.0,
        task_fusion: bool = False,
        fusion_init_logit: float = -2.0,
        input_layernorm: bool = False,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.reserve_ratio = reserve_ratio
        self.min_keep_dims = int(min_keep_dims)
        self.concrete_temp = concrete_temp
        self.task_fusion = task_fusion
        # Avoid inplace ReLU here: this MLP feeds masks that split into multiple forward branches;
        # inplace ops can corrupt shared autograd state and produce NaNs after backward.
        self.score_ln = nn.LayerNorm(data_dim) if input_layernorm else None
        self.score_mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, data_dim),
        )
        if task_fusion:
            self.fusion_logit = nn.Parameter(
                torch.tensor(float(fusion_init_logit), dtype=torch.float32)
            )
        else:
            self.register_parameter("fusion_logit", None)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [N, D]
        Returns dict with keys: full, causal, spurious, mix, mask (hard mask for interpretability).
        ``full`` 在 task_fusion 开启时为融合后的主任务表示，否则为原始 x。
        """
        n, d = x.shape
        score_in = self.score_ln(x) if self.score_ln is not None else x
        weights = _concrete_bernoulli(
            self.score_mlp(score_in), self.concrete_temp, self.training
        )

        k = max(1, int(self.reserve_ratio * d))
        # For very low-dimensional inputs (e.g., Moons D=2), top-k with ratio may collapse to 1.
        # min_keep_dims lets us keep at least a configured number of invariant dimensions.
        k = max(k, self.min_keep_dims)
        k = min(k, d)
        # Hard top-k causal dimensions per sample
        top_idx = torch.topk(weights, k, dim=-1).indices
        mask_c = torch.zeros_like(weights)
        mask_c.scatter_(1, top_idx, 1.0)
        # Straight-through: forward hard mask, backward through soft weights
        mask_ste = mask_c + (weights - weights.detach())

        x_c = x * mask_ste
        x_s = x * (1.0 - mask_ste)

        if n > 1:
            perm = torch.randperm(n, device=x.device)
            x_mix = x_c + x[perm] * (1.0 - mask_ste)
        else:
            x_mix = x

        if self.task_fusion and self.fusion_logit is not None:
            g = torch.sigmoid(self.fusion_logit)
            x_full = x + g * (x_c - x)
        else:
            x_full = x

        return {
            "full": x_full,
            "causal": x_c,
            "spurious": x_s,
            "mix": x_mix,
            "mask": mask_c,
            "soft_weights": weights,
        }


def entropy_from_log_softmax(log_probs: torch.Tensor) -> torch.Tensor:
    """Mean entropy H(p) given log p, shape [N, C]."""
    p = torch.exp(log_probs)
    ent = -(p * log_probs).sum(dim=-1)
    return ent.mean()


def binary_entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Mean binary entropy for sigmoid outputs, shape [N, 1] or [N]."""
    # float32: (1.0 - 1e-8) rounds to 1.0, so log(1-p) becomes -inf and 0*inf -> NaN.
    eps = 1e-6
    p = probs.view(-1).clamp(eps, 1.0 - eps)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return ent.mean()
