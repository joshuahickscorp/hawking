"""Tiny torch models for the Ramanujan formal system components."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BagEncoder(nn.Module):
    """Hashed bag-of-tokens -> low-dim embedding."""

    def __init__(self, bag_dim: int = 4096, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bag_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bag_dim) or (bag_dim,)
        return self.net(x)


class RetrieverModel(nn.Module):
    """Dual-encoder retriever: score(goal, premise_name) = cos(E(g), E(p))."""

    def __init__(self, bag_dim: int = 4096, emb_dim: int = 64):
        super().__init__()
        self.goal_enc = BagEncoder(bag_dim, emb_dim)
        self.prem_enc = BagEncoder(bag_dim, emb_dim)

    def encode_goal(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.goal_enc(x), dim=-1)

    def encode_prem(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.prem_enc(x), dim=-1)

    def score(self, goal_x: torch.Tensor, prem_x: torch.Tensor) -> torch.Tensor:
        g = self.encode_goal(goal_x)
        p = self.encode_prem(prem_x)
        if g.dim() == 1:
            return (g * p).sum(dim=-1)
        # goal (B,d), prem (B,C,d) or (B,d)
        if p.dim() == 2 and g.dim() == 2 and p.shape[0] == g.shape[0]:
            return (g * p).sum(dim=-1)
        return torch.matmul(g, p.transpose(-1, -2))


class ClassifierModel(nn.Module):
    """Bag -> class logits (formalizer first-tactic / prover next-tactic / repair fix)."""

    def __init__(self, bag_dim: int = 4096, n_classes: int = 100, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bag_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueModel(nn.Module):
    """State bag -> P(closed next) + predicted remaining steps."""

    def __init__(self, bag_dim: int = 4096, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(bag_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.closed_head = nn.Linear(hidden, 1)
        self.steps_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        closed_logit = self.closed_head(h).squeeze(-1)
        steps = F.softplus(self.steps_head(h).squeeze(-1))
        return closed_logit, steps
