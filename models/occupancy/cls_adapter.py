"""Residual adapter that fuses occupancy matching features into cls_feature.

Gate initialisation near zero ensures the adapter starts as a near-identity
transform, so training is stable even if the occupancy predictor is not yet
well-calibrated.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OccupancyAdapter(nn.Module):
    """Fuse per-mode occupancy matching features into cls_feature via a gated residual.

    Parameters
    ----------
    cls_dim : int
        Dimension of the incoming cls_feature (default 160 = 128+32).
    match_dim : int
        Dimension of occupancy matching features per mode (default 6).
    hidden_dim : int
        Internal projection dimension.
    """

    def __init__(
        self,
        cls_dim: int = 160,
        match_dim: int = 6,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cls_dim = int(cls_dim)

        # Gate: small init so sigmoid ≈ 0.5 at start; output ∈ (0,1)
        self.gate = nn.Sequential(
            nn.Linear(match_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cls_dim),
            nn.Sigmoid(),
        )

        # Delta projection
        self.delta = nn.Sequential(
            nn.Linear(cls_dim + match_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cls_dim),
        )

        # Zero-init gate and delta output so adapter starts as identity
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.constant_(self.gate[-2].bias, -4.0)   # sigmoid(-4) ≈ 0.02
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        cls_feature: torch.Tensor,        # [N, M, cls_dim]
        match_features: torch.Tensor,     # [N, M, match_dim]
    ) -> torch.Tensor:
        """Return augmented cls_feature [N, M, cls_dim]."""
        gate = self.gate(match_features)                            # [N, M, cls_dim]
        delta = self.delta(torch.cat([cls_feature, match_features], dim=-1))  # [N, M, cls_dim]
        return cls_feature + gate * delta
