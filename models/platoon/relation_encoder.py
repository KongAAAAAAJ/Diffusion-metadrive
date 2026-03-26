from __future__ import annotations

import torch
from torch import Tensor, nn


class RelationEncoder(nn.Module):
    """Encode 12D platoon relation state into a normalized 12D embedding."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, output_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.net(x))
