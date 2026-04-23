from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def _hidden_tuple(hidden: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden, int):
        return (hidden, hidden)
    return tuple(int(v) for v in hidden)


def _build_mlp(input_dim: int, hidden: int | Sequence[int]) -> nn.Sequential:
    dims = (int(input_dim),) + _hidden_tuple(hidden)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
    return nn.Sequential(*layers)


class CoPreferenceModel(nn.Module):
    """Shared co-preference policy over relation-aware single-car status."""

    def __init__(
        self,
        status_dim: int = 8,
        relation_dim: int = 12,
        hidden: int | Sequence[int] = (256, 256),
        num_topologies: int = 4,
    ) -> None:
        super().__init__()
        self.status_dim = int(status_dim)
        self.relation_dim = int(relation_dim)
        self.num_topologies = int(num_topologies)
        trunk_hidden = _hidden_tuple(hidden)
        self.trunk = _build_mlp(self.status_dim + self.relation_dim, trunk_hidden)
        final_dim = trunk_hidden[-1]
        self.topology_head = nn.Linear(final_dim, self.num_topologies)
        self.s_head = nn.Linear(final_dim, 1)

    def forward(
        self,
        *,
        status_feature: Tensor,
        formation_relation_state: Tensor,
        topology_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if formation_relation_state.ndim == 1:
            formation_relation_state = formation_relation_state.unsqueeze(0)
        x = torch.cat([status_feature.float(), formation_relation_state.float()], dim=-1)
        hidden = self.trunk(x)
        topology_logits = self.topology_head(hidden)
        if topology_mask is not None:
            mask = topology_mask.to(device=topology_logits.device, dtype=torch.bool)
            topology_logits = topology_logits.masked_fill(~mask, -1e9)
        s_raw = self.s_head(hidden).squeeze(-1)
        return {
            "topology_logits": topology_logits,
            "s_raw": s_raw,
            "s": torch.sigmoid(s_raw),
        }
