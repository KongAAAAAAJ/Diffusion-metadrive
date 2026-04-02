from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def _hidden_tuple(hidden: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden, int):
        return (hidden, hidden)
    return tuple(int(v) for v in hidden)


def _build_mlp(input_dim: int, output_dim: int, hidden: int | Sequence[int]) -> nn.Sequential:
    dims = (int(input_dim),) + _hidden_tuple(hidden) + (int(output_dim),)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != output_dim:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class IntentSelectorActor(nn.Module):
    """Shared selector actor over structured planner observations."""

    def __init__(
        self,
        agent_context_dim: int,
        relation_dim: int,
        mode_embedding_dim: int,
        summary_dim: int,
        num_modes: int,
        hidden: int | Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.agent_context_dim = int(agent_context_dim)
        self.relation_dim = int(relation_dim)
        self.mode_embedding_dim = int(mode_embedding_dim)
        self.summary_dim = int(summary_dim)
        self.num_modes = int(num_modes)
        input_dim = (
            self.agent_context_dim
            + self.relation_dim
            + self.num_modes * (self.mode_embedding_dim + self.summary_dim)
        )
        self.net = _build_mlp(input_dim, self.num_modes, hidden)

    def forward(
        self,
        *,
        agent_context: Tensor,
        formation_relation_state: Tensor,
        mode_embeddings: Tensor,
        candidate_summary: Tensor,
    ) -> Tensor:
        if agent_context.ndim == 1:
            agent_context = agent_context.unsqueeze(0)
        if formation_relation_state.ndim == 1:
            formation_relation_state = formation_relation_state.unsqueeze(0)
        if mode_embeddings.ndim == 2:
            mode_embeddings = mode_embeddings.unsqueeze(0)
        if candidate_summary.ndim == 2:
            candidate_summary = candidate_summary.unsqueeze(0)

        flat_modes = mode_embeddings.reshape(mode_embeddings.shape[0], -1).float()
        flat_summary = candidate_summary.reshape(candidate_summary.shape[0], -1).float()
        actor_input = torch.cat(
            [
                agent_context.float(),
                formation_relation_state.float(),
                flat_modes,
                flat_summary,
            ],
            dim=-1,
        )
        return self.net(actor_input)


class IntentSelectorCritic(nn.Module):
    """Centralized critic over team-level global state."""

    def __init__(self, global_state_dim: int, hidden: int | Sequence[int] = (256, 256)):
        super().__init__()
        self.global_state_dim = int(global_state_dim)
        self.net = _build_mlp(self.global_state_dim, 1, hidden)

    def forward(self, global_state: Tensor) -> Tensor:
        if global_state.ndim == 1:
            global_state = global_state.unsqueeze(0)
        return self.net(global_state.float()).squeeze(-1)


CentralizedCritic = IntentSelectorCritic
IntentSelector = IntentSelectorActor
