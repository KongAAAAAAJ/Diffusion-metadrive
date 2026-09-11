from __future__ import annotations

import torch
from torch import Tensor

from .config import RiskPACTConfig


RELATION_NEIGHBORS = 2
RELATION_FEATURES_PER_NEIGHBOR = 6


def build_platoon_actor_state(
    ego_state: Tensor,
    formation_relation_state: Tensor,
    relation_valid_mask: Tensor,
    *,
    config: RiskPACTConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert formation-relation features to the shared 8-D actor contract.

    Source relation per neighbor:
      [x_rel, y_rel, d_heading_rad, delta_speed_km_h, desired_x, reserved]

    Output actor state:
      [x_rel, y_rel, sin(d_heading), cos(d_heading),
       dvx_local, dvy_local, length, width]

    ``ego_state[...,0]`` is ego speed in m/s.  Relative speed in the relation
    tensor is km/h in the current environment implementation.
    """

    cfg = config or RiskPACTConfig()
    if ego_state.ndim != 3 or ego_state.shape[-1] < 1:
        raise ValueError("ego_state must have shape [B,R,8+]" )
    if formation_relation_state.ndim != 3 or formation_relation_state.shape[-1] != 12:
        raise ValueError("formation_relation_state must have shape [B,R,12]")
    if relation_valid_mask.ndim != 3 or relation_valid_mask.shape[-1] != RELATION_NEIGHBORS:
        raise ValueError("relation_valid_mask must have shape [B,R,2]")
    if relation_valid_mask.dtype != torch.bool:
        raise ValueError("relation_valid_mask must be boolean")
    if tuple(ego_state.shape[:2]) != tuple(formation_relation_state.shape[:2]) or tuple(
        ego_state.shape[:2]
    ) != tuple(relation_valid_mask.shape[:2]):
        raise ValueError("ego/relation tensors must share B/R axes")

    relation = formation_relation_state.reshape(
        *formation_relation_state.shape[:2], RELATION_NEIGHBORS, RELATION_FEATURES_PER_NEIGHBOR
    )
    x_rel = relation[..., 0]
    y_rel = relation[..., 1]
    d_heading = relation[..., 2]
    delta_speed_mps = relation[..., 3] / 3.6

    ego_speed = ego_state[..., 0].unsqueeze(-1)
    neighbor_speed = (ego_speed + delta_speed_mps).clamp_min(0.0)
    sin_h = torch.sin(d_heading)
    cos_h = torch.cos(d_heading)
    dvx = neighbor_speed * cos_h - ego_speed
    dvy = neighbor_speed * sin_h

    length = torch.full_like(x_rel, float(cfg.platoon_vehicle_length_m))
    width = torch.full_like(y_rel, float(cfg.platoon_vehicle_width_m))
    state = torch.stack(
        (x_rel, y_rel, sin_h, cos_h, dvx, dvy, length, width), dim=-1
    )
    state = torch.where(
        relation_valid_mask.unsqueeze(-1), state, torch.zeros_like(state)
    )
    return state.to(dtype=torch.float32), relation_valid_mask
