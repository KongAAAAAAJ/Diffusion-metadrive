from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import RiskPACTConfig
from .road_field import RoadBoundaryRiskField


@dataclass(frozen=True)
class RiskFieldResult:
    """Risk queried along candidate trajectories.

    ``risk`` is the final bounded multi-source union.  ``per_actor_risk`` and
    ``actor_future_xy`` concatenate enabled background and platoon actors for
    backward-compatible diagnostics.  Component tensors expose the source of
    each safety signal.
    """

    risk: Tensor
    per_actor_risk: Tensor
    actor_future_xy: Tensor
    background_risk: Tensor | None = None
    platoon_risk: Tensor | None = None
    road_risk: Tensor | None = None
    road_signed_distance_m: Tensor | None = None


class DynamicGaussianRiskField:
    """Analytic differentiable spatio-temporal actor risk field."""

    def __init__(self, config: RiskPACTConfig | None = None) -> None:
        self.config = config or RiskPACTConfig()

    def actor_future_xy(self, actor_state: Tensor, horizon_steps: int) -> Tensor:
        if actor_state.ndim != 4 or actor_state.shape[-1] != 8:
            raise ValueError("actor_state must have shape [B,R,A,8]")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        dtype, device = actor_state.dtype, actor_state.device
        times = (
            torch.arange(1, horizon_steps + 1, dtype=dtype, device=device)
            * float(self.config.horizon_dt_s)
        )
        xy0 = actor_state[..., 0:2]
        velocity = actor_state[..., 4:6]
        return xy0.unsqueeze(-2) + velocity.unsqueeze(-2) * times.view(1, 1, 1, -1, 1)

    def query(
        self,
        trajectory_xy: Tensor,
        actor_state: Tensor,
        actor_valid_mask: Tensor,
    ) -> RiskFieldResult:
        squeeze_mode = False
        if trajectory_xy.ndim == 4:
            trajectory_xy = trajectory_xy.unsqueeze(2)
            squeeze_mode = True
        if trajectory_xy.ndim != 5 or trajectory_xy.shape[-1] != 2:
            raise ValueError("trajectory_xy must have shape [B,R,M,H,2] or [B,R,H,2]")
        if actor_state.ndim != 4 or actor_state.shape[-1] != 8:
            raise ValueError("actor_state must have shape [B,R,A,8]")
        if actor_valid_mask.shape != actor_state.shape[:-1]:
            raise ValueError("actor_valid_mask must have shape [B,R,A]")
        if actor_valid_mask.dtype != torch.bool:
            raise ValueError("actor_valid_mask must be boolean")
        if trajectory_xy.shape[:2] != actor_state.shape[:2]:
            raise ValueError("trajectory and actor state B/R axes must match")

        _, _, _, h, _ = trajectory_xy.shape
        future = self.actor_future_xy(actor_state, h)  # [B,R,A,H,2]
        future_hr = future.permute(0, 1, 3, 2, 4)  # [B,R,H,A,2]
        delta = trajectory_xy.unsqueeze(-2) - future_hr.unsqueeze(2)

        sin_yaw = actor_state[..., 2]
        cos_yaw = actor_state[..., 3]
        norm = torch.sqrt(cos_yaw.square() + sin_yaw.square()).clamp_min(1.0e-6)
        cos_yaw = (cos_yaw / norm)[:, :, None, None, :]
        sin_yaw = (sin_yaw / norm)[:, :, None, None, :]

        dx, dy = delta[..., 0], delta[..., 1]
        longitudinal = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy

        length = actor_state[..., 6].abs().clamp_min(0.1)
        width = actor_state[..., 7].abs().clamp_min(0.1)
        ego_half_length = 0.5 * float(self.config.ego_length_m) if self.config.inflate_actor_by_ego_footprint else 0.0
        ego_half_width = 0.5 * float(self.config.ego_width_m) if self.config.inflate_actor_by_ego_footprint else 0.0
        sigma_x = torch.maximum(
            0.5 * length + ego_half_length + float(self.config.longitudinal_margin_m),
            torch.full_like(length, float(self.config.minimum_sigma_x_m)),
        )[:, :, None, None, :]
        sigma_y = torch.maximum(
            0.5 * width + ego_half_width + float(self.config.lateral_margin_m),
            torch.full_like(width, float(self.config.minimum_sigma_y_m)),
        )[:, :, None, None, :]

        mahalanobis = (longitudinal / sigma_x).square() + (lateral / sigma_y).square()
        per_actor = torch.exp(-0.5 * mahalanobis)
        valid = actor_valid_mask[:, :, None, None, :]
        per_actor = torch.where(valid, per_actor, torch.zeros_like(per_actor))
        one_minus = (1.0 - per_actor).clamp(1.0e-6, 1.0)
        union_risk = (1.0 - torch.prod(one_minus, dim=-1)).clamp(0.0, 1.0)

        if squeeze_mode:
            union_risk = union_risk.squeeze(2)
            per_actor = per_actor.squeeze(2)
        return RiskFieldResult(
            risk=union_risk,
            per_actor_risk=per_actor,
            actor_future_xy=future_hr,
        )


class MultiSourceSafetyRiskField:
    """Background + platoon + road bounded union used by Risk-PACT-lite."""

    def __init__(self, config: RiskPACTConfig | None = None) -> None:
        self.config = config or RiskPACTConfig()
        self.actor_field = DynamicGaussianRiskField(self.config)
        self.road_field = RoadBoundaryRiskField(self.config)

    @staticmethod
    def _bounded_union(parts: list[Tensor]) -> Tensor:
        if not parts:
            raise ValueError("at least one risk component must be present")
        one_minus = torch.ones_like(parts[0])
        for part in parts:
            one_minus = one_minus * (1.0 - part).clamp(1.0e-6, 1.0)
        return (1.0 - one_minus).clamp(0.0, 1.0)

    def query(
        self,
        trajectory_xy: Tensor,
        *,
        background_actor_state: Tensor | None = None,
        background_actor_valid_mask: Tensor | None = None,
        platoon_actor_state: Tensor | None = None,
        platoon_actor_valid_mask: Tensor | None = None,
        road_sdf: Tensor | None = None,
    ) -> RiskFieldResult:
        squeeze_mode = trajectory_xy.ndim == 4
        canonical = trajectory_xy.unsqueeze(2) if squeeze_mode else trajectory_xy
        if canonical.ndim != 5 or canonical.shape[-1] != 2:
            raise ValueError("trajectory_xy must have shape [B,R,M,H,2] or [B,R,H,2]")

        component_risks: list[Tensor] = []
        per_actor_parts: list[Tensor] = []
        future_parts: list[Tensor] = []
        background_risk = None
        platoon_risk = None
        road_risk = None
        road_signed_distance = None

        if self.config.use_background_actor and background_actor_state is not None:
            if background_actor_valid_mask is None:
                raise ValueError("background actor state requires a validity mask")
            bg = self.actor_field.query(canonical, background_actor_state, background_actor_valid_mask)
            background_risk = bg.risk
            component_risks.append(background_risk)
            per_actor_parts.append(bg.per_actor_risk)
            future_parts.append(bg.actor_future_xy)

        if self.config.use_platoon_actor and platoon_actor_state is not None:
            if platoon_actor_valid_mask is None:
                raise ValueError("platoon actor state requires a validity mask")
            platoon = self.actor_field.query(canonical, platoon_actor_state, platoon_actor_valid_mask)
            platoon_risk = platoon.risk
            component_risks.append(platoon_risk)
            per_actor_parts.append(platoon.per_actor_risk)
            future_parts.append(platoon.actor_future_xy)

        if self.config.use_road_boundary and road_sdf is not None:
            road = self.road_field.query(canonical, road_sdf)
            road_risk = road.risk
            road_signed_distance = road.signed_distance_m
            component_risks.append(road_risk)

        if not component_risks:
            raise ValueError("no enabled Risk-PACT component has scene data")

        combined = self._bounded_union(component_risks)
        b, r, _, horizon, _ = canonical.shape
        if per_actor_parts:
            per_actor = torch.cat(per_actor_parts, dim=-1)
            future = torch.cat(future_parts, dim=3)
        else:
            per_actor = canonical.new_zeros((*canonical.shape[:-1], 0))
            future = canonical.new_zeros((b, r, horizon, 0, 2))

        if squeeze_mode:
            combined = combined.squeeze(2)
            per_actor = per_actor.squeeze(2)
            if background_risk is not None:
                background_risk = background_risk.squeeze(2)
            if platoon_risk is not None:
                platoon_risk = platoon_risk.squeeze(2)
            if road_risk is not None:
                road_risk = road_risk.squeeze(2)
            if road_signed_distance is not None:
                road_signed_distance = road_signed_distance.squeeze(2)

        return RiskFieldResult(
            risk=combined,
            per_actor_risk=per_actor,
            actor_future_xy=future,
            background_risk=background_risk,
            platoon_risk=platoon_risk,
            road_risk=road_risk,
            road_signed_distance_m=road_signed_distance,
        )
