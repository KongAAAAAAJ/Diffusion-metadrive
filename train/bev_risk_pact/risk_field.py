from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import RiskPACTConfig
from .road_field import RoadBoundaryRiskField


@dataclass(frozen=True)
class RiskFieldResult:
    """Risk queried along candidate trajectories.

    ``risk`` is the final smooth-max multi-source field. ``per_actor_risk`` and
    ``actor_future_xy`` concatenate enabled background and platoon actors for
    diagnostics. Component tensors expose each source separately.
    """

    risk: Tensor
    per_actor_risk: Tensor
    actor_future_xy: Tensor
    background_risk: Tensor | None = None
    platoon_risk: Tensor | None = None
    road_risk: Tensor | None = None
    road_signed_distance_m: Tensor | None = None
    per_actor_signed_clearance_m: Tensor | None = None
    background_signed_clearance_m: Tensor | None = None
    platoon_signed_clearance_m: Tensor | None = None


def _masked_softmax_weighted_max(values: Tensor, valid: Tensor, *, beta: float, dim: int) -> Tensor:
    """Differentiable bounded smooth-max that is invariant to invalid entries.

    This is a softmax-weighted average, not log-sum-exp. Therefore if all valid
    entries have the same risk r, the aggregate remains exactly r instead of
    increasing with the number of actors/components.
    """

    if valid.dtype != torch.bool:
        raise ValueError("valid mask must be boolean")
    if values.shape != valid.shape:
        raise ValueError("values and valid must have identical shapes")
    masked_logits = torch.where(
        valid,
        float(beta) * values,
        torch.full_like(values, -1.0e9),
    )
    weights = torch.softmax(masked_logits, dim=dim) * valid.to(values.dtype)
    denom = weights.sum(dim=dim, keepdim=True)
    weights = weights / denom.clamp_min(1.0e-12)
    result = (weights * values).sum(dim=dim)
    has_valid = valid.any(dim=dim)
    return torch.where(has_valid, result, torch.zeros_like(result))


class DynamicActorClearanceRiskField:
    """Differentiable dynamic actor field based on oriented-box clearance.

    For an actor-aligned relative point (longitudinal, lateral), construct an
    ego-inflated rectangular safety set with half extents

      a = (L_actor + L_ego)/2 + longitudinal_clearance
      b = (W_actor + W_ego)/2 + lateral_clearance

    and compute the standard oriented-box signed distance. Positive distance is
    outside the safety box; negative distance is inside. Risk is

      sigmoid(-signed_clearance / actor_temperature_m).

    Actor aggregation uses a smooth max, avoiding the actor-count inflation of
    the previous probabilistic union.
    """

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

    @staticmethod
    def _oriented_box_signed_distance(
        longitudinal: Tensor,
        lateral: Tensor,
        half_extent_x: Tensor,
        half_extent_y: Tensor,
    ) -> Tensor:
        # Standard rectangle SDF. It is differentiable almost everywhere and
        # provides a physically interpretable zero level set.
        qx = longitudinal.abs() - half_extent_x
        qy = lateral.abs() - half_extent_y
        outside_x = torch.relu(qx)
        outside_y = torch.relu(qy)
        outside = torch.sqrt(outside_x.square() + outside_y.square() + 1.0e-12)
        inside = torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx))
        return outside + inside

    def query(
        self,
        trajectory_xy: Tensor,
        actor_state: Tensor,
        actor_valid_mask: Tensor,
        *,
        longitudinal_clearance_m: float | None = None,
        lateral_clearance_m: float | None = None,
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
        if longitudinal_clearance_m is None:
            longitudinal_clearance_m = float(self.config.background_longitudinal_clearance_m)
        if lateral_clearance_m is None:
            lateral_clearance_m = float(self.config.background_lateral_clearance_m)
        if longitudinal_clearance_m < 0.0 or lateral_clearance_m < 0.0:
            raise ValueError("actor clearances must be non-negative")

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
        half_extent_x = (
            0.5 * length
            + 0.5 * float(self.config.ego_length_m)
            + float(longitudinal_clearance_m)
        )[:, :, None, None, :]
        half_extent_y = (
            0.5 * width
            + 0.5 * float(self.config.ego_width_m)
            + float(lateral_clearance_m)
        )[:, :, None, None, :]

        signed_clearance = self._oriented_box_signed_distance(
            longitudinal, lateral, half_extent_x, half_extent_y
        )
        per_actor = torch.sigmoid(-signed_clearance / float(self.config.actor_temperature_m))
        valid = actor_valid_mask[:, :, None, None, :].expand_as(per_actor)
        per_actor = torch.where(valid, per_actor, torch.zeros_like(per_actor))
        aggregate = _masked_softmax_weighted_max(
            per_actor,
            valid,
            beta=float(self.config.actor_softmax_beta),
            dim=-1,
        ).clamp(0.0, 1.0)

        signed_clearance = torch.where(valid, signed_clearance, torch.full_like(signed_clearance, float("inf")))
        if squeeze_mode:
            aggregate = aggregate.squeeze(2)
            per_actor = per_actor.squeeze(2)
            signed_clearance = signed_clearance.squeeze(2)
        return RiskFieldResult(
            risk=aggregate,
            per_actor_risk=per_actor,
            actor_future_xy=future_hr,
            per_actor_signed_clearance_m=signed_clearance,
        )


# Compatibility alias for older pilot imports. Its semantics are Step-5.6
# signed-clearance, not Gaussian.
DynamicGaussianRiskField = DynamicActorClearanceRiskField


class MultiSourceSafetyRiskField:
    """Background + platoon + road smooth-max field used by Risk-PACT-lite."""

    def __init__(self, config: RiskPACTConfig | None = None) -> None:
        self.config = config or RiskPACTConfig()
        self.actor_field = DynamicActorClearanceRiskField(self.config)
        self.road_field = RoadBoundaryRiskField(self.config)

    def _smooth_component_max(self, parts: list[Tensor]) -> Tensor:
        if not parts:
            raise ValueError("at least one risk component must be present")
        stacked = torch.stack(parts, dim=-1)
        valid = torch.ones_like(stacked, dtype=torch.bool)
        return _masked_softmax_weighted_max(
            stacked,
            valid,
            beta=float(self.config.component_softmax_beta),
            dim=-1,
        ).clamp(0.0, 1.0)

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
        background_signed_clearance = None
        platoon_signed_clearance = None

        if self.config.use_background_actor and background_actor_state is not None:
            if background_actor_valid_mask is None:
                raise ValueError("background actor state requires a validity mask")
            bg = self.actor_field.query(
                canonical,
                background_actor_state,
                background_actor_valid_mask,
                longitudinal_clearance_m=float(self.config.background_longitudinal_clearance_m),
                lateral_clearance_m=float(self.config.background_lateral_clearance_m),
            )
            background_risk = bg.risk
            background_signed_clearance = bg.per_actor_signed_clearance_m
            component_risks.append(background_risk)
            per_actor_parts.append(bg.per_actor_risk)
            future_parts.append(bg.actor_future_xy)

        if self.config.use_platoon_actor and platoon_actor_state is not None:
            if platoon_actor_valid_mask is None:
                raise ValueError("platoon actor state requires a validity mask")
            platoon = self.actor_field.query(
                canonical,
                platoon_actor_state,
                platoon_actor_valid_mask,
                longitudinal_clearance_m=float(self.config.platoon_longitudinal_clearance_m),
                lateral_clearance_m=float(self.config.platoon_lateral_clearance_m),
            )
            platoon_risk = platoon.risk
            platoon_signed_clearance = platoon.per_actor_signed_clearance_m
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

        combined = self._smooth_component_max(component_risks)
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
            if background_signed_clearance is not None:
                background_signed_clearance = background_signed_clearance.squeeze(2)
            if platoon_signed_clearance is not None:
                platoon_signed_clearance = platoon_signed_clearance.squeeze(2)

        all_signed_clearance = None
        clearance_parts = [
            value for value in (background_signed_clearance, platoon_signed_clearance)
            if value is not None
        ]
        if clearance_parts:
            all_signed_clearance = torch.cat(clearance_parts, dim=-1)

        return RiskFieldResult(
            risk=combined,
            per_actor_risk=per_actor,
            actor_future_xy=future,
            background_risk=background_risk,
            platoon_risk=platoon_risk,
            road_risk=road_risk,
            road_signed_distance_m=road_signed_distance,
            per_actor_signed_clearance_m=all_signed_clearance,
            background_signed_clearance_m=background_signed_clearance,
            platoon_signed_clearance_m=platoon_signed_clearance,
        )
