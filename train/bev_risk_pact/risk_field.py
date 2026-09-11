from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import RiskPACTConfig


@dataclass(frozen=True)
class RiskFieldResult:
    """Risk queried along candidate trajectories.

    Attributes
    ----------
    risk:
        Union-like actor risk in [0,1], shape ``[..., H]`` where the leading
        axes are the candidate trajectory axes (e.g. [B, R, M]).
    per_actor_risk:
        Individual anisotropic-Gaussian risks, shape ``[..., H, A]``.
    actor_future_xy:
        Constant-velocity actor forecasts, shape [B, R, H, A, 2].
    """

    risk: Tensor
    per_actor_risk: Tensor
    actor_future_xy: Tensor


class DynamicGaussianRiskField:
    """Analytic differentiable spatio-temporal risk field for the pilot.

    Background actor state contract follows the current BEV planner's 8-D
    actor token convention inferred from its normalization scales:
    [x, y, sin(yaw), cos(yaw), vx, vy, length, width].

    Actor motion is constant velocity over the 8-step planning horizon.  The
    risk field is an anisotropic Gaussian aligned with each actor's heading.
    Individual actor risks are combined with a differentiable probabilistic
    union ``1-prod(1-r_j)`` so the aggregate stays in [0,1].
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

    def query(
        self,
        trajectory_xy: Tensor,
        actor_state: Tensor,
        actor_valid_mask: Tensor,
    ) -> RiskFieldResult:
        """Query risk along candidate center-point trajectories.

        Parameters
        ----------
        trajectory_xy:
            [B,R,M,H,2] or [B,R,H,2].
        actor_state:
            [B,R,A,8].
        actor_valid_mask:
            [B,R,A] boolean.
        """
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

        b, r, _, h, _ = trajectory_xy.shape
        a = actor_state.shape[2]
        future = self.actor_future_xy(actor_state, h)  # [B,R,A,H,2]
        future_hr = future.permute(0, 1, 3, 2, 4)  # [B,R,H,A,2]

        delta = trajectory_xy.unsqueeze(-2) - future_hr.unsqueeze(2)  # [B,R,M,H,A,2]

        sin_yaw = actor_state[..., 2]
        cos_yaw = actor_state[..., 3]
        norm = torch.sqrt(cos_yaw.square() + sin_yaw.square()).clamp_min(1.0e-6)
        cos_yaw = cos_yaw / norm
        sin_yaw = sin_yaw / norm
        cos_yaw = cos_yaw[:, :, None, None, :]
        sin_yaw = sin_yaw[:, :, None, None, :]

        dx, dy = delta[..., 0], delta[..., 1]
        longitudinal = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy

        length = actor_state[..., 6].abs().clamp_min(0.1)
        width = actor_state[..., 7].abs().clamp_min(0.1)
        sigma_x = torch.maximum(
            0.5 * length + float(self.config.longitudinal_margin_m),
            torch.full_like(length, float(self.config.minimum_sigma_x_m)),
        )[:, :, None, None, :]
        sigma_y = torch.maximum(
            0.5 * width + float(self.config.lateral_margin_m),
            torch.full_like(width, float(self.config.minimum_sigma_y_m)),
        )[:, :, None, None, :]

        mahalanobis = (longitudinal / sigma_x).square() + (lateral / sigma_y).square()
        per_actor = torch.exp(-0.5 * mahalanobis)
        valid = actor_valid_mask[:, :, None, None, :]
        per_actor = torch.where(valid, per_actor, torch.zeros_like(per_actor))

        # Smooth bounded union. Clamp avoids log/gradient pathologies at exactly 1.
        one_minus = (1.0 - per_actor).clamp(1.0e-6, 1.0)
        union_risk = 1.0 - torch.prod(one_minus, dim=-1)
        union_risk = union_risk.clamp(0.0, 1.0)

        future_out = future_hr
        if squeeze_mode:
            union_risk = union_risk.squeeze(2)
            per_actor = per_actor.squeeze(2)
        return RiskFieldResult(
            risk=union_risk,
            per_actor_risk=per_actor,
            actor_future_xy=future_out,
        )
