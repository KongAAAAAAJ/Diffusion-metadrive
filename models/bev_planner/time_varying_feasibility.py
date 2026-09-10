"""Differentiable time-varying trajectory feasibility losses for GRPO training.

The module operates on the metric clean trajectory predicted at each diffusion
step. It converts XY waypoints to a regularized path curvature, maps curvature
to the bicycle-model equivalent steering angle ``delta = atan(L * kappa)``, and
penalizes violations of steering-angle and steering-rate bounds with normalized
squared hinge losses.

No safe target is generated and the GRPO rollout distribution is unchanged.
The losses are fully differentiable with respect to the planner trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


class TimeVaryingFeasibilityError(RuntimeError):
    """Raised when the trajectory-feasibility contract is violated."""


@dataclass(frozen=True)
class SteeringFeasibilityResult:
    """Per-trajectory differentiable losses and graph-free diagnostics.

    Loss tensors have shape ``xy.shape[:-2]`` and retain gradients. Diagnostic
    tensors have the same leading shape and are detached.
    """

    steering_loss: Tensor
    steering_rate_loss: Tensor
    steering_limit_deg: float
    steering_rate_limit_deg_s: float
    steering_violation_fraction: Tensor
    steering_rate_violation_fraction: Tensor
    max_abs_steering_rad: Tensor
    max_abs_steering_rate_rad_s: Tensor
    degenerate_segment_fraction: Tensor


def _validate_xy(xy: Tensor) -> None:
    if not isinstance(xy, Tensor) or xy.ndim < 2 or xy.shape[-1] != 2:
        raise TimeVaryingFeasibilityError(
            "xy must be a tensor with shape [...,H,2]"
        )
    if xy.shape[-2] < 4:
        raise TimeVaryingFeasibilityError(
            "steering-rate feasibility requires at least four path points"
        )
    if not xy.is_floating_point():
        raise TimeVaryingFeasibilityError("xy must use a floating point dtype")
    if not bool(torch.isfinite(xy).all()):
        raise TimeVaryingFeasibilityError("xy must be finite")


def _regularized_path_geometry(
    xy: Tensor,
    *,
    min_segment_length_m: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return curvature, valid triplets, and segment lengths.

    Curvature is wrapped heading change divided by centered path length. A
    triplet is valid only when both adjacent segments are at least
    ``min_segment_length_m``. Near-overlapping segments are masked so they do
    not create singular steering gradients.
    """

    _validate_xy(xy)
    if (
        not math.isfinite(float(min_segment_length_m))
        or float(min_segment_length_m) <= 0.0
    ):
        raise TimeVaryingFeasibilityError(
            "min_segment_length_m must be positive and finite"
        )

    segment = xy[..., 1:, :] - xy[..., :-1, :]
    segment_length = torch.linalg.vector_norm(segment, dim=-1)
    valid_segment = segment_length >= float(min_segment_length_m)

    # atan2(0, 0) is ill-conditioned. Invalid segments use a constant fallback
    # direction and are excluded from every loss through the validity mask.
    fallback = torch.zeros_like(segment)
    fallback[..., 0] = 1.0
    safe_segment = torch.where(valid_segment[..., None], segment, fallback)
    heading = torch.atan2(safe_segment[..., 1], safe_segment[..., 0])

    raw_heading_delta = heading[..., 1:] - heading[..., :-1]
    heading_delta = torch.atan2(
        torch.sin(raw_heading_delta), torch.cos(raw_heading_delta)
    )
    centered_ds = 0.5 * (segment_length[..., :-1] + segment_length[..., 1:])
    centered_ds = centered_ds.clamp_min(float(min_segment_length_m))
    valid_triplet = valid_segment[..., :-1] & valid_segment[..., 1:]

    curvature = heading_delta / centered_ds
    curvature = torch.where(valid_triplet, curvature, torch.zeros_like(curvature))
    return curvature, valid_triplet, segment_length


def steering_angle_xy(
    xy: Tensor,
    *,
    wheelbase_m: float,
    min_segment_length_m: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return equivalent steering angle, valid triplets, and segment lengths.

    Steering is in radians and follows the kinematic bicycle relation
    ``delta = atan(L * kappa)``.
    """

    if not math.isfinite(float(wheelbase_m)) or float(wheelbase_m) <= 0.0:
        raise TimeVaryingFeasibilityError("wheelbase_m must be positive and finite")
    curvature, valid_triplet, segment_length = _regularized_path_geometry(
        xy, min_segment_length_m=min_segment_length_m
    )
    steering = torch.atan(float(wheelbase_m) * curvature)
    return steering, valid_triplet, segment_length


def time_varying_limit(
    *,
    timestep: int,
    initial_timestep: int,
    initial_limit: float,
    final_limit: float,
    schedule_power: float,
) -> float:
    """Return a loose-to-tight positive bound for one diffusion timestep."""

    if isinstance(timestep, bool) or not isinstance(timestep, int):
        raise TimeVaryingFeasibilityError("timestep must be an integer")
    if (
        isinstance(initial_timestep, bool)
        or not isinstance(initial_timestep, int)
        or initial_timestep <= 0
    ):
        raise TimeVaryingFeasibilityError(
            "initial_timestep must be a positive integer"
        )
    if timestep < 0 or timestep > initial_timestep:
        raise TimeVaryingFeasibilityError(
            "timestep must lie in [0, initial_timestep]"
        )
    for name, value in (
        ("initial_limit", initial_limit),
        ("final_limit", final_limit),
        ("schedule_power", schedule_power),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise TimeVaryingFeasibilityError(f"{name} must be positive and finite")
    if float(initial_limit) < float(final_limit):
        raise TimeVaryingFeasibilityError(
            "initial_limit must be greater than or equal to final_limit"
        )

    remaining = float(timestep) / float(initial_timestep)
    return float(final_limit) + (
        float(initial_limit) - float(final_limit)
    ) * remaining ** float(schedule_power)


def _masked_mean_last(value: Tensor, mask: Tensor) -> Tensor:
    if value.shape != mask.shape or mask.dtype != torch.bool:
        raise TimeVaryingFeasibilityError("masked feasibility tensors are invalid")
    count = mask.to(value.dtype).sum(dim=-1)
    total = torch.where(mask, value, torch.zeros_like(value)).sum(dim=-1)
    return torch.where(
        count > 0,
        total / count.clamp_min(1.0),
        torch.zeros_like(total),
    )


def _masked_max_abs_last(value: Tensor, mask: Tensor) -> Tensor:
    if value.shape != mask.shape or mask.dtype != torch.bool:
        raise TimeVaryingFeasibilityError("masked feasibility tensors are invalid")
    return torch.where(mask, value.abs(), torch.zeros_like(value)).amax(dim=-1)


def steering_feasibility_loss(
    xy_metric: Tensor,
    *,
    steering_limit_deg: float,
    steering_rate_limit_deg_s: float,
    wheelbase_m: float,
    trajectory_dt_s: float = 0.5,
    min_segment_length_m: float = 0.2,
) -> SteeringFeasibilityResult:
    """Compute differentiable steering and steering-rate feasibility losses.

    The normalized losses are

    ``relu(|delta| / delta_limit - 1)^2`` and
    ``relu(|delta_dot| / delta_dot_limit - 1)^2``.

    Each component is first averaged over valid local geometry within one
    trajectory. Near-degenerate segments are excluded and reported separately.
    """

    _validate_xy(xy_metric)
    for name, value in (
        ("steering_limit_deg", steering_limit_deg),
        ("steering_rate_limit_deg_s", steering_rate_limit_deg_s),
        ("wheelbase_m", wheelbase_m),
        ("trajectory_dt_s", trajectory_dt_s),
        ("min_segment_length_m", min_segment_length_m),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise TimeVaryingFeasibilityError(f"{name} must be positive and finite")
    if float(steering_limit_deg) >= 90.0:
        raise TimeVaryingFeasibilityError(
            "steering_limit_deg must be smaller than 90 degrees"
        )

    steering_limit_rad = math.radians(float(steering_limit_deg))
    steering_rate_limit_rad_s = math.radians(float(steering_rate_limit_deg_s))

    steering, valid_triplet, segment_length = steering_angle_xy(
        xy_metric,
        wheelbase_m=wheelbase_m,
        min_segment_length_m=min_segment_length_m,
    )
    steering_excess = F.relu(steering.abs() / steering_limit_rad - 1.0)
    steering_loss = _masked_mean_last(steering_excess.square(), valid_triplet)

    valid_rate = valid_triplet[..., 1:] & valid_triplet[..., :-1]
    steering_rate = (
        steering[..., 1:] - steering[..., :-1]
    ) / float(trajectory_dt_s)
    steering_rate_excess = F.relu(
        steering_rate.abs() / steering_rate_limit_rad_s - 1.0
    )
    steering_rate_loss = _masked_mean_last(
        steering_rate_excess.square(), valid_rate
    )

    steering_violation = _masked_mean_last(
        (steering.abs() > steering_limit_rad).to(xy_metric.dtype),
        valid_triplet,
    )
    steering_rate_violation = _masked_mean_last(
        (steering_rate.abs() > steering_rate_limit_rad_s).to(xy_metric.dtype),
        valid_rate,
    )
    max_abs_steering = _masked_max_abs_last(steering, valid_triplet)
    max_abs_steering_rate = _masked_max_abs_last(steering_rate, valid_rate)
    degenerate_fraction = (
        segment_length < float(min_segment_length_m)
    ).to(xy_metric.dtype).mean(dim=-1)

    diagnostics = (
        steering_violation,
        steering_rate_violation,
        max_abs_steering,
        max_abs_steering_rate,
        degenerate_fraction,
    )
    if not all(bool(torch.isfinite(value).all()) for value in diagnostics):
        raise TimeVaryingFeasibilityError(
            "steering feasibility diagnostics contain non-finite values"
        )
    if not bool(torch.isfinite(steering_loss).all()) or not bool(
        torch.isfinite(steering_rate_loss).all()
    ):
        raise TimeVaryingFeasibilityError(
            "steering feasibility loss contains non-finite values"
        )

    return SteeringFeasibilityResult(
        steering_loss=steering_loss,
        steering_rate_loss=steering_rate_loss,
        steering_limit_deg=float(steering_limit_deg),
        steering_rate_limit_deg_s=float(steering_rate_limit_deg_s),
        steering_violation_fraction=steering_violation.detach(),
        steering_rate_violation_fraction=steering_rate_violation.detach(),
        max_abs_steering_rad=max_abs_steering.detach(),
        max_abs_steering_rate_rad_s=max_abs_steering_rate.detach(),
        degenerate_segment_fraction=degenerate_fraction.detach(),
    )


__all__ = [
    "SteeringFeasibilityResult",
    "TimeVaryingFeasibilityError",
    "steering_angle_xy",
    "steering_feasibility_loss",
    "time_varying_limit",
]
