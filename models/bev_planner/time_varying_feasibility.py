"""Differentiable steering / steering-rate feasibility loss for GRPO replay."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


class TimeVaryingFeasibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class SteeringFeasibilityResult:
    steering_loss: Tensor
    steering_rate_loss: Tensor
    steering_limit_deg: float
    steering_rate_limit_deg_s: float
    steering_violation_fraction: Tensor
    steering_rate_violation_fraction: Tensor
    max_abs_steering_rad: Tensor
    max_abs_steering_rate_rad_s: Tensor
    degenerate_segment_fraction: Tensor


def time_varying_limit(*, timestep: int, initial_timestep: int, initial_limit: float,
                       final_limit: float, schedule_power: float = 1.0) -> float:
    if initial_timestep <= 0:
        raise TimeVaryingFeasibilityError("initial_timestep must be positive")
    if schedule_power <= 0.0:
        raise TimeVaryingFeasibilityError("schedule_power must be positive")
    ratio = max(0.0, min(1.0, float(timestep) / float(initial_timestep)))
    # t=initial -> loose initial_limit; t=0 -> tight final_limit.
    progress = ratio ** float(schedule_power)
    return float(final_limit) + (float(initial_limit) - float(final_limit)) * progress


def _geometry(xy: Tensor, min_segment_length_m: float) -> tuple[Tensor, Tensor, Tensor]:
    if xy.shape[-2] < 3 or xy.shape[-1] != 2:
        raise TimeVaryingFeasibilityError("xy must have shape [...,H,2] with H>=3")
    delta = xy[..., 1:, :] - xy[..., :-1, :]
    seg = torch.linalg.vector_norm(delta, dim=-1)
    valid_seg = seg >= float(min_segment_length_m)
    heading = torch.atan2(delta[..., 1], delta[..., 0])
    dhead = torch.atan2(
        torch.sin(heading[..., 1:] - heading[..., :-1]),
        torch.cos(heading[..., 1:] - heading[..., :-1]),
    )
    ds = 0.5 * (seg[..., 1:] + seg[..., :-1])
    valid_triplet = valid_seg[..., 1:] & valid_seg[..., :-1]
    curvature = torch.where(
        valid_triplet,
        dhead / ds.clamp_min(float(min_segment_length_m)),
        torch.zeros_like(dhead),
    )
    return curvature, valid_triplet, valid_seg


def steering_angle_xy(xy_metric: Tensor, *, wheelbase_m: float,
                      min_segment_length_m: float = 0.2) -> tuple[Tensor, Tensor, Tensor]:
    curvature, valid_triplet, valid_seg = _geometry(xy_metric, min_segment_length_m)
    steering = torch.atan(float(wheelbase_m) * curvature)
    return steering, valid_triplet, valid_seg


def _masked_mean_last(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(value.dtype)
    count = weight.sum(dim=-1)
    num = (value * weight).sum(dim=-1)
    return torch.where(count > 0, num / count.clamp_min(1.0), torch.zeros_like(num))


def _masked_max_abs_last(value: Tensor, mask: Tensor) -> Tensor:
    neg_inf = torch.full_like(value, float('-inf'))
    masked = torch.where(mask, value.abs(), neg_inf)
    out = masked.max(dim=-1).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def steering_feasibility_loss(
    xy_metric: Tensor,
    *,
    steering_limit_deg: float,
    steering_rate_limit_deg_s: float,
    wheelbase_m: float,
    trajectory_dt_s: float,
    min_segment_length_m: float = 0.2,
) -> SteeringFeasibilityResult:
    if not xy_metric.is_floating_point() or not bool(torch.isfinite(xy_metric).all()):
        raise TimeVaryingFeasibilityError("xy_metric must be finite floating point")
    if trajectory_dt_s <= 0 or wheelbase_m <= 0 or min_segment_length_m <= 0:
        raise TimeVaryingFeasibilityError("geometry parameters must be positive")
    steering_limit_rad = math.radians(float(steering_limit_deg))
    steering_rate_limit_rad_s = math.radians(float(steering_rate_limit_deg_s))
    steering, valid_triplet, valid_seg = steering_angle_xy(
        xy_metric, wheelbase_m=wheelbase_m,
        min_segment_length_m=min_segment_length_m,
    )
    steering_norm = steering.abs() / steering_limit_rad
    steering_penalty = torch.relu(steering_norm - 1.0).square()
    steering_loss = _masked_mean_last(steering_penalty, valid_triplet)

    steering_rate = (steering[..., 1:] - steering[..., :-1]) / float(trajectory_dt_s)
    valid_rate = valid_triplet[..., 1:] & valid_triplet[..., :-1]
    rate_norm = steering_rate.abs() / steering_rate_limit_rad_s
    rate_penalty = torch.relu(rate_norm - 1.0).square()
    steering_rate_loss = _masked_mean_last(rate_penalty, valid_rate)

    steering_violation = _masked_mean_last(
        (steering.abs() > steering_limit_rad).to(xy_metric.dtype), valid_triplet
    )
    steering_rate_violation = _masked_mean_last(
        (steering_rate.abs() > steering_rate_limit_rad_s).to(xy_metric.dtype), valid_rate
    )
    max_abs_steering = _masked_max_abs_last(steering, valid_triplet)
    max_abs_rate = _masked_max_abs_last(steering_rate, valid_rate)
    degenerate_fraction = (~valid_seg).to(xy_metric.dtype).mean(dim=-1)

    for value in (steering_loss, steering_rate_loss, steering_violation,
                  steering_rate_violation, max_abs_steering, max_abs_rate,
                  degenerate_fraction):
        if not bool(torch.isfinite(value).all()):
            raise TimeVaryingFeasibilityError("feasibility result contains non-finite values")

    return SteeringFeasibilityResult(
        steering_loss=steering_loss,
        steering_rate_loss=steering_rate_loss,
        steering_limit_deg=float(steering_limit_deg),
        steering_rate_limit_deg_s=float(steering_rate_limit_deg_s),
        steering_violation_fraction=steering_violation.detach(),
        steering_rate_violation_fraction=steering_rate_violation.detach(),
        max_abs_steering_rad=max_abs_steering.detach(),
        max_abs_steering_rate_rad_s=max_abs_rate.detach(),
        degenerate_segment_fraction=degenerate_fraction.detach(),
    )


__all__ = [
    "SteeringFeasibilityResult", "TimeVaryingFeasibilityError",
    "steering_angle_xy", "steering_feasibility_loss", "time_varying_limit",
]
