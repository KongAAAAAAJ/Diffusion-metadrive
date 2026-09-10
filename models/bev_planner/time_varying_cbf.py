"""Lightweight time-varying steering-angle barrier targets for diffusion training.

The module stays independent of the GRPO sampler. It converts a metric XY path
into a regularized path curvature, maps curvature to an equivalent bicycle-model
steering angle ``delta = atan(L * kappa)``, and projects the path toward a
progressively tightened steering-feasible set. The returned target is detached,
so the original GRPO behavior policy and log-probability contract are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class TimeVaryingCBFError(RuntimeError):
    """Raised when the steering barrier target contract is violated."""


@dataclass(frozen=True)
class SteeringCBFTargetResult:
    """Detached target and diagnostics for one batched steering projection."""

    target_xy: Tensor
    steering_limit_deg: float
    nominal_max_abs_steering_rad: Tensor
    target_max_abs_steering_rad: Tensor
    nominal_violation_fraction: Tensor
    target_violation_fraction: Tensor
    nominal_degenerate_segment_fraction: Tensor
    target_degenerate_segment_fraction: Tensor
    correction_rms_m: Tensor
    correction_clipped_fraction: Tensor


def _validate_xy(xy: Tensor) -> None:
    if not isinstance(xy, Tensor) or xy.ndim < 2 or xy.shape[-1] != 2:
        raise TimeVaryingCBFError("xy must be a tensor with shape [...,H,2]")
    if xy.shape[-2] < 3:
        raise TimeVaryingCBFError("steering geometry requires at least three path points")
    if not xy.is_floating_point():
        raise TimeVaryingCBFError("xy must use a floating point dtype")
    if not bool(torch.isfinite(xy).all()):
        raise TimeVaryingCBFError("xy must be finite")


def _regularized_path_geometry(
    xy: Tensor,
    *,
    min_segment_length_m: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return regularized curvature, valid triplets, and segment lengths.

    Curvature is computed from wrapped heading change divided by centered path
    length. A triplet is valid only when both adjacent segments are at least
    ``min_segment_length_m`` long. Invalid/near-overlapping triplets are masked
    to zero curvature instead of allowing the geometric curvature denominator
    to become singular.

    Returns:
      curvature: ``[..., H-2]`` in 1/m.
      valid_triplet: bool ``[..., H-2]``.
      segment_length: ``[..., H-1]`` in m.
    """

    _validate_xy(xy)
    if (
        not math.isfinite(float(min_segment_length_m))
        or float(min_segment_length_m) <= 0.0
    ):
        raise TimeVaryingCBFError("min_segment_length_m must be positive and finite")

    segment = xy[..., 1:, :] - xy[..., :-1, :]
    segment_length = torch.linalg.vector_norm(segment, dim=-1)
    valid_segment = segment_length >= float(min_segment_length_m)

    # atan2(0, 0) has an ill-conditioned derivative. Replace invalid segments
    # with a constant forward vector before computing headings; torch.where
    # prevents gradients from flowing through the invalid branch.
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


def regularized_curvature_xy(
    xy: Tensor,
    *,
    min_segment_length_m: float = 0.2,
) -> Tensor:
    """Return regularized path curvature for metric XY trajectories."""

    curvature, _, _ = _regularized_path_geometry(
        xy, min_segment_length_m=min_segment_length_m
    )
    return curvature


def steering_angle_xy(
    xy: Tensor,
    *,
    wheelbase_m: float,
    min_segment_length_m: float = 0.2,
) -> Tensor:
    """Return bicycle-model equivalent steering angle in radians.

    ``delta = atan(L * kappa)`` where ``L`` is the configured wheelbase.
    Near-degenerate triplets are masked to zero by ``regularized_curvature_xy``.
    """

    if not math.isfinite(float(wheelbase_m)) or float(wheelbase_m) <= 0.0:
        raise TimeVaryingCBFError("wheelbase_m must be positive and finite")
    curvature = regularized_curvature_xy(
        xy, min_segment_length_m=min_segment_length_m
    )
    return torch.atan(float(wheelbase_m) * curvature)


def time_varying_steering_limit_deg(
    *,
    timestep: int,
    initial_timestep: int,
    initial_limit_deg: float,
    final_limit_deg: float,
    schedule_power: float,
) -> float:
    """Return a loose-to-tight steering-angle bound in degrees."""

    if isinstance(timestep, bool) or not isinstance(timestep, int):
        raise TimeVaryingCBFError("timestep must be an integer")
    if (
        isinstance(initial_timestep, bool)
        or not isinstance(initial_timestep, int)
        or initial_timestep <= 0
    ):
        raise TimeVaryingCBFError("initial_timestep must be a positive integer")
    if timestep < 0 or timestep > initial_timestep:
        raise TimeVaryingCBFError("timestep must lie in [0, initial_timestep]")
    for name, value in (
        ("initial_limit_deg", initial_limit_deg),
        ("final_limit_deg", final_limit_deg),
        ("schedule_power", schedule_power),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise TimeVaryingCBFError(f"{name} must be positive and finite")
    if float(initial_limit_deg) < float(final_limit_deg):
        raise TimeVaryingCBFError(
            "initial steering limit must be greater than or equal to final limit"
        )
    if float(initial_limit_deg) >= 90.0 or float(final_limit_deg) >= 90.0:
        raise TimeVaryingCBFError("steering limits must be smaller than 90 degrees")

    remaining = float(timestep) / float(initial_timestep)
    return float(final_limit_deg) + (
        float(initial_limit_deg) - float(final_limit_deg)
    ) * remaining ** float(schedule_power)


def _steering_diagnostics(
    xy: Tensor,
    *,
    steering_limit_rad: float,
    wheelbase_m: float,
    min_segment_length_m: float,
) -> tuple[Tensor, Tensor, Tensor]:
    curvature, valid_triplet, segment_length = _regularized_path_geometry(
        xy, min_segment_length_m=min_segment_length_m
    )
    steering = torch.atan(float(wheelbase_m) * curvature).abs()
    valid_count = valid_triplet.float().sum(dim=-1)
    violation_count = (
        (steering > float(steering_limit_rad)) & valid_triplet
    ).float().sum(dim=-1)
    violation_fraction = torch.where(
        valid_count > 0,
        violation_count / valid_count.clamp_min(1.0),
        torch.zeros_like(valid_count),
    )
    max_abs = torch.where(
        valid_triplet,
        steering,
        torch.zeros_like(steering),
    ).amax(dim=-1)
    degenerate_segment_fraction = (
        segment_length < float(min_segment_length_m)
    ).float().mean(dim=-1)
    return max_abs, violation_fraction, degenerate_segment_fraction


def steering_cbf_safe_target(
    xy_metric: Tensor,
    *,
    steering_limit_deg: float,
    wheelbase_m: float,
    min_segment_length_m: float = 0.2,
    projection_passes: int = 2,
    max_correction_m: float = 0.0,
    eps: float = 1e-6,
) -> SteeringCBFTargetResult:
    """Build a detached minimum-deviation steering-barrier teacher target.

    For every valid path triplet, the barrier is
    ``h_k = delta_limit^2 - delta_k^2 >= 0`` with
    ``delta_k = atan(wheelbase * kappa_k)``. Each violated constraint is
    linearized around the current path and projected with the closed-form
    Euclidean half-space update. Near-overlapping segments shorter than
    ``min_segment_length_m`` are excluded from the steering barrier and are
    reported separately through the degenerate-segment diagnostics.

    ``max_correction_m == 0`` disables correction clipping. The returned
    teacher target is always detached from planner parameters.
    """

    _validate_xy(xy_metric)
    for name, value in (
        ("steering_limit_deg", steering_limit_deg),
        ("wheelbase_m", wheelbase_m),
        ("min_segment_length_m", min_segment_length_m),
        ("eps", eps),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise TimeVaryingCBFError(f"{name} must be positive and finite")
    if float(steering_limit_deg) >= 90.0:
        raise TimeVaryingCBFError("steering_limit_deg must be smaller than 90")
    if (
        isinstance(projection_passes, bool)
        or not isinstance(projection_passes, int)
        or projection_passes <= 0
    ):
        raise TimeVaryingCBFError("projection_passes must be a positive integer")
    if not math.isfinite(float(max_correction_m)) or float(max_correction_m) < 0.0:
        raise TimeVaryingCBFError("max_correction_m must be non-negative and finite")

    steering_limit_rad = math.radians(float(steering_limit_deg))
    nominal = xy_metric.detach()
    nominal_max, nominal_fraction, nominal_degenerate = _steering_diagnostics(
        nominal,
        steering_limit_rad=steering_limit_rad,
        wheelbase_m=wheelbase_m,
        min_segment_length_m=min_segment_length_m,
    )
    target = nominal.clone()
    local_constraint_count = int(target.shape[-2] - 2)

    # The GRPO post-update stability check can run under no_grad(). Re-enable
    # autograd only for this small geometric teacher graph. The target remains
    # detached from planner parameters.
    with torch.enable_grad():
        for _ in range(projection_passes):
            for constraint_index in range(local_constraint_count):
                work = target.detach().requires_grad_(True)
                curvature, valid_triplet, _ = _regularized_path_geometry(
                    work, min_segment_length_m=min_segment_length_m
                )
                steering = torch.atan(float(wheelbase_m) * curvature)[
                    ..., constraint_index
                ]
                valid = valid_triplet[..., constraint_index]
                barrier = steering_limit_rad**2 - steering.square()
                violated = valid & (barrier < 0.0)
                if not bool(violated.any()):
                    target = work.detach()
                    continue

                active_barrier = torch.where(
                    violated, barrier, torch.zeros_like(barrier)
                )
                gradient = torch.autograd.grad(
                    active_barrier.sum(),
                    work,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                gradient_sq = gradient.square().sum(dim=(-2, -1)).clamp_min(
                    float(eps)
                )
                scale = torch.where(
                    violated,
                    (-barrier).clamp_min(0.0) / gradient_sq,
                    torch.zeros_like(barrier),
                )
                target = (work + scale[..., None, None] * gradient).detach()

    correction_clipped = torch.zeros(
        nominal.shape[:-2], device=nominal.device, dtype=nominal.dtype
    )
    if float(max_correction_m) > 0.0:
        delta = target - nominal
        point_norm = torch.linalg.vector_norm(delta, dim=-1)
        max_point_norm = point_norm.amax(dim=-1)
        correction_clipped = (
            max_point_norm > float(max_correction_m)
        ).to(nominal.dtype)
        scale = (
            float(max_correction_m)
            / max_point_norm.clamp_min(float(eps))
        ).clamp_max(1.0)
        target = nominal + scale[..., None, None] * delta

    target = target.detach()
    target_max, target_fraction, target_degenerate = _steering_diagnostics(
        target,
        steering_limit_rad=steering_limit_rad,
        wheelbase_m=wheelbase_m,
        min_segment_length_m=min_segment_length_m,
    )
    correction_rms = (target - nominal).square().mean(dim=(-2, -1)).sqrt()
    return SteeringCBFTargetResult(
        target_xy=target,
        steering_limit_deg=float(steering_limit_deg),
        nominal_max_abs_steering_rad=nominal_max.detach(),
        target_max_abs_steering_rad=target_max.detach(),
        nominal_violation_fraction=nominal_fraction.detach(),
        target_violation_fraction=target_fraction.detach(),
        nominal_degenerate_segment_fraction=nominal_degenerate.detach(),
        target_degenerate_segment_fraction=target_degenerate.detach(),
        correction_rms_m=correction_rms.detach(),
        correction_clipped_fraction=correction_clipped.detach(),
    )


__all__ = [
    "SteeringCBFTargetResult",
    "TimeVaryingCBFError",
    "regularized_curvature_xy",
    "steering_angle_xy",
    "steering_cbf_safe_target",
    "time_varying_steering_limit_deg",
]
