"""Lightweight time-varying curvature barrier targets for diffusion training.

The module intentionally stays independent of the GRPO sampler.  It projects a
metric XY trajectory onto a progressively tightened curvature-safe set and
returns a detached teacher target.  This keeps the original GRPO behavior
policy and its log-probability contract unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class TimeVaryingCBFError(RuntimeError):
    """Raised when the curvature barrier target contract is violated."""


@dataclass(frozen=True)
class CurvatureCBFTargetResult:
    """Detached target and diagnostics for one batched curvature projection."""

    target_xy: Tensor
    curvature_limit_inv_m: float
    nominal_max_abs_curvature: Tensor
    target_max_abs_curvature: Tensor
    nominal_violation_fraction: Tensor
    target_violation_fraction: Tensor
    correction_rms_m: Tensor
    correction_clipped_fraction: Tensor


def discrete_curvature_xy(xy: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Return three-point signed curvature for ``[..., H, 2]`` metric XY paths.

    The output shape is ``[..., H-2]`` and curvature is in ``1 / metre`` when
    ``xy`` is expressed in metres.
    """

    if not isinstance(xy, Tensor) or xy.ndim < 2 or xy.shape[-1] != 2:
        raise TimeVaryingCBFError("xy must be a tensor with shape [...,H,2]")
    if xy.shape[-2] < 3:
        raise TimeVaryingCBFError("curvature requires at least three path points")
    if not xy.is_floating_point():
        raise TimeVaryingCBFError("xy must use a floating point dtype")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise TimeVaryingCBFError("eps must be positive and finite")

    p0 = xy[..., :-2, :]
    p1 = xy[..., 1:-1, :]
    p2 = xy[..., 2:, :]
    a = p1 - p0
    b = p2 - p1
    chord = p2 - p0
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    denom = (
        torch.linalg.vector_norm(a, dim=-1)
        * torch.linalg.vector_norm(b, dim=-1)
        * torch.linalg.vector_norm(chord, dim=-1)
    ).clamp_min(float(eps))
    return 2.0 * cross / denom


def time_varying_curvature_limit(
    *,
    timestep: int,
    initial_timestep: int,
    initial_limit_inv_m: float,
    final_limit_inv_m: float,
    schedule_power: float,
) -> float:
    """Return a loose-to-tight curvature bound for the current denoising step."""

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
        ("initial_limit_inv_m", initial_limit_inv_m),
        ("final_limit_inv_m", final_limit_inv_m),
        ("schedule_power", schedule_power),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise TimeVaryingCBFError(f"{name} must be positive and finite")
    if float(initial_limit_inv_m) < float(final_limit_inv_m):
        raise TimeVaryingCBFError(
            "initial curvature limit must be greater than or equal to final limit"
        )

    remaining = float(timestep) / float(initial_timestep)
    return float(final_limit_inv_m) + (
        float(initial_limit_inv_m) - float(final_limit_inv_m)
    ) * remaining ** float(schedule_power)


def _curvature_diagnostics(
    xy: Tensor,
    *,
    curvature_limit_inv_m: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    curvature = discrete_curvature_xy(xy, eps=eps).abs()
    max_abs = curvature.amax(dim=-1)
    violation_fraction = (curvature > float(curvature_limit_inv_m)).float().mean(dim=-1)
    return max_abs, violation_fraction


def curvature_cbf_safe_target(
    xy_metric: Tensor,
    *,
    curvature_limit_inv_m: float,
    projection_passes: int = 2,
    max_correction_m: float = 0.0,
    eps: float = 1e-6,
) -> CurvatureCBFTargetResult:
    """Build a detached minimum-deviation curvature barrier teacher target.

    Each curvature constraint is linearized around the current projected path,
    ``h_k = kappa_limit^2 - kappa_k^2 >= 0``.  A violated linearized half-space
    has a closed-form Euclidean projection.  Cycling through all local
    constraints for a small number of passes provides a vectorized,
    solver-free approximation to the corresponding minimum-deviation QP.

    ``max_correction_m == 0`` disables correction clipping.  A positive value
    caps the per-point displacement of the final teacher target relative to the
    nominal path.  The returned target is always detached.
    """

    if not isinstance(xy_metric, Tensor) or xy_metric.ndim < 2 or xy_metric.shape[-1] != 2:
        raise TimeVaryingCBFError("xy_metric must have shape [...,H,2]")
    if xy_metric.shape[-2] < 3:
        raise TimeVaryingCBFError("curvature target requires at least three points")
    if not xy_metric.is_floating_point():
        raise TimeVaryingCBFError("xy_metric must use a floating point dtype")
    if not bool(torch.isfinite(xy_metric).all()):
        raise TimeVaryingCBFError("xy_metric must be finite")
    if not math.isfinite(float(curvature_limit_inv_m)) or float(curvature_limit_inv_m) <= 0.0:
        raise TimeVaryingCBFError("curvature_limit_inv_m must be positive and finite")
    if isinstance(projection_passes, bool) or not isinstance(projection_passes, int) or projection_passes <= 0:
        raise TimeVaryingCBFError("projection_passes must be a positive integer")
    if not math.isfinite(float(max_correction_m)) or float(max_correction_m) < 0.0:
        raise TimeVaryingCBFError("max_correction_m must be non-negative and finite")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise TimeVaryingCBFError("eps must be positive and finite")

    nominal = xy_metric.detach()
    nominal_max, nominal_fraction = _curvature_diagnostics(
        nominal,
        curvature_limit_inv_m=curvature_limit_inv_m,
        eps=eps,
    )
    target = nominal.clone()
    local_constraint_count = int(target.shape[-2] - 2)

    # The surrounding GRPO post-update stability check runs under no_grad().
    # Re-enable autograd only for the tiny geometric projection graph; the
    # teacher stays detached from planner parameters.
    with torch.enable_grad():
        for _ in range(projection_passes):
            for constraint_index in range(local_constraint_count):
                work = target.detach().requires_grad_(True)
                curvature = discrete_curvature_xy(work, eps=eps)[..., constraint_index]
                barrier = float(curvature_limit_inv_m) ** 2 - curvature.square()
                if not bool((barrier < 0.0).any()):
                    target = work.detach()
                    continue
                gradient = torch.autograd.grad(
                    barrier.sum(),
                    work,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                gradient_sq = gradient.square().sum(dim=(-2, -1)).clamp_min(float(eps))
                # Exact Euclidean projection onto the current linearized
                # half-space h + grad(h)^T delta >= 0.
                scale = (-barrier).clamp_min(0.0) / gradient_sq
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
    target_max, target_fraction = _curvature_diagnostics(
        target,
        curvature_limit_inv_m=curvature_limit_inv_m,
        eps=eps,
    )
    correction_rms = (target - nominal).square().mean(dim=(-2, -1)).sqrt()
    return CurvatureCBFTargetResult(
        target_xy=target,
        curvature_limit_inv_m=float(curvature_limit_inv_m),
        nominal_max_abs_curvature=nominal_max.detach(),
        target_max_abs_curvature=target_max.detach(),
        nominal_violation_fraction=nominal_fraction.detach(),
        target_violation_fraction=target_fraction.detach(),
        correction_rms_m=correction_rms.detach(),
        correction_clipped_fraction=correction_clipped.detach(),
    )


__all__ = [
    "CurvatureCBFTargetResult",
    "TimeVaryingCBFError",
    "curvature_cbf_safe_target",
    "discrete_curvature_xy",
    "time_varying_curvature_limit",
]
