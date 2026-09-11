"""Curriculum scheduling utilities for Risk-PACT-lite post-training.

Step 3 is intentionally side-effect free: it only maps an accepted optimizer
update index to a curriculum strength.  It does not touch rollout, loss,
optimizer, or model parameters.

The scheduler exposes both the absolute curriculum strength ``scale`` used by
our current PACT-lite teacher and the incremental ``delta_scale`` that will be
useful if we later move toward the paper's stricter incremental curriculum
projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
import math


class RiskPACTCurriculumConfigLike(Protocol):
    """Structural type accepted by the scheduler.

    ``train.bev_joint_grpo_online.config.RiskPACTCurriculumConfig`` already
    satisfies this protocol, but importing it here would introduce an avoidable
    package dependency/circular-import risk.
    """

    start_scale: float
    end_scale: float
    warmup_updates: int
    ramp_updates: int
    schedule: str


@dataclass(frozen=True)
class RiskPACTCurriculumState:
    """Resolved curriculum state at one accepted optimizer update."""

    update_step: int
    scale: float
    previous_scale: float
    delta_scale: float
    progress: float
    phase: Literal["warmup", "ramp", "steady"]


def _validate_update_step(update_step: int) -> int:
    if isinstance(update_step, bool) or not isinstance(update_step, int):
        raise TypeError("Risk-PACT update_step must be an integer")
    if update_step < 0:
        raise ValueError("Risk-PACT update_step must be non-negative")
    return update_step


def _validate_config(config: RiskPACTCurriculumConfigLike) -> tuple[float, float, int, int, str]:
    start = float(config.start_scale)
    end = float(config.end_scale)
    warmup = int(config.warmup_updates)
    ramp = int(config.ramp_updates)
    schedule = str(config.schedule)

    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Risk-PACT curriculum scales must be finite")
    if not (0.0 <= start <= end <= 1.0):
        raise ValueError("Risk-PACT curriculum requires 0 <= start_scale <= end_scale <= 1")
    if warmup < 0:
        raise ValueError("Risk-PACT warmup_updates must be non-negative")
    if ramp <= 0:
        raise ValueError("Risk-PACT ramp_updates must be positive")
    if schedule != "linear":
        raise ValueError("Risk-PACT Step-3 scheduler currently supports only schedule='linear'")
    return start, end, warmup, ramp, schedule


def _absolute_scale(
    update_step: int,
    *,
    start: float,
    end: float,
    warmup: int,
    ramp: int,
) -> tuple[float, float, Literal["warmup", "ramp", "steady"]]:
    """Return ``(scale, progress, phase)`` for a validated step/config."""

    if update_step < warmup:
        return start, 0.0, "warmup"

    elapsed = update_step - warmup
    if elapsed >= ramp:
        return end, 1.0, "steady"

    progress = float(elapsed) / float(ramp)
    scale = start + (end - start) * progress
    return scale, progress, "ramp"


def risk_pact_curriculum_state(
    update_step: int,
    config: RiskPACTCurriculumConfigLike,
) -> RiskPACTCurriculumState:
    """Resolve the curriculum state for an accepted optimizer-update index.

    Semantics with the default config ``start=0.2, end=1.0, warmup=0,
    ramp=100`` are:

    * step 0   -> scale 0.2
    * step 50  -> scale 0.6
    * step 100 -> scale 1.0
    * step >100 -> scale 1.0

    ``delta_scale`` is ``scale(step) - scale(step-1)`` with a virtual pre-step
    scale of zero at step 0.  Current PACT-lite will use ``scale``; the delta is
    exposed now so a later strict PACT implementation can use incremental
    curriculum projection without changing this API.
    """

    step = _validate_update_step(update_step)
    start, end, warmup, ramp, _ = _validate_config(config)

    scale, progress, phase = _absolute_scale(
        step,
        start=start,
        end=end,
        warmup=warmup,
        ramp=ramp,
    )

    if step == 0:
        previous_scale = 0.0
    else:
        previous_scale, _, _ = _absolute_scale(
            step - 1,
            start=start,
            end=end,
            warmup=warmup,
            ramp=ramp,
        )

    delta_scale = max(0.0, scale - previous_scale)
    return RiskPACTCurriculumState(
        update_step=step,
        scale=float(scale),
        previous_scale=float(previous_scale),
        delta_scale=float(delta_scale),
        progress=float(progress),
        phase=phase,
    )


def risk_pact_curriculum_scale(
    update_step: int,
    config: RiskPACTCurriculumConfigLike,
) -> float:
    """Convenience wrapper returning only the absolute PACT-lite scale."""

    return risk_pact_curriculum_state(update_step, config).scale


__all__ = [
    "RiskPACTCurriculumConfigLike",
    "RiskPACTCurriculumState",
    "risk_pact_curriculum_state",
    "risk_pact_curriculum_scale",
]
