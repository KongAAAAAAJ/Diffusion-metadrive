"""Deterministic execution-time projection for selected BEV trajectories.

The diffusion policy and GRPO probability model remain defined on the raw
trajectory. This module is an action transform at the environment boundary:
it preserves the sampled mode while enforcing the production kinematic
contract on the eight fixed-time waypoints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass

import numpy as np

from .mode_contract import (
    HardModeMaskConfig,
    ModeIndex,
    NUM_MODES,
    TRAJECTORY_DIM,
    TRAJECTORY_STEPS,
    validate_trajectory_kinematics,
)


class TrajectoryOptimizationError(RuntimeError):
    """Raised when a selected action cannot satisfy the execution contract."""


@dataclass(frozen=True)
class KinematicTrajectoryOptimizerConfig:
    dt_s: float = 0.5
    min_accel_mps2: float = -8.0
    max_accel_mps2: float = 5.0
    max_speed_mps: float = 100.0 / 3.6
    max_yaw_rate_rad_s: float = 1.0
    max_curvature_per_m: float = 0.25
    max_lateral_accel_mps2: float = 6.0
    raw_residual_limit_m: float = 2.0
    heading_residual_limit_rad: float = 0.2
    line_search_fractions: tuple[float, ...] = (
        1.0,
        0.75,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.0,
    )
    limit_safety_factor: float = 0.98
    movement_epsilon_m: float = 1.0e-3

    def __post_init__(self) -> None:
        for name in (
            "dt_s",
            "max_speed_mps",
            "max_yaw_rate_rad_s",
            "max_curvature_per_m",
            "max_lateral_accel_mps2",
            "raw_residual_limit_m",
            "heading_residual_limit_rad",
            "movement_epsilon_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise TrajectoryOptimizationError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in ("min_accel_mps2", "max_accel_mps2"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise TrajectoryOptimizationError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.min_accel_mps2 >= self.max_accel_mps2:
            raise TrajectoryOptimizationError("acceleration limits are reversed")
        safety = float(self.limit_safety_factor)
        if not math.isfinite(safety) or not 0.0 < safety < 1.0:
            raise TrajectoryOptimizationError(
                "limit_safety_factor must be strictly between zero and one"
            )
        object.__setattr__(self, "limit_safety_factor", safety)
        fractions = tuple(float(value) for value in self.line_search_fractions)
        if (
            not fractions
            or fractions[-1] != 0.0
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions)
            or any(first <= second for first, second in zip(fractions, fractions[1:]))
        ):
            raise TrajectoryOptimizationError(
                "line_search_fractions must decrease strictly from <=1 to zero"
            )
        object.__setattr__(self, "line_search_fractions", fractions)

    def sha256(self) -> str:
        payload = json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrajectoryOptimizationResult:
    raw_trajectories: np.ndarray
    optimized_trajectories: np.ndarray
    selected_modes: np.ndarray
    raw_valid: np.ndarray
    optimized_valid: np.ndarray
    intervention_ade_m: np.ndarray
    intervention_fde_m: np.ndarray
    retained_raw_fraction: np.ndarray
    raw_violations: tuple[tuple[str, ...], ...]
    elapsed_ms: float
    config_sha256: str

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_trajectories)
        optimized = np.asarray(self.optimized_trajectories)
        modes = np.asarray(self.selected_modes)
        expected = raw.shape[:-2]
        if (
            raw.shape[-2:] != (TRAJECTORY_STEPS, TRAJECTORY_DIM)
            or optimized.shape != raw.shape
            or raw.dtype != np.float32
            or optimized.dtype != np.float32
            or modes.shape != expected
            or modes.dtype != np.int64
        ):
            raise TrajectoryOptimizationError("trajectory optimization result shape mismatch")
        for name in (
            "raw_valid",
            "optimized_valid",
            "intervention_ade_m",
            "intervention_fde_m",
            "retained_raw_fraction",
        ):
            if np.asarray(getattr(self, name)).shape != expected:
                raise TrajectoryOptimizationError(f"{name} shape mismatch")
        if len(self.raw_violations) != int(np.prod(expected, dtype=np.int64)):
            raise TrajectoryOptimizationError("raw violation count mismatch")
        if not np.isfinite(raw).all() or not np.isfinite(optimized).all():
            raise TrajectoryOptimizationError("optimized trajectories must be finite")
        if not bool(np.asarray(self.optimized_valid, dtype=bool).all()):
            raise TrajectoryOptimizationError("optimizer returned an invalid trajectory")
        for name in ("raw_trajectories", "optimized_trajectories", "selected_modes"):
            value = np.ascontiguousarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class KinematicTrajectoryOptimizer:
    """Project selected raw trajectories without changing their policy mode."""

    def __init__(self, config: KinematicTrajectoryOptimizerConfig | None = None) -> None:
        self.config = config or KinematicTrajectoryOptimizerConfig()
        self._audit_config = HardModeMaskConfig(
            dt_s=self.config.dt_s,
            max_speed_mps=self.config.max_speed_mps,
            min_accel_mps2=self.config.min_accel_mps2,
            max_accel_mps2=self.config.max_accel_mps2,
            max_yaw_rate_rad_s=self.config.max_yaw_rate_rad_s,
            max_curvature_per_m=self.config.max_curvature_per_m,
            max_lateral_accel_mps2=self.config.max_lateral_accel_mps2,
            movement_epsilon_m=self.config.movement_epsilon_m,
        )

    @staticmethod
    def _validate_inputs(
        raw: np.ndarray,
        coarse: np.ndarray,
        speeds: np.ndarray,
        modes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
        raw_values = np.asarray(raw)
        if (
            raw_values.ndim < 3
            or raw_values.shape[-3:] != (3, TRAJECTORY_STEPS, TRAJECTORY_DIM)
            or not np.issubdtype(raw_values.dtype, np.floating)
            or not np.isfinite(raw_values).all()
        ):
            raise TrajectoryOptimizationError(
                "raw trajectories must be finite floating-point [...,3,8,3]"
            )
        prefix = raw_values.shape[:-3]
        mode_values = np.asarray(modes)
        if mode_values.shape != prefix + (3,) or not np.issubdtype(
            mode_values.dtype, np.integer
        ):
            raise TrajectoryOptimizationError("selected modes must be integer [...,3]")
        if np.any(mode_values < 0) or np.any(mode_values >= NUM_MODES):
            raise TrajectoryOptimizationError("selected mode is outside [0,10)")
        speed_values = np.asarray(speeds)
        try:
            speed_values = np.broadcast_to(speed_values, prefix + (3,))
        except ValueError as exc:
            raise TrajectoryOptimizationError(
                "current speeds must broadcast to [...,3]"
            ) from exc
        if (
            not np.issubdtype(speed_values.dtype, np.floating)
            or not np.isfinite(speed_values).all()
            or np.any(speed_values < 0.0)
        ):
            raise TrajectoryOptimizationError("current speeds must be finite and non-negative")
        coarse_values = np.asarray(coarse)
        if (
            coarse_values.ndim < 4
            or coarse_values.shape[-4:]
            != (3, NUM_MODES, TRAJECTORY_STEPS, TRAJECTORY_DIM)
            or not np.issubdtype(coarse_values.dtype, np.floating)
            or not np.isfinite(coarse_values).all()
        ):
            raise TrajectoryOptimizationError(
                "coarse trajectories must be finite [...,3,10,8,3]"
            )
        coarse_prefix = coarse_values.shape[:-4]
        if coarse_prefix not in ((), prefix):
            raise TrajectoryOptimizationError(
                "coarse trajectory prefix must be empty or match raw groups"
            )
        if coarse_prefix == () and prefix:
            coarse_values = np.broadcast_to(coarse_values, prefix + coarse_values.shape[-4:])
        return (
            np.asarray(raw_values, dtype=np.float32),
            np.asarray(coarse_values, dtype=np.float32),
            np.asarray(speed_values, dtype=np.float64),
            np.asarray(mode_values, dtype=np.int64),
            prefix,
        )

    def _trusted_targets(self, raw: np.ndarray, coarse: np.ndarray) -> np.ndarray:
        residual = raw[:, :2].astype(np.float64) - coarse[:, :2].astype(np.float64)
        norm = np.linalg.norm(residual, axis=1, keepdims=True)
        scale = np.minimum(
            1.0, self.config.raw_residual_limit_m / np.maximum(norm, 1.0e-12)
        )
        return coarse[:, :2].astype(np.float64) + residual * scale

    def _project_one(
        self,
        raw: np.ndarray,
        coarse: np.ndarray,
        current_speed_mps: float,
        mode: int,
    ) -> tuple[np.ndarray, float]:
        cfg = self.config
        coarse_audit = validate_trajectory_kinematics(
            coarse,
            current_speed_mps,
            np.zeros(3, dtype=np.float64),
            self._audit_config,
        )
        if not coarse_audit.valid:
            raise TrajectoryOptimizationError(
                "selected hard-valid coarse trajectory is invalid: "
                + ",".join(coarse_audit.violations)
            )
        trusted_xy = self._trusted_targets(raw, coarse)
        xy_residual = trusted_xy - coarse[:, :2].astype(np.float64)
        heading_residual = np.asarray(
            [_wrap(float(a) - float(b)) for a, b in zip(raw[:, 2], coarse[:, 2])],
            dtype=np.float64,
        )
        heading_residual = np.clip(
            heading_residual,
            -cfg.heading_residual_limit_rad,
            cfg.heading_residual_limit_rad,
        )
        fractions = (0.0,) if int(mode) == int(ModeIndex.STOP) else cfg.line_search_fractions
        coarse_lateral_sign = float(np.sign(coarse[-1, 1]))
        lane_change = int(mode) in (3, 4, 5, 6, 7, 8)
        for fraction in fractions:
            candidate = np.asarray(coarse, dtype=np.float64).copy()
            candidate[:, :2] += float(fraction) * xy_residual
            candidate[:, 2] = np.asarray(
                [
                    _wrap(float(base) + float(fraction) * float(delta))
                    for base, delta in zip(coarse[:, 2], heading_residual)
                ],
                dtype=np.float64,
            )
            candidate32 = np.asarray(candidate, dtype=np.float32)
            if lane_change and float(np.sign(candidate32[-1, 1])) != coarse_lateral_sign:
                continue
            audit = validate_trajectory_kinematics(
                candidate32,
                current_speed_mps,
                np.zeros(3, dtype=np.float64),
                self._audit_config,
            )
            if audit.valid:
                return candidate32, float(fraction)
        raise TrajectoryOptimizationError(
            "line search failed even at the selected hard-valid coarse anchor"
        )

    def optimize(
        self,
        raw_trajectories: np.ndarray,
        coarse_trajectories: np.ndarray,
        current_speeds_mps: np.ndarray,
        selected_modes: np.ndarray,
    ) -> TrajectoryOptimizationResult:
        """Optimize ``[...,3,8,3]`` selected actions, never all ten modes."""

        start = time.perf_counter()
        raw, coarse, speeds, modes, prefix = self._validate_inputs(
            raw_trajectories, coarse_trajectories, current_speeds_mps, selected_modes
        )
        flat_raw = raw.reshape((-1, 3, TRAJECTORY_STEPS, TRAJECTORY_DIM))
        flat_coarse = coarse.reshape(
            (-1, 3, NUM_MODES, TRAJECTORY_STEPS, TRAJECTORY_DIM)
        )
        flat_speeds = speeds.reshape((-1, 3))
        flat_modes = modes.reshape((-1, 3))
        optimized = np.empty_like(flat_raw, dtype=np.float32)
        raw_valid = np.zeros((flat_raw.shape[0], 3), dtype=np.bool_)
        optimized_valid = np.zeros_like(raw_valid)
        retained_fraction = np.zeros_like(raw_valid, dtype=np.float32)
        raw_violations: list[tuple[str, ...]] = []
        for group in range(flat_raw.shape[0]):
            for role in range(3):
                mode = int(flat_modes[group, role])
                raw_audit = validate_trajectory_kinematics(
                    flat_raw[group, role],
                    float(flat_speeds[group, role]),
                    np.zeros(3, dtype=np.float64),
                    self._audit_config,
                )
                raw_valid[group, role] = raw_audit.valid
                raw_violations.append(raw_audit.violations)
                value, retained = self._project_one(
                    flat_raw[group, role],
                    flat_coarse[group, role, mode],
                    float(flat_speeds[group, role]),
                    mode,
                )
                audit = validate_trajectory_kinematics(
                    value,
                    float(flat_speeds[group, role]),
                    np.zeros(3, dtype=np.float64),
                    self._audit_config,
                )
                if not audit.valid:
                    raise TrajectoryOptimizationError(
                        f"optimized trajectory {group}/{role} is invalid: "
                        + ",".join(audit.violations)
                    )
                optimized[group, role] = value
                optimized_valid[group, role] = True
                retained_fraction[group, role] = retained

        optimized = optimized.reshape(raw.shape)
        delta = np.linalg.norm(optimized[..., :2] - raw[..., :2], axis=-1)
        result_shape = prefix + (3,)
        return TrajectoryOptimizationResult(
            raw_trajectories=np.ascontiguousarray(raw, dtype=np.float32),
            optimized_trajectories=np.ascontiguousarray(optimized, dtype=np.float32),
            selected_modes=np.ascontiguousarray(modes, dtype=np.int64),
            raw_valid=raw_valid.reshape(result_shape),
            optimized_valid=optimized_valid.reshape(result_shape),
            intervention_ade_m=np.asarray(delta.mean(axis=-1), dtype=np.float32),
            intervention_fde_m=np.asarray(delta[..., -1], dtype=np.float32),
            retained_raw_fraction=retained_fraction.reshape(result_shape),
            raw_violations=tuple(raw_violations),
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            config_sha256=self.config.sha256(),
        )


__all__ = [
    "KinematicTrajectoryOptimizer",
    "KinematicTrajectoryOptimizerConfig",
    "TrajectoryOptimizationError",
    "TrajectoryOptimizationResult",
]
