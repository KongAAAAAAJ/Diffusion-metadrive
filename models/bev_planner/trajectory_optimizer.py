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

from models.controller.longitudinal_reference import (
    EXECUTABLE_MAX_ACCEL_MPS2,
    EXECUTABLE_MIN_ACCEL_MPS2,
    sample_path_at_arc,
)

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
    # Reserve one third of the calibrated positive actuator authority for
    # speed/position feedback.  A trajectory exactly at the 1.5 m/s2 actuator
    # ceiling is kinematically reachable but cannot reject delay or drag, and
    # was observed to remain at full throttle for entire four-second branches.
    max_accel_mps2: float = min(EXECUTABLE_MAX_ACCEL_MPS2, 1.0)
    max_speed_mps: float = 100.0 / 3.6
    max_yaw_rate_rad_s: float = 1.0
    max_curvature_per_m: float = 0.25
    max_lateral_accel_mps2: float = 6.0
    max_heading_alignment_error_rad: float = 0.15
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
    # Round 13.91b execution contract.  The zero-start actuator calibration
    # found a two-control-step drive response lag, while dedicated fixed brake
    # pulses reached their commanded deceleration within one sample.  These
    # values shape the reference presented to the unchanged cascade
    # controller; they do not alter the controller gains or actuator mapping.
    internal_dt_s: float = 0.1
    drive_time_constant_s: float = 0.2
    brake_time_constant_s: float = 0.0
    max_drive_jerk_mps3: float = 3.0
    max_brake_jerk_mps3: float = 80.0
    profile_speed_weight: float = 0.05
    profile_terminal_arc_weight: float = 20.0
    profile_jerk_regularizations: tuple[float, ...] = (
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    )

    def __post_init__(self) -> None:
        for name in (
            "dt_s",
            "max_speed_mps",
            "max_yaw_rate_rad_s",
            "max_curvature_per_m",
            "max_lateral_accel_mps2",
            "max_heading_alignment_error_rad",
            "raw_residual_limit_m",
            "heading_residual_limit_rad",
            "movement_epsilon_m",
            "internal_dt_s",
            "max_drive_jerk_mps3",
            "max_brake_jerk_mps3",
            "profile_speed_weight",
            "profile_terminal_arc_weight",
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
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in fractions
            )
            or any(first <= second for first, second in zip(fractions, fractions[1:]))
        ):
            raise TrajectoryOptimizationError(
                "line_search_fractions must decrease strictly from <=1 to zero"
            )
        object.__setattr__(self, "line_search_fractions", fractions)
        if not math.isclose(
            self.dt_s / self.internal_dt_s,
            round(self.dt_s / self.internal_dt_s),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise TrajectoryOptimizationError(
                "dt_s must be an integer multiple of internal_dt_s"
            )
        for name in ("drive_time_constant_s", "brake_time_constant_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise TrajectoryOptimizationError(
                    f"{name} must be finite and non-negative"
                )
            object.__setattr__(self, name, value)
        regularizations = tuple(
            float(value) for value in self.profile_jerk_regularizations
        )
        if (
            not regularizations
            or any(
                not math.isfinite(value) or value <= 0.0 for value in regularizations
            )
            or any(
                first >= second
                for first, second in zip(regularizations, regularizations[1:])
            )
        ):
            raise TrajectoryOptimizationError(
                "profile_jerk_regularizations must be finite, positive and increasing"
            )
        object.__setattr__(self, "profile_jerk_regularizations", regularizations)

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
    profile_regularization: np.ndarray
    predicted_max_positive_jerk_mps3: np.ndarray
    predicted_max_brake_jerk_mps3: np.ndarray
    predicted_command_acceleration_min_mps2: np.ndarray
    predicted_command_acceleration_max_mps2: np.ndarray
    terminal_arc_error_m: np.ndarray
    brake_to_drive_transition_s: np.ndarray
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
            raise TrajectoryOptimizationError(
                "trajectory optimization result shape mismatch"
            )
        for name in (
            "raw_valid",
            "optimized_valid",
            "intervention_ade_m",
            "intervention_fde_m",
            "retained_raw_fraction",
            "profile_regularization",
            "predicted_max_positive_jerk_mps3",
            "predicted_max_brake_jerk_mps3",
            "predicted_command_acceleration_min_mps2",
            "predicted_command_acceleration_max_mps2",
            "terminal_arc_error_m",
            "brake_to_drive_transition_s",
        ):
            if np.asarray(getattr(self, name)).shape != expected:
                raise TrajectoryOptimizationError(f"{name} shape mismatch")
        if len(self.raw_violations) != int(np.prod(expected, dtype=np.int64)):
            raise TrajectoryOptimizationError("raw violation count mismatch")
        if not np.isfinite(raw).all() or not np.isfinite(optimized).all():
            raise TrajectoryOptimizationError("optimized trajectories must be finite")
        if not bool(np.asarray(self.optimized_valid, dtype=bool).all()):
            raise TrajectoryOptimizationError(
                "optimizer returned an invalid trajectory"
            )
        for name in ("raw_trajectories", "optimized_trajectories", "selected_modes"):
            value = np.ascontiguousarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class _ActuatorProfileDiagnostics:
    regularization: float
    max_positive_jerk_mps3: float
    max_brake_jerk_mps3: float
    command_acceleration_min_mps2: float
    command_acceleration_max_mps2: float
    terminal_arc_error_m: float
    brake_to_drive_transition_s: float


def _zero_profile_diagnostics() -> _ActuatorProfileDiagnostics:
    return _ActuatorProfileDiagnostics(
        regularization=0.0,
        max_positive_jerk_mps3=0.0,
        max_brake_jerk_mps3=0.0,
        command_acceleration_min_mps2=0.0,
        command_acceleration_max_mps2=0.0,
        terminal_arc_error_m=0.0,
        brake_to_drive_transition_s=0.0,
    )


class KinematicTrajectoryOptimizer:
    """Project selected raw trajectories without changing their policy mode."""

    def __init__(
        self, config: KinematicTrajectoryOptimizerConfig | None = None
    ) -> None:
        self.config = config or KinematicTrajectoryOptimizerConfig()
        self._source_audit_config = HardModeMaskConfig(
            dt_s=self.config.dt_s,
            max_speed_mps=self.config.max_speed_mps,
            min_accel_mps2=-8.0,
            max_accel_mps2=5.0,
            max_yaw_rate_rad_s=self.config.max_yaw_rate_rad_s,
            max_curvature_per_m=self.config.max_curvature_per_m,
            max_lateral_accel_mps2=self.config.max_lateral_accel_mps2,
            movement_epsilon_m=self.config.movement_epsilon_m,
        )
        self._audit_config = HardModeMaskConfig(
            dt_s=self.config.dt_s,
            max_speed_mps=self.config.max_speed_mps,
            min_accel_mps2=self.config.min_accel_mps2,
            max_accel_mps2=self.config.max_accel_mps2,
            max_yaw_rate_rad_s=self.config.max_yaw_rate_rad_s,
            max_curvature_per_m=self.config.max_curvature_per_m,
            max_lateral_accel_mps2=self.config.max_lateral_accel_mps2,
            max_heading_alignment_error_rad=(
                self.config.max_heading_alignment_error_rad
            ),
            movement_epsilon_m=self.config.movement_epsilon_m,
        )
        self._prepare_profile_solvers()

    def _prepare_profile_solvers(self) -> None:
        """Cache the config-only four-variable least-squares operators."""

        cfg = self.config
        self._profile_sparse_times = (
            np.arange(1, TRAJECTORY_STEPS + 1, dtype=np.float64) * cfg.dt_s
        )
        horizon_s = float(self._profile_sparse_times[-1])
        self._profile_dense_times = np.arange(
            0.0,
            horizon_s + 0.5 * cfg.internal_dt_s,
            cfg.internal_dt_s,
            dtype=np.float64,
        )
        powers = np.arange(4, dtype=np.float64)
        self._profile_arc_basis = np.column_stack(
            [
                self._profile_sparse_times ** (power + 2.0)
                / ((power + 1.0) * (power + 2.0))
                for power in powers
            ]
        )
        self._profile_speed_basis = np.column_stack(
            [
                self._profile_sparse_times ** (power + 1.0) / (power + 1.0)
                for power in powers
            ]
        )
        self._profile_jerk_basis = np.column_stack(
            (
                np.zeros_like(self._profile_dense_times),
                np.ones_like(self._profile_dense_times),
                2.0 * self._profile_dense_times,
                3.0 * self._profile_dense_times**2,
            )
        )
        self._profile_solvers = tuple(
            (
                regularization,
                np.linalg.pinv(
                    np.vstack(
                        (
                            self._profile_arc_basis,
                            math.sqrt(cfg.profile_speed_weight)
                            * self._profile_speed_basis,
                            math.sqrt(regularization) * self._profile_jerk_basis,
                            math.sqrt(cfg.profile_terminal_arc_weight)
                            * self._profile_arc_basis[-1:],
                        )
                    )
                ),
            )
            for regularization in cfg.profile_jerk_regularizations
        )

    def _feedback_executable_trajectory(
        self,
        candidate: np.ndarray,
        current_speed_mps: float,
        mode: int,
    ) -> tuple[np.ndarray, _ActuatorProfileDiagnostics]:
        """Retain spatial geometry and fit an actuator-executable time profile.

        The fit uses a cubic acceleration polynomial.  Its two integrations
        are linear in the coefficients, so every regularization candidate is
        a deterministic four-variable least-squares problem.  The first fit
        whose executed acceleration, jerk and inverse-lag actuator command are
        all feasible is used.  This removes an abrupt brake-to-drive rebound
        without mutating the raw policy action or the selected mode.
        """

        candidate32 = np.ascontiguousarray(candidate, dtype=np.float32)
        candidate32 = self._authority_limited_baseline(candidate32, current_speed_mps)
        baseline_audit = validate_trajectory_kinematics(
            candidate32,
            current_speed_mps,
            np.zeros(3, dtype=np.float64),
            self._audit_config,
        )
        if not baseline_audit.valid:
            raise TrajectoryOptimizationError(
                "actuator-authority baseline is invalid: "
                + ",".join(baseline_audit.violations)
            )
        sparse_acceleration = np.asarray(
            baseline_audit.acceleration_mps2, dtype=np.float64
        )
        sparse_jerk = np.diff(sparse_acceleration) / self.config.dt_s
        brake_release = (sparse_acceleration[:-1] < -0.25) & (
            sparse_jerk > self.config.max_drive_jerk_mps3
        )
        if int(mode) == int(ModeIndex.STOP) or not bool(np.any(brake_release)):
            return candidate32, _ActuatorProfileDiagnostics(
                regularization=0.0,
                max_positive_jerk_mps3=float(
                    max(np.max(sparse_jerk), 0.0) if sparse_jerk.size else 0.0
                ),
                max_brake_jerk_mps3=float(
                    max(-np.min(sparse_jerk), 0.0) if sparse_jerk.size else 0.0
                ),
                command_acceleration_min_mps2=float(np.min(sparse_acceleration)),
                command_acceleration_max_mps2=float(np.max(sparse_acceleration)),
                terminal_arc_error_m=0.0,
                brake_to_drive_transition_s=0.0,
            )
        path = np.concatenate(
            (np.zeros((1, 3), dtype=np.float64), candidate32.astype(np.float64)),
            axis=0,
        )
        path_arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        )
        if path_arc[-1] <= self.config.movement_epsilon_m:
            return candidate32, _zero_profile_diagnostics()
        cfg = self.config
        sparse_times = self._profile_sparse_times
        horizon_s = float(sparse_times[-1])
        dense_times = self._profile_dense_times
        arc_basis = self._profile_arc_basis
        jerk_basis = self._profile_jerk_basis
        target_speed = np.diff(path_arc) / cfg.dt_s
        base_arc = float(current_speed_mps) * sparse_times
        base_speed = np.full(TRAJECTORY_STEPS, float(current_speed_mps))

        for regularization, solver in self._profile_solvers:
            target = np.concatenate(
                (
                    path_arc[1:] - base_arc,
                    math.sqrt(cfg.profile_speed_weight) * (target_speed - base_speed),
                    np.zeros(dense_times.size, dtype=np.float64),
                    np.asarray(
                        [
                            math.sqrt(cfg.profile_terminal_arc_weight)
                            * (path_arc[-1] - base_arc[-1])
                        ],
                        dtype=np.float64,
                    ),
                )
            )
            coefficients = solver @ target
            acceleration = sum(
                coefficients[index] * dense_times**index for index in range(4)
            )
            jerk = jerk_basis @ coefficients
            dense_arc = float(current_speed_mps) * dense_times + sum(
                coefficients[index]
                * dense_times ** (index + 2)
                / ((index + 1) * (index + 2))
                for index in range(4)
            )
            # A uniform acceleration offset is the unique minimum-jerk
            # correction that restores the selected path's terminal arc.
            terminal_correction = 2.0 * (path_arc[-1] - dense_arc[-1]) / horizon_s**2
            acceleration = acceleration + terminal_correction
            dense_arc = dense_arc + 0.5 * terminal_correction * dense_times**2
            dense_speed = (
                float(current_speed_mps)
                + sum(
                    coefficients[index] * dense_times ** (index + 1) / (index + 1)
                    for index in range(4)
                )
                + terminal_correction * dense_times
            )
            time_constant = np.where(
                jerk >= 0.0,
                cfg.drive_time_constant_s,
                cfg.brake_time_constant_s,
            )
            actuator_command = acceleration + time_constant * jerk
            initial_jerk = float(acceleration[0]) / cfg.internal_dt_s
            initial_time_constant = (
                cfg.drive_time_constant_s
                if initial_jerk >= 0.0
                else cfg.brake_time_constant_s
            )
            initial_command = float(acceleration[0]) + (
                initial_time_constant * initial_jerk
            )
            tolerance = 1.0e-7
            if (
                np.min(acceleration) < cfg.min_accel_mps2 - tolerance
                or np.max(acceleration) > cfg.max_accel_mps2 + tolerance
                or np.min(jerk) < -cfg.max_brake_jerk_mps3 - tolerance
                or np.max(jerk) > cfg.max_drive_jerk_mps3 + tolerance
                or np.min(actuator_command) < cfg.min_accel_mps2 - tolerance
                or np.max(actuator_command) > cfg.max_accel_mps2 + tolerance
                or initial_jerk < -cfg.max_brake_jerk_mps3 - tolerance
                or initial_jerk > cfg.max_drive_jerk_mps3 + tolerance
                or initial_command < cfg.min_accel_mps2 - tolerance
                or initial_command > cfg.max_accel_mps2 + tolerance
                or np.min(dense_speed) < -tolerance
                or np.max(dense_speed) > cfg.max_speed_mps + tolerance
                or np.any(np.diff(dense_arc) < -tolerance)
                or np.min(dense_arc) < -tolerance
                or np.max(dense_arc) > path_arc[-1] + tolerance
            ):
                continue
            executable_arc = np.interp(sparse_times, dense_times, dense_arc)
            try:
                sampled = sample_path_at_arc(path, path_arc, executable_arc)
            except ValueError:
                continue
            minimum_index = int(np.argmin(acceleration))
            nonnegative_after = np.flatnonzero(acceleration[minimum_index:] >= -1.0e-6)
            transition = (
                float(nonnegative_after[0]) * cfg.internal_dt_s
                if nonnegative_after.size
                else float(horizon_s - dense_times[minimum_index])
            )
            diagnostics = _ActuatorProfileDiagnostics(
                regularization=float(regularization),
                max_positive_jerk_mps3=float(max(np.max(jerk), 0.0)),
                max_brake_jerk_mps3=float(max(-np.min(jerk), 0.0)),
                command_acceleration_min_mps2=float(np.min(actuator_command)),
                command_acceleration_max_mps2=float(np.max(actuator_command)),
                terminal_arc_error_m=float(executable_arc[-1] - path_arc[-1]),
                brake_to_drive_transition_s=transition,
            )
            return np.ascontiguousarray(sampled, dtype=np.float32), diagnostics

        # Some hard-valid anchors request a terminal distance outside the
        # positive actuator authority, or intentionally stop and remain
        # stationary.  A cubic fit cannot represent those active constraints
        # without oscillation.  Use the unique monotone endpoint profile:
        # lagged constant drive command, constant braking, or brake-to-stop.
        dt = cfg.internal_dt_s
        dense_arc = np.zeros_like(dense_times)
        dense_speed = np.zeros_like(dense_times)
        dense_acceleration = np.zeros_like(dense_times)
        dense_speed[0] = float(current_speed_mps)
        constant_speed_arc = float(current_speed_mps) * horizon_s
        target_terminal_arc = float(path_arc[-1])
        actuator_command_value = 0.0
        stopped_index: int | None = None
        if target_terminal_arc >= constant_speed_arc - 1.0e-9:

            def drive_profile(
                command: float,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                arc_value = np.zeros_like(dense_times)
                speed_value = np.zeros_like(dense_times)
                acceleration_value = np.zeros_like(dense_times)
                speed_value[0] = float(current_speed_mps)
                alpha = (
                    1.0
                    if cfg.drive_time_constant_s == 0.0
                    else dt / (cfg.drive_time_constant_s + dt)
                )
                for index in range(dense_times.size - 1):
                    next_acceleration = acceleration_value[index] + alpha * (
                        command - acceleration_value[index]
                    )
                    next_speed = min(
                        cfg.max_speed_mps,
                        speed_value[index]
                        + 0.5 * (acceleration_value[index] + next_acceleration) * dt,
                    )
                    arc_value[index + 1] = (
                        arc_value[index] + 0.5 * (speed_value[index] + next_speed) * dt
                    )
                    speed_value[index + 1] = next_speed
                    acceleration_value[index + 1] = next_acceleration
                return arc_value, speed_value, acceleration_value

            maximum_command = min(
                cfg.max_accel_mps2,
                cfg.max_drive_jerk_mps3
                * dt
                * (cfg.drive_time_constant_s + dt)
                / max(dt, 1.0e-12),
            )
            maximum_profile = drive_profile(maximum_command)
            reachable_terminal = float(maximum_profile[0][-1])
            requested_terminal = min(target_terminal_arc, reachable_terminal)
            reachable_increment = reachable_terminal - constant_speed_arc
            requested_increment = requested_terminal - constant_speed_arc
            actuator_command_value = (
                0.0
                if reachable_increment <= 1.0e-10
                else float(maximum_command)
                * float(np.clip(requested_increment / reachable_increment, 0.0, 1.0))
            )
            dense_arc, dense_speed, dense_acceleration = drive_profile(
                actuator_command_value
            )
        else:
            half_constant_arc = 0.5 * float(current_speed_mps) * horizon_s
            if target_terminal_arc >= half_constant_arc - 1.0e-9:
                actuator_command_value = (
                    2.0 * (target_terminal_arc - constant_speed_arc) / horizon_s**2
                )
                if actuator_command_value < cfg.min_accel_mps2 - 1.0e-7:
                    raise TrajectoryOptimizationError(
                        "monotone braking profile exceeds actuator authority"
                    )
                dense_acceleration.fill(actuator_command_value)
                dense_speed = np.maximum(
                    0.0,
                    float(current_speed_mps) + actuator_command_value * dense_times,
                )
                dense_arc = (
                    float(current_speed_mps) * dense_times
                    + 0.5 * actuator_command_value * dense_times**2
                )
            else:
                if target_terminal_arc <= cfg.movement_epsilon_m:
                    raise TrajectoryOptimizationError(
                        "non-STOP spatial path has zero stopping distance"
                    )
                actuator_command_value = -float(current_speed_mps) ** 2 / (
                    2.0 * target_terminal_arc
                )
                if actuator_command_value < cfg.min_accel_mps2 - 1.0e-7:
                    raise TrajectoryOptimizationError(
                        "selected spatial path is shorter than calibrated stopping distance"
                    )
                stop_time = -float(current_speed_mps) / actuator_command_value
                moving = dense_times < stop_time
                dense_speed[moving] = (
                    float(current_speed_mps)
                    + actuator_command_value * dense_times[moving]
                )
                dense_acceleration[moving] = actuator_command_value
                dense_arc[moving] = (
                    float(current_speed_mps) * dense_times[moving]
                    + 0.5 * actuator_command_value * dense_times[moving] ** 2
                )
                dense_arc[~moving] = target_terminal_arc
                stopped = np.flatnonzero(~moving)
                stopped_index = int(stopped[0]) if stopped.size else None

        if (
            np.any(np.diff(dense_arc) < -1.0e-7)
            or np.min(dense_speed) < -1.0e-7
            or np.max(dense_arc) > path_arc[-1] + 1.0e-7
        ):
            raise TrajectoryOptimizationError(
                "monotone actuator profile violates arc or speed bounds"
            )
        dense_jerk = np.diff(np.concatenate(([0.0], dense_acceleration))) / dt
        # Releasing the brake after v=0 is owned by the over-zero guard and is
        # not a drive transition.  Exclude only that physically stationary
        # edge from the moving-vehicle jerk audit.
        moving_jerk = dense_jerk.copy()
        if stopped_index is not None and stopped_index < moving_jerk.size:
            moving_jerk[stopped_index] = 0.0
        if (
            np.min(moving_jerk) < -cfg.max_brake_jerk_mps3 - 1.0e-7
            or np.max(moving_jerk) > cfg.max_drive_jerk_mps3 + 1.0e-7
        ):
            raise TrajectoryOptimizationError(
                "monotone actuator profile violates the jerk contract"
            )
        executable_arc = np.interp(sparse_times, dense_times, dense_arc)
        try:
            sampled = sample_path_at_arc(path, path_arc, executable_arc)
        except ValueError as exc:
            raise TrajectoryOptimizationError(
                f"monotone actuator profile exhausted the spatial path: {exc}"
            ) from exc
        diagnostics = _ActuatorProfileDiagnostics(
            regularization=float(cfg.profile_jerk_regularizations[-1]),
            max_positive_jerk_mps3=float(max(np.max(moving_jerk), 0.0)),
            max_brake_jerk_mps3=float(max(-np.min(moving_jerk), 0.0)),
            command_acceleration_min_mps2=float(min(actuator_command_value, 0.0)),
            command_acceleration_max_mps2=float(max(actuator_command_value, 0.0)),
            terminal_arc_error_m=float(executable_arc[-1] - path_arc[-1]),
            brake_to_drive_transition_s=0.0,
        )
        return np.ascontiguousarray(sampled, dtype=np.float32), diagnostics

    def _authority_limited_baseline(
        self,
        candidate: np.ndarray,
        current_speed_mps: float,
    ) -> np.ndarray:
        """Round 13.9 projection retained as the non-reversal baseline."""

        candidate32 = np.ascontiguousarray(candidate, dtype=np.float32)
        direct_audit = validate_trajectory_kinematics(
            candidate32,
            current_speed_mps,
            np.zeros(3, dtype=np.float64),
            self._audit_config,
        )
        if direct_audit.valid:
            return candidate32
        path = np.concatenate(
            (np.zeros((1, 3), dtype=np.float64), candidate32.astype(np.float64)),
            axis=0,
        )
        path_arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        )
        if path_arc[-1] <= self.config.movement_epsilon_m:
            return candidate32
        desired_segment_speed = np.diff(path_arc) / self.config.dt_s
        executable_speed = np.empty_like(desired_segment_speed)
        previous_speed = float(current_speed_mps)
        for index, desired_speed in enumerate(desired_segment_speed):
            acceleration_time = self.config.dt_s * (0.5 if index == 0 else 1.0)
            lower = max(
                0.0,
                previous_speed
                + self.config.min_accel_mps2
                * self.config.limit_safety_factor
                * acceleration_time,
            )
            upper = min(
                self.config.max_speed_mps,
                previous_speed
                + self.config.max_accel_mps2
                * self.config.limit_safety_factor
                * acceleration_time,
            )
            executable_speed[index] = np.clip(desired_speed, lower, upper)
            previous_speed = float(executable_speed[index])
        executable_arc = np.cumsum(executable_speed * self.config.dt_s)
        if executable_arc[-1] > path_arc[-1] + 1.0e-8:
            raise TrajectoryOptimizationError(
                "feedback-executable arc exceeds the selected spatial path"
            )
        try:
            sampled = sample_path_at_arc(path, path_arc, executable_arc)
        except ValueError as exc:
            raise TrajectoryOptimizationError(
                f"feedback-executable trajectory construction failed: {exc}"
            ) from exc
        return np.ascontiguousarray(sampled, dtype=np.float32)

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
            raise TrajectoryOptimizationError(
                "current speeds must be finite and non-negative"
            )
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
            coarse_values = np.broadcast_to(
                coarse_values, prefix + coarse_values.shape[-4:]
            )
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
    ) -> tuple[np.ndarray, float, _ActuatorProfileDiagnostics]:
        cfg = self.config
        coarse_audit = validate_trajectory_kinematics(
            coarse,
            current_speed_mps,
            np.zeros(3, dtype=np.float64),
            self._source_audit_config,
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
        fractions = (
            (0.0,) if int(mode) == int(ModeIndex.STOP) else cfg.line_search_fractions
        )
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
            if (
                lane_change
                and float(np.sign(candidate32[-1, 1])) != coarse_lateral_sign
            ):
                continue
            try:
                candidate32, profile = self._feedback_executable_trajectory(
                    candidate32, current_speed_mps, mode
                )
            except TrajectoryOptimizationError:
                continue
            if (
                lane_change
                and float(np.sign(candidate32[-1, 1])) != coarse_lateral_sign
            ):
                continue
            audit = validate_trajectory_kinematics(
                candidate32,
                current_speed_mps,
                np.zeros(3, dtype=np.float64),
                self._audit_config,
            )
            if audit.valid:
                return candidate32, float(fraction), profile
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
        profile_regularization = np.zeros_like(raw_valid, dtype=np.float32)
        maximum_positive_jerk = np.zeros_like(raw_valid, dtype=np.float32)
        maximum_brake_jerk = np.zeros_like(raw_valid, dtype=np.float32)
        command_minimum = np.zeros_like(raw_valid, dtype=np.float32)
        command_maximum = np.zeros_like(raw_valid, dtype=np.float32)
        terminal_arc_error = np.zeros_like(raw_valid, dtype=np.float32)
        transition_duration = np.zeros_like(raw_valid, dtype=np.float32)
        raw_violations: list[tuple[str, ...]] = []
        for group in range(flat_raw.shape[0]):
            for role in range(3):
                mode = int(flat_modes[group, role])
                raw_audit = validate_trajectory_kinematics(
                    flat_raw[group, role],
                    float(flat_speeds[group, role]),
                    np.zeros(3, dtype=np.float64),
                    self._source_audit_config,
                )
                raw_valid[group, role] = raw_audit.valid
                raw_violations.append(raw_audit.violations)
                try:
                    value, retained, profile = self._project_one(
                        flat_raw[group, role],
                        flat_coarse[group, role, mode],
                        float(flat_speeds[group, role]),
                        mode,
                    )
                except TrajectoryOptimizationError as exc:
                    raise TrajectoryOptimizationError(
                        f"trajectory group={group} role={role} mode={mode}: {exc}"
                    ) from exc
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
                profile_regularization[group, role] = profile.regularization
                maximum_positive_jerk[group, role] = profile.max_positive_jerk_mps3
                maximum_brake_jerk[group, role] = profile.max_brake_jerk_mps3
                command_minimum[group, role] = profile.command_acceleration_min_mps2
                command_maximum[group, role] = profile.command_acceleration_max_mps2
                terminal_arc_error[group, role] = profile.terminal_arc_error_m
                transition_duration[group, role] = profile.brake_to_drive_transition_s

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
            profile_regularization=profile_regularization.reshape(result_shape),
            predicted_max_positive_jerk_mps3=maximum_positive_jerk.reshape(
                result_shape
            ),
            predicted_max_brake_jerk_mps3=maximum_brake_jerk.reshape(result_shape),
            predicted_command_acceleration_min_mps2=command_minimum.reshape(
                result_shape
            ),
            predicted_command_acceleration_max_mps2=command_maximum.reshape(
                result_shape
            ),
            terminal_arc_error_m=terminal_arc_error.reshape(result_shape),
            brake_to_drive_transition_s=transition_duration.reshape(result_shape),
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
