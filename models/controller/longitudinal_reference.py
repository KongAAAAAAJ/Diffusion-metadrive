"""Explicit longitudinal references and feedback-executable arc profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MAX_SPEED_MPS = 100.0 / 3.6
MIN_ACCEL_MPS2 = -8.0
MAX_ACCEL_MPS2 = 5.0

ACCELERATION_BIAS_MPS2 = 0.0
# Pooled through-origin fit over natural zero-start warmups (seeds 17/31/47,
# all three XL vehicles).  This is actuator gain; it is intentionally distinct
# from EXECUTABLE_MAX_ACCEL_MPS2, which remains the requested-acceleration cap.
DRIVE_ACCELERATION_SCALE_MPS2 = 1.4999988847662202
# Fixed-command calibration on the sensorless XL vehicle, from a natural
# zero-speed start at 8 m/s.  Negative MetaDrive throttle is a brake fraction,
# not a normalized request over EXECUTABLE_MIN_ACCEL_MPS2.
BRAKE_ACCELERATION_SCALE_MPS2 = 9.374925668233
# The earlier +0.4/-2.6 authority cap came from a non-zero-speed transient.
# The accepted zero-start drive fit and fixed-command brake fit supersede it.
# Keep authority within the unchanged hard trajectory contract.
EXECUTABLE_MIN_ACCEL_MPS2 = max(
    MIN_ACCEL_MPS2, -BRAKE_ACCELERATION_SCALE_MPS2
)
EXECUTABLE_MAX_ACCEL_MPS2 = min(
    MAX_ACCEL_MPS2, DRIVE_ACCELERATION_SCALE_MPS2
)
# Generated references are later represented by float32 XY/heading and
# independently reconstructed.  Stay conservatively inside the hard boundary
# so representation error cannot turn -8.0 into an invalid -8.00004.
PROFILE_ACCELERATION_GUARD_MPS2 = 1.0e-3


class LongitudinalReferenceError(ValueError):
    """Raised when an explicit longitudinal reference violates its contract."""


class LongitudinalCascadeController:
    """Previewed acceleration feedforward with asymmetric speed PID control."""

    def __init__(
        self,
        *,
        dt_s: float = 0.1,
        acceleration_speed_kp: float = 0.45,
        acceleration_speed_ki: float = 0.08,
        braking_speed_kp: float = 0.30,
        braking_speed_ki: float = 0.04,
        position_kp: float = 0.15,
        position_term_limit_mps2: float = 0.5,
        integral_limit: float = 2.0,
        actuator_delay_s: float = 0.30,
        stop_release_speed_mps: float = 0.30,
        acceleration_bias_mps2: float = ACCELERATION_BIAS_MPS2,
        drive_acceleration_scale_mps2: float = DRIVE_ACCELERATION_SCALE_MPS2,
        brake_acceleration_scale_mps2: float = BRAKE_ACCELERATION_SCALE_MPS2,
    ) -> None:
        values = np.asarray(
            [
                acceleration_speed_kp,
                acceleration_speed_ki,
                braking_speed_kp,
                braking_speed_ki,
                position_kp,
                position_term_limit_mps2,
                integral_limit,
                actuator_delay_s,
                stop_release_speed_mps,
                drive_acceleration_scale_mps2,
                brake_acceleration_scale_mps2,
                acceleration_bias_mps2,
            ],
            dtype=np.float64,
        )
        if (
            dt_s <= 0.0
            or not np.isfinite(values).all()
            or np.any(values[:-1] < 0.0)
            or integral_limit <= 0.0
            or actuator_delay_s < 0.0
            or drive_acceleration_scale_mps2 <= 0.0
            or brake_acceleration_scale_mps2 <= 0.0
        ):
            raise LongitudinalReferenceError("cascade timing and limits must be positive")
        self.dt_s = float(dt_s)
        self.acceleration_speed_pid = (
            float(acceleration_speed_kp),
            float(acceleration_speed_ki),
        )
        self.braking_speed_pid = (
            float(braking_speed_kp),
            float(braking_speed_ki),
        )
        self.position_kp = float(position_kp)
        self.position_term_limit_mps2 = float(position_term_limit_mps2)
        self.integral_limit = float(integral_limit)
        self.actuator_delay_s = float(actuator_delay_s)
        self.stop_release_speed_mps = float(stop_release_speed_mps)
        self.acceleration_bias_mps2 = float(acceleration_bias_mps2)
        self.drive_acceleration_scale_mps2 = float(
            drive_acceleration_scale_mps2
        )
        self.brake_acceleration_scale_mps2 = float(
            brake_acceleration_scale_mps2
        )
        self._integral: dict[str, float] = {}
        self._control_regime: dict[str, str] = {}

    def reset(self) -> None:
        self._integral.clear()
        self._control_regime.clear()

    def compute(
        self,
        agent_id: str,
        current_speed_mps: float,
        reference: "LongitudinalTrackingReference",
        *,
        gap_acceleration_mps2: float = 0.0,
    ) -> tuple[float, dict[str, float | bool | str]]:
        speed = float(current_speed_mps)
        gap = float(np.clip(gap_acceleration_mps2, -1.0, 1.0))
        if not np.isfinite(speed) or not np.isfinite(gap):
            raise LongitudinalReferenceError("cascade inputs must be finite")
        # ``stop_requested`` describes the terminal state of the four-second
        # profile.  It must not erase the time-parameterized deceleration that
        # precedes the stop.  The immediate reference already reaches exactly
        # zero when the vehicle is supposed to be stationary.
        target_speed, preview_acceleration = reference.sample_speed_acceleration_at_time(
            self.actuator_delay_s
        )
        # A terminal STOP is an individual motion contract.  Formation-gap
        # recovery must never cancel its braking profile or restart a vehicle
        # after it has nearly stopped.
        if reference.stop_requested:
            gap = 0.0
        speed_error = float(target_speed - speed)
        feedforward = float(preview_acceleration)
        regime = "braking" if feedforward < -1.0e-6 or speed_error < 0.0 else "acceleration"
        gains = (
            self.braking_speed_pid
            if regime == "braking"
            else self.acceleration_speed_pid
        )
        key = str(agent_id)
        if self._control_regime.get(key) != regime:
            self._integral[key] = 0.0
        previous_integral = float(self._integral.get(str(agent_id), 0.0))
        proposed_integral = float(
            np.clip(
                previous_integral + speed_error * self.dt_s,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        position_term = float(
            np.clip(
                self.position_kp * reference.original_arc_error_m,
                -self.position_term_limit_mps2,
                self.position_term_limit_mps2,
            )
        )
        raw_acceleration = (
            feedforward
            + gains[0] * speed_error
            + gains[1] * proposed_integral
            + position_term
            + gap
        )
        desired_acceleration = float(
            np.clip(
                raw_acceleration,
                EXECUTABLE_MIN_ACCEL_MPS2,
                EXECUTABLE_MAX_ACCEL_MPS2,
            )
        )
        if reference.stop_requested:
            desired_acceleration = min(desired_acceleration, 0.0)
        saturated = not np.isclose(raw_acceleration, desired_acceleration, atol=1.0e-9)
        if not saturated:
            self._integral[key] = proposed_integral
        self._control_regime[key] = regime
        overzero_guard = False
        if speed < -1.0e-3:
            # A negative chassis speed is already a contract violation.  Do
            # not invent an uncalibrated recovery throttle near a STOP point.
            desired_acceleration = 0.0
            overzero_guard = True
            self._integral[key] = 0.0
        elif (
            reference.stop_requested
            and target_speed <= self.stop_release_speed_mps
            and speed <= self.stop_release_speed_mps
        ):
            desired_acceleration = 0.0
            overzero_guard = True
            self._integral[key] = 0.0
        compensated_acceleration = float(
            desired_acceleration - self.acceleration_bias_mps2
        )
        scale = (
            self.drive_acceleration_scale_mps2
            if compensated_acceleration >= 0.0
            else self.brake_acceleration_scale_mps2
        )
        throttle = float(np.clip(compensated_acceleration / scale, -1.0, 1.0))
        return throttle, {
            "longitudinal_reference_source": reference.source,
            "reference_speed_mps": float(target_speed),
            "reference_acceleration_mps2": feedforward,
            "original_arc_error_m": float(reference.original_arc_error_m),
            "speed_error_mps": speed_error,
            "signed_current_speed_mps": speed,
            "actuator_delay_s": self.actuator_delay_s,
            "control_regime": regime,
            "speed_integral": float(self._integral.get(str(agent_id), previous_integral)),
            "position_feedback_mps2": position_term,
            "gap_feedback_mps2": gap,
            "desired_acceleration_mps2": desired_acceleration,
            "compensated_acceleration_mps2": compensated_acceleration,
            "acceleration_bias_mps2": self.acceleration_bias_mps2,
            "drive_acceleration_scale_mps2": (
                self.drive_acceleration_scale_mps2
            ),
            "brake_acceleration_scale_mps2": (
                self.brake_acceleration_scale_mps2
            ),
            "raw_desired_acceleration_mps2": float(raw_acceleration),
            "control_saturated": bool(saturated or abs(throttle) >= 0.999),
            "normalized_throttle": throttle,
            "speed_overzero_guard": overzero_guard,
        }


def _readonly_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (9,) or not np.isfinite(result).all():
        raise LongitudinalReferenceError(f"{name} must be finite float64 [9]")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LongitudinalTrackingReference:
    """Four-second longitudinal profile independent from lateral tracking."""

    sample_times_s: np.ndarray
    arc_position_m: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    original_arc_error_m: float
    stop_requested: bool
    source: str

    def __post_init__(self) -> None:
        times = _readonly_vector(self.sample_times_s, "sample_times_s")
        arc = _readonly_vector(self.arc_position_m, "arc_position_m")
        speed = _readonly_vector(self.speed_mps, "speed_mps")
        acceleration = _readonly_vector(
            self.acceleration_mps2, "acceleration_mps2"
        )
        if not np.allclose(times, np.arange(9, dtype=np.float64) * 0.5):
            raise LongitudinalReferenceError(
                "sample_times_s must be exactly [0, 0.5, ..., 4.0]"
            )
        if np.any(np.diff(arc) < -1.0e-8):
            raise LongitudinalReferenceError("arc_position_m must be monotonic")
        if np.any(speed < -1.0e-8) or np.any(speed > MAX_SPEED_MPS + 1.0e-6):
            raise LongitudinalReferenceError("speed_mps violates [0, 100 km/h]")
        if (
            np.any(acceleration < MIN_ACCEL_MPS2 - 1.0e-6)
            or np.any(acceleration > MAX_ACCEL_MPS2 + 1.0e-6)
        ):
            raise LongitudinalReferenceError(
                "acceleration_mps2 violates the hard [-8, 5] contract"
            )
        arc_error = float(self.original_arc_error_m)
        if not np.isfinite(arc_error):
            raise LongitudinalReferenceError("original_arc_error_m must be finite")
        source = str(self.source)
        if not source:
            raise LongitudinalReferenceError("source must be non-empty")
        if bool(self.stop_requested):
            stopped = np.flatnonzero(speed <= 1.0e-6)
            if stopped.size:
                first = int(stopped[0])
                if not (
                    np.all(speed[first:] <= 1.0e-6)
                    and np.allclose(arc[first:], arc[first], atol=1.0e-8)
                ):
                    raise LongitudinalReferenceError(
                        "STOP profile must remain stationary after stopping"
                    )
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "arc_position_m", arc)
        object.__setattr__(self, "speed_mps", speed)
        object.__setattr__(self, "acceleration_mps2", acceleration)
        object.__setattr__(self, "original_arc_error_m", arc_error)
        object.__setattr__(self, "stop_requested", bool(self.stop_requested))
        object.__setattr__(self, "source", source)

    @property
    def target_speed_mps(self) -> float:
        """Immediate executable target, previewed by one control period."""

        return float(np.interp(0.1, self.sample_times_s, self.speed_mps))

    def sample_speed_acceleration_at_time(
        self, preview_time_s: float
    ) -> tuple[float, float]:
        """Sample the explicit time profile without mixing in position error."""

        preview = float(preview_time_s)
        if not np.isfinite(preview) or preview < 0.0:
            raise LongitudinalReferenceError(
                "preview_time_s must be finite and non-negative"
            )
        query = min(preview, float(self.sample_times_s[-1]))
        return (
            float(np.interp(query, self.sample_times_s, self.speed_mps)),
            float(
                np.interp(query, self.sample_times_s, self.acceleration_mps2)
            ),
        )

    @property
    def feedforward_acceleration_mps2(self) -> float:
        return float(self.acceleration_mps2[0])


def signed_longitudinal_speed_mps(vehicle) -> float:
    """Return velocity projected onto the vehicle's forward heading.

    MetaDrive's ``speed`` and ``speed_km_h`` are magnitudes and therefore hide
    reverse motion.  Strict trajectory control must preserve the sign.
    """

    velocity = np.asarray(getattr(vehicle, "velocity", ()), dtype=np.float64).reshape(-1)
    if velocity.size < 2 or not np.isfinite(velocity[:2]).all():
        raise LongitudinalReferenceError(
            "vehicle.velocity must provide a finite world-frame [vx, vy]"
        )
    heading = getattr(vehicle, "heading", None)
    if heading is None:
        theta = float(getattr(vehicle, "heading_theta", np.nan))
        if not np.isfinite(theta):
            raise LongitudinalReferenceError("vehicle heading must be finite")
        tangent = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
    else:
        tangent = np.asarray(heading, dtype=np.float64).reshape(-1)
        if tangent.size < 2 or not np.isfinite(tangent[:2]).all():
            raise LongitudinalReferenceError("vehicle.heading must be finite [x,y]")
        tangent = tangent[:2]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1.0e-9:
        raise LongitudinalReferenceError("vehicle heading has zero norm")
    return float(np.dot(velocity[:2], tangent / norm))


def trajectory_to_longitudinal_reference(
    trajectory: np.ndarray,
    current_speed_mps: float,
    *,
    source: str = "online_trajectory",
) -> LongitudinalTrackingReference:
    """Build an explicit profile from a current ego-local `[8,3]` trajectory."""

    value = np.asarray(trajectory)
    speed0 = float(current_speed_mps)
    if (
        value.shape != (8, 3)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or not np.isfinite(speed0)
        or speed0 < 0.0
    ):
        raise LongitudinalReferenceError(
            "trajectory must be finite floating [8,3] and speed non-negative"
        )
    xy = np.concatenate(
        (np.zeros((1, 2), dtype=np.float64), value[:, :2].astype(np.float64)),
        axis=0,
    )
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    segment_speed = np.diff(arc) / 0.5
    speeds = np.concatenate(([speed0], segment_speed))
    acceleration = np.empty((9,), dtype=np.float64)
    acceleration[:-1] = np.diff(speeds) / 0.5
    acceleration[-1] = acceleration[-2]
    acceleration = np.clip(acceleration, MIN_ACCEL_MPS2, MAX_ACCEL_MPS2)
    stopped = bool(segment_speed[-1] <= 1.0e-6)
    return LongitudinalTrackingReference(
        sample_times_s=np.arange(9, dtype=np.float64) * 0.5,
        arc_position_m=arc,
        speed_mps=np.clip(speeds, 0.0, MAX_SPEED_MPS),
        acceleration_mps2=acceleration,
        original_arc_error_m=0.0,
        stop_requested=stopped,
        source=source,
    )


def project_point_to_path_arc(
    point_xy: np.ndarray,
    path_xy: np.ndarray,
    path_arc_m: np.ndarray,
) -> tuple[float, float]:
    """Return nearest path arc and Euclidean projection error."""

    point = np.asarray(point_xy, dtype=np.float64)
    path = np.asarray(path_xy, dtype=np.float64)
    arc = np.asarray(path_arc_m, dtype=np.float64)
    if (
        point.shape != (2,)
        or path.ndim != 2
        or path.shape[1] != 2
        or path.shape[0] < 2
        or arc.shape != (path.shape[0],)
        or not np.isfinite(point).all()
        or not np.isfinite(path).all()
        or not np.isfinite(arc).all()
    ):
        raise LongitudinalReferenceError("path projection inputs are invalid")
    best_distance = float("inf")
    best_arc = float(arc[0])
    for index, segment in enumerate(np.diff(path, axis=0)):
        norm2 = float(segment @ segment)
        fraction = (
            0.0
            if norm2 <= 1.0e-12
            else float(np.clip(((point - path[index]) @ segment) / norm2, 0.0, 1.0))
        )
        projection = path[index] + fraction * segment
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_distance = distance
            best_arc = float(
                arc[index] + fraction * (arc[index + 1] - arc[index])
            )
    return best_arc, best_distance


def sample_path_at_arc(
    path_world: np.ndarray,
    path_arc_m: np.ndarray,
    query_arc_m: np.ndarray,
) -> np.ndarray:
    """Sample a stored spatial path without restarting its lateral geometry."""

    path = np.asarray(path_world, dtype=np.float64)
    arc = np.asarray(path_arc_m, dtype=np.float64)
    query = np.asarray(query_arc_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or arc.shape != (path.shape[0],):
        raise LongitudinalReferenceError("spatial path shape is invalid")
    keep = np.concatenate(([True], np.diff(arc) > 1.0e-9))
    unique_arc = arc[keep]
    unique_path = path[keep]
    if unique_arc.size < 2 or np.any(query < unique_arc[0] - 1.0e-8) or np.any(
        query > unique_arc[-1] + 1.0e-8
    ):
        raise LongitudinalReferenceError("spatial path arc buffer is exhausted")
    heading = np.unwrap(unique_path[:, 2])
    result = np.column_stack(
        (
            np.interp(query, unique_arc, unique_path[:, 0]),
            np.interp(query, unique_arc, unique_path[:, 1]),
            np.interp(query, unique_arc, heading),
        )
    )
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    return result


def build_feedback_executable_profile(
    *,
    path_times_s: np.ndarray,
    path_arc_m: np.ndarray,
    elapsed_s: float,
    actual_arc_m: float,
    actual_speed_mps: float,
    position_gain: float = 0.8,
    speed_gain: float = 1.2,
    internal_dt_s: float = 0.1,
    path_speed_limit_mps: np.ndarray | None = None,
    source: str = "committed_roll",
) -> LongitudinalTrackingReference:
    """Roll a path-relative profile from the actual state under bounded feedback."""

    times = np.asarray(path_times_s, dtype=np.float64)
    arc = np.asarray(path_arc_m, dtype=np.float64)
    elapsed = float(elapsed_s)
    actual_arc = float(actual_arc_m)
    actual_speed = float(actual_speed_mps)
    if (
        times.ndim != 1
        or arc.shape != times.shape
        or times.size < 2
        or not np.isfinite(times).all()
        or not np.isfinite(arc).all()
        or np.any(np.diff(times) <= 0.0)
        or np.any(np.diff(arc) < -1.0e-8)
        or not np.isfinite([elapsed, actual_arc, actual_speed]).all()
        or actual_speed < 0.0
        or internal_dt_s <= 0.0
    ):
        raise LongitudinalReferenceError("feedback profile inputs are invalid")
    if path_speed_limit_mps is None:
        speed_limit = np.full_like(arc, MAX_SPEED_MPS)
    else:
        speed_limit = np.asarray(path_speed_limit_mps, dtype=np.float64)
        if (
            speed_limit.shape != arc.shape
            or not np.isfinite(speed_limit).all()
            or np.any(speed_limit <= 0.0)
        ):
            raise LongitudinalReferenceError("path speed limit is invalid")
    dense_offsets = np.arange(41, dtype=np.float64) * float(internal_dt_s)
    absolute_times = elapsed + dense_offsets
    if absolute_times[-1] > times[-1] + 1.0e-8:
        raise LongitudinalReferenceError("longitudinal reference buffer is exhausted")
    original_arc = np.interp(absolute_times, times, arc)
    segment_speed = np.diff(arc) / np.diff(times)
    speed_times = 0.5 * (times[:-1] + times[1:])
    original_speed = np.interp(
        absolute_times,
        speed_times,
        segment_speed,
        left=segment_speed[0],
        right=segment_speed[-1],
    )
    segment_acceleration = np.gradient(segment_speed, speed_times)
    original_acceleration = np.interp(
        absolute_times,
        speed_times,
        segment_acceleration,
        left=segment_acceleration[0],
        right=segment_acceleration[-1],
    )
    executed_arc = np.empty_like(dense_offsets)
    executed_speed = np.empty_like(dense_offsets)
    executed_acceleration = np.empty_like(dense_offsets)
    executed_arc[0] = actual_arc
    executed_speed[0] = min(actual_speed, MAX_SPEED_MPS)
    for index in range(dense_offsets.size - 1):
        preview_end = min(
            executed_arc[index] + max(executed_speed[index], 3.0), arc[-1]
        )
        preview_mask = (arc >= executed_arc[index] - 1.0e-9) & (
            arc <= preview_end + 1.0e-9
        )
        curve_speed_limit = float(
            np.min(speed_limit[preview_mask])
            if np.any(preview_mask)
            else np.interp(executed_arc[index], arc, speed_limit)
        )
        target_speed = min(float(original_speed[index]), curve_speed_limit)
        acceleration = float(
            np.clip(
                original_acceleration[index]
                + float(position_gain) * (original_arc[index] - executed_arc[index])
                + float(speed_gain) * (target_speed - executed_speed[index]),
                EXECUTABLE_MIN_ACCEL_MPS2
                + PROFILE_ACCELERATION_GUARD_MPS2,
                EXECUTABLE_MAX_ACCEL_MPS2
                - PROFILE_ACCELERATION_GUARD_MPS2,
            )
        )
        dt = float(internal_dt_s)
        if acceleration < 0.0 and executed_speed[index] + acceleration * dt < 0.0:
            stop_time = executed_speed[index] / -acceleration
            distance = executed_speed[index] * stop_time + 0.5 * acceleration * stop_time**2
            next_speed = 0.0
        else:
            distance = executed_speed[index] * dt + 0.5 * acceleration * dt**2
            next_speed = float(
                np.clip(executed_speed[index] + acceleration * dt, 0.0, MAX_SPEED_MPS)
            )
        executed_arc[index + 1] = executed_arc[index] + max(distance, 0.0)
        executed_speed[index + 1] = next_speed
        executed_acceleration[index] = acceleration
    executed_acceleration[-1] = executed_acceleration[-2]
    sparse_indices = np.arange(0, 41, 5, dtype=np.int64)
    sparse_speed = executed_speed[sparse_indices]
    sparse_arc = executed_arc[sparse_indices]
    sparse_acceleration = executed_acceleration[sparse_indices]
    stop_requested = bool(
        original_speed[-1] <= 1.0e-6 and sparse_speed[-1] <= 1.0e-6
    )
    if stop_requested:
        stopped = np.flatnonzero(sparse_speed <= 1.0e-6)
        if stopped.size:
            first = int(stopped[0])
            sparse_speed[first:] = 0.0
            sparse_arc[first:] = sparse_arc[first]
            sparse_acceleration[first:] = 0.0
    return LongitudinalTrackingReference(
        sample_times_s=np.arange(9, dtype=np.float64) * 0.5,
        arc_position_m=sparse_arc,
        speed_mps=sparse_speed,
        acceleration_mps2=sparse_acceleration,
        original_arc_error_m=float(original_arc[0] - actual_arc),
        stop_requested=stop_requested,
        source=source,
    )


__all__ = [
    "ACCELERATION_BIAS_MPS2",
    "BRAKE_ACCELERATION_SCALE_MPS2",
    "DRIVE_ACCELERATION_SCALE_MPS2",
    "EXECUTABLE_MAX_ACCEL_MPS2",
    "EXECUTABLE_MIN_ACCEL_MPS2",
    "PROFILE_ACCELERATION_GUARD_MPS2",
    "LongitudinalReferenceError",
    "LongitudinalCascadeController",
    "LongitudinalTrackingReference",
    "build_feedback_executable_profile",
    "project_point_to_path_arc",
    "sample_path_at_arc",
    "signed_longitudinal_speed_mps",
    "trajectory_to_longitudinal_reference",
]
