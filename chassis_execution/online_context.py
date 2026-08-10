"""Strict MetaDrive-to-surrogate context adapter for CF-7 diagnostics.

MetaDrive is planar, so roll and roll-rate are explicitly represented as zero;
they are not claimed as measured TruckSim signals.  Vehicle conditions remain
checkpoint-bound inputs supplied by the CF-3/CF-4 provenance chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .contracts import NUM_ROLES, VEHICLE_CONDITION_FIELDS
from .grpo_adapter import ChassisExecutionContext, ChassisFusionGRPOError


class OnlineChassisContextError(RuntimeError):
    """Raised when an online MetaDrive state cannot satisfy the surrogate input."""


def _wrap(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _finite_vector(value: object, *, name: str, minimum_size: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < minimum_size or not np.isfinite(array[:minimum_size]).all():
        raise OnlineChassisContextError(f"{name} must contain finite XY values")
    return array


@dataclass(frozen=True)
class OnlineContextDiagnostics:
    timestamp_s: float
    planar_roll_assumption: bool
    vehicle_condition_source: str
    acceleration_from_history: tuple[bool, bool, bool]


class MetaDriveOnlineChassisContextBuilder:
    """Build the exact CF-0 context from the current three-agent MetaDrive state."""

    def __init__(
        self,
        vehicle_condition: Tensor,
        *,
        vehicle_condition_source: str,
        agent_ids: tuple[str, str, str] = ("agent0", "agent1", "agent2"),
    ) -> None:
        if (
            not isinstance(vehicle_condition, Tensor)
            or vehicle_condition.dtype != torch.float32
            or tuple(vehicle_condition.shape)
            != (NUM_ROLES, len(VEHICLE_CONDITION_FIELDS))
            or not bool(torch.isfinite(vehicle_condition).all())
            or vehicle_condition.requires_grad
        ):
            raise OnlineChassisContextError(
                "vehicle_condition must be detached finite float32 [3,14]"
            )
        if tuple(agent_ids) != ("agent0", "agent1", "agent2"):
            raise OnlineChassisContextError(
                "online context requires joint-first agent0/agent1/agent2"
            )
        if not isinstance(vehicle_condition_source, str) or not vehicle_condition_source:
            raise OnlineChassisContextError("vehicle_condition_source is required")
        self.vehicle_condition = vehicle_condition.detach().cpu().clone()
        self.vehicle_condition_source = vehicle_condition_source
        self.agent_ids = tuple(agent_ids)
        self._previous: dict[str, tuple[float, np.ndarray, float]] = {}
        self._last_timestamp_s: float | None = None
        self.last_diagnostics: OnlineContextDiagnostics | None = None

    def reset(self) -> None:
        self._previous.clear()
        self._last_timestamp_s = None
        self.last_diagnostics = None

    @staticmethod
    def _speed_components(vehicle: object) -> tuple[np.ndarray, float, float]:
        velocity = _finite_vector(
            getattr(vehicle, "velocity", ()), name="vehicle.velocity"
        )[:2]
        heading = float(getattr(vehicle, "heading_theta", float("nan")))
        if not math.isfinite(heading):
            raise OnlineChassisContextError("vehicle heading must be finite")
        forward = np.asarray([math.cos(heading), math.sin(heading)])
        left = np.asarray([-math.sin(heading), math.cos(heading)])
        return velocity, float(np.dot(velocity, forward)), float(np.dot(velocity, left))

    @staticmethod
    def _lane_heading_error(vehicle: object) -> float:
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            raise OnlineChassisContextError("vehicle lane is unavailable")
        position = _finite_vector(
            getattr(vehicle, "position", ()), name="vehicle.position"
        )[:2]
        try:
            longitudinal, _ = lane.local_coordinates(position)
            lane_heading = float(lane.heading_theta_at(float(longitudinal)))
        except (AttributeError, TypeError, ValueError) as exc:
            raise OnlineChassisContextError("lane heading is unavailable") from exc
        heading = float(getattr(vehicle, "heading_theta", float("nan")))
        return _wrap(heading - lane_heading)

    @staticmethod
    def _controller_state(env: object, agent_id: str) -> tuple[float, float, float, float]:
        longitudinal = getattr(env, "_trajectory_longitudinal_controller", None)
        integral_map = getattr(longitudinal, "_integral", {}) if longitudinal is not None else {}
        speed_integral = float(integral_map.get(agent_id, 0.0))
        lateral_map = getattr(env, "_lateral_preview_pid_state", {}) or {}
        lateral_integral, previous_lateral, _ = lateral_map.get(
            agent_id, (0.0, 0.0, False)
        )
        debug = (getattr(env, "_last_longitudinal_control_debug", {}) or {}).get(
            agent_id, {}
        )
        arc_error = float(debug.get("original_arc_error_m", 0.0))
        values = (speed_integral, float(previous_lateral), float(lateral_integral), arc_error)
        if not np.isfinite(values).all():
            raise OnlineChassisContextError("controller state contains non-finite values")
        return values

    def build(
        self, env: object, timestamp_s: float
    ) -> tuple[ChassisExecutionContext, OnlineContextDiagnostics]:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise OnlineChassisContextError("timestamp_s must be finite and non-negative")
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise OnlineChassisContextError("online context timestamps must increase strictly")
        agents = getattr(env, "agents", {}) or {}
        if any(agent_id not in agents for agent_id in self.agent_ids):
            raise OnlineChassisContextError("online context requires all three active agents")
        initial = np.zeros((NUM_ROLES, 10), dtype=np.float32)
        controller = np.zeros((NUM_ROLES, 8), dtype=np.float32)
        modes = np.zeros((NUM_ROLES,), dtype=np.int64)
        roles = np.arange(NUM_ROLES, dtype=np.int64)
        derivative_valid: list[bool] = []
        kinematics: dict[str, tuple[np.ndarray, float, float, float]] = {}
        for agent_id in self.agent_ids:
            vehicle = agents[agent_id]
            velocity, longitudinal_speed, lateral_speed = self._speed_components(vehicle)
            heading = float(vehicle.heading_theta)
            kinematics[agent_id] = (velocity, longitudinal_speed, lateral_speed, heading)

        formation_getter = getattr(env, "trajectory_formation_constraint_enabled", None)
        formation_enabled = bool(formation_getter()) if callable(formation_getter) else True
        modes[:] = 1 if formation_enabled else 0
        for role, agent_id in enumerate(self.agent_ids):
            vehicle = agents[agent_id]
            velocity, longitudinal_speed, lateral_speed, heading = kinematics[agent_id]
            previous = self._previous.get(agent_id)
            long_accel = lateral_accel = yaw_rate = 0.0
            valid = previous is not None
            if previous is not None:
                previous_time, previous_velocity, previous_heading = previous
                elapsed = timestamp - previous_time
                if elapsed <= 0.0:
                    raise OnlineChassisContextError("derivative interval must be positive")
                acceleration_world = (velocity - previous_velocity) / elapsed
                forward = np.asarray([math.cos(heading), math.sin(heading)])
                left = np.asarray([-math.sin(heading), math.cos(heading)])
                long_accel = float(np.dot(acceleration_world, forward))
                lateral_accel = float(np.dot(acceleration_world, left))
                yaw_rate = _wrap(heading - previous_heading) / elapsed
            derivative_valid.append(valid)
            steering = float(getattr(vehicle, "steering", 0.0) or 0.0)
            max_steering_deg = float(getattr(vehicle, "max_steering", float("nan")))
            throttle_brake = float(getattr(vehicle, "throttle_brake", 0.0) or 0.0)
            tire_radius = float(getattr(vehicle, "TIRE_RADIUS", float("nan")))
            if (
                not np.isfinite(
                    [longitudinal_speed, lateral_speed, long_accel, lateral_accel,
                     yaw_rate, steering, max_steering_deg, throttle_brake, tire_radius]
                ).all()
                or max_steering_deg <= 0.0
                or tire_radius <= 0.0
            ):
                raise OnlineChassisContextError("vehicle dynamic state is invalid")
            initial[role] = np.asarray(
                [
                    max(longitudinal_speed, 0.0),
                    lateral_speed,
                    long_accel,
                    lateral_accel,
                    yaw_rate,
                    0.0,
                    0.0,
                    steering * math.radians(max_steering_deg),
                    float(np.clip(throttle_brake, -1.0, 1.0)),
                    max(longitudinal_speed, 0.0) / tire_radius,
                ],
                dtype=np.float32,
            )
            speed_integral, previous_lateral, lateral_integral, arc_error = (
                self._controller_state(env, agent_id)
            )
            heading_error = self._lane_heading_error(vehicle)
            actual_gap = desired_gap = relative_speed = 0.0
            if role > 0:
                predecessor_id = self.agent_ids[role - 1]
                predecessor = agents[predecessor_id]
                ego_position = _finite_vector(vehicle.position, name="ego.position")[:2]
                front_position = _finite_vector(
                    predecessor.position, name="predecessor.position"
                )[:2]
                forward = np.asarray([math.cos(heading), math.sin(heading)])
                actual_gap = float(np.dot(front_position - ego_position, forward))
                spacing = getattr(env, "_desired_center_spacing_m", None)
                if not callable(spacing):
                    raise OnlineChassisContextError("desired platoon spacing is unavailable")
                desired_gap = float(spacing(agent_id, predecessor_id))
                relative_speed = float(
                    kinematics[predecessor_id][1] - longitudinal_speed
                )
            controller[role] = np.asarray(
                [
                    speed_integral,
                    previous_lateral,
                    lateral_integral,
                    heading_error,
                    arc_error,
                    actual_gap,
                    desired_gap,
                    relative_speed,
                ],
                dtype=np.float32,
            )
            self._previous[agent_id] = (timestamp, velocity.copy(), heading)
        if not np.isfinite(initial).all() or not np.isfinite(controller).all():
            raise OnlineChassisContextError("constructed online context is non-finite")
        self._last_timestamp_s = timestamp
        try:
            result = ChassisExecutionContext(
                initial_state=torch.from_numpy(initial).unsqueeze(0),
                vehicle_condition=self.vehicle_condition.unsqueeze(0),
                controller_context=torch.from_numpy(controller).unsqueeze(0),
                controller_mode=torch.from_numpy(modes).unsqueeze(0),
                agent_role=torch.from_numpy(roles).unsqueeze(0),
            )
        except ChassisFusionGRPOError as exc:
            raise OnlineChassisContextError("constructed context violates CF-0") from exc
        diagnostics = OnlineContextDiagnostics(
            timestamp_s=timestamp,
            planar_roll_assumption=True,
            vehicle_condition_source=self.vehicle_condition_source,
            acceleration_from_history=tuple(derivative_valid),
        )
        self.last_diagnostics = diagnostics
        return result, diagnostics


__all__ = [
    "MetaDriveOnlineChassisContextBuilder",
    "OnlineChassisContextError",
    "OnlineContextDiagnostics",
]
