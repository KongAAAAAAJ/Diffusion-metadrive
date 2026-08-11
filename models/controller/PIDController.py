"""PID-based trajectory tracking controller for platoon agents."""

from __future__ import annotations

import numpy as np

from models.controller.base_controller import BaseController
from models.controller.controller_helper import save_pid_debug_plot
from models.controller.longitudinal_reference import (
    BRAKE_ACCELERATION_SCALE_MPS2,
    DRIVE_ACCELERATION_SCALE_MPS2,
    LongitudinalCascadeController,
    LongitudinalTrackingReference,
    signed_longitudinal_speed_mps,
    trajectory_to_longitudinal_reference,
)


def _wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _world_trajectory_to_ego_local(vehicle, trajectory_world: np.ndarray) -> np.ndarray:
    trajectory_world = np.asarray(trajectory_world, dtype=np.float32)
    if trajectory_world.ndim != 2 or trajectory_world.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
    heading = float(getattr(vehicle, "heading_theta", 0.0))
    delta = trajectory_world[:, :2] - pos.reshape(1, 2)
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    local_x = cos_h * delta[:, 0] + sin_h * delta[:, 1]
    local_y = -sin_h * delta[:, 0] + cos_h * delta[:, 1]
    if trajectory_world.shape[1] >= 3:
        local_heading = np.asarray(
            [_wrap_to_pi(float(h) - heading) for h in trajectory_world[:, 2]],
            dtype=np.float32,
        )
    else:
        local_heading = np.zeros((trajectory_world.shape[0],), dtype=np.float32)
    return np.stack([local_x, local_y, local_heading], axis=1).astype(np.float32, copy=False)


class PIDTrajectoryController(BaseController):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        decision_dt = float(cfg.get("physics_world_step_size", 0.02)) * int(
            cfg.get("decision_repeat", 5)
        )
        if not np.isfinite(decision_dt) or decision_dt <= 0.0:
            raise ValueError("controller decision timestep must be positive")
        self.lookahead_index = int(cfg.get("pid_lookahead_index", cfg.get("lookahead_index", 2)))
        self.dt = float(cfg.get("pid_dt", decision_dt))
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("pid_dt must be positive")
        self.lateral_kp = float(cfg.get("pid_lateral_kp", 1.6))
        self.lateral_ki = float(cfg.get("pid_lateral_ki", 0.0))
        self.lateral_kd = float(cfg.get("pid_lateral_kd", 0.12))
        self.heading_kp = float(cfg.get("pid_heading_kp", 0.35))
        self.heading_ki = float(cfg.get("pid_heading_ki", 0.0))
        self.heading_kd = float(cfg.get("pid_heading_kd", 0.04))
        self.preview_lookahead_time_s = float(
            cfg.get("preview_lookahead_time_s", 0.6)
        )
        self.preview_lookahead_min_m = float(
            cfg.get("preview_lookahead_min_m", 3.0)
        )
        self.preview_lookahead_max_m = float(
            cfg.get("preview_lookahead_max_m", 8.0)
        )
        self.preview_heading_weight = float(
            cfg.get("preview_heading_weight", 0.5)
        )
        self.cross_track_kp = float(cfg.get("pid_cross_track_kp", 0.4))
        self.s9_cross_track_kp = float(cfg.get("s9_pid_cross_track_kp", 0.0))
        if (
            not np.isfinite(self.cross_track_kp)
            or self.cross_track_kp < 0.0
            or not np.isfinite(self.s9_cross_track_kp)
            or self.s9_cross_track_kp < 0.0
        ):
            raise ValueError("PID cross-track gains must be finite and non-negative")
        self._state: dict[str, dict[str, float]] = {}
        self._longitudinal = LongitudinalCascadeController(
            dt_s=decision_dt,
            acceleration_bias_mps2=float(
                cfg.get("acceleration_bias_mps2", 0.0)
            ),
            drive_acceleration_scale_mps2=float(
                cfg.get(
                    "drive_acceleration_scale_mps2",
                    DRIVE_ACCELERATION_SCALE_MPS2,
                )
            ),
            brake_acceleration_scale_mps2=float(
                cfg.get(
                    "brake_acceleration_scale_mps2",
                    BRAKE_ACCELERATION_SCALE_MPS2,
                )
            ),
        )
        self._last_debug: dict[str, dict[str, object]] = {}

    def reset(self) -> None:
        self._state.clear()
        self._longitudinal.reset()
        self._last_debug = {}

    def get_last_debug(self) -> dict[str, dict[str, object]]:
        return {key: dict(value) for key, value in self._last_debug.items()}

    def compute_actions(
        self,
        env,
        trajectories_world: dict[str, np.ndarray],
        longitudinal_references: dict[str, LongitudinalTrackingReference] | None = None,
        lateral_tracking_errors_m: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        debug: dict[str, dict[str, object]] = {}
        agents = getattr(env, "agents", {}) or {}
        scenario_id = str((getattr(env, "config", {}) or {}).get("scenario_id", ""))
        effective_cross_track_kp = (
            self.s9_cross_track_kp
            if scenario_id == "S9_narrow_channel_negotiation"
            else self.cross_track_kp
        )
        for agent_id, trajectory_world in (trajectories_world or {}).items():
            vehicle = agents.get(agent_id)
            if vehicle is None:
                continue
            trajectory_local = _world_trajectory_to_ego_local(vehicle, trajectory_world)
            reference = (
                (longitudinal_references or {}).get(agent_id)
                if longitudinal_references is not None
                else None
            )
            action, agent_debug = self._single_control_with_debug(
                agent_id,
                vehicle,
                trajectory_local,
                reference,
                cross_track_error_m=float(
                    (lateral_tracking_errors_m or {}).get(agent_id, 0.0)
                ),
                cross_track_kp=effective_cross_track_kp,
            )
            actions[agent_id] = action
            debug[agent_id] = agent_debug
        self._last_debug = debug
        return actions

    def _pid(self, agent_id: str, key: str, error: float, kp: float, ki: float, kd: float) -> float:
        state = self._state.setdefault(agent_id, {})
        prev_key = f"{key}_prev"
        int_key = f"{key}_int"
        prev = float(state.get(prev_key, error))
        integral = float(state.get(int_key, 0.0)) + error * self.dt
        derivative = (error - prev) / max(self.dt, 1e-6)
        state[prev_key] = error
        state[int_key] = integral
        return kp * error + ki * integral + kd * derivative

    def _single_control(self, agent_id: str, vehicle, trajectory_local: np.ndarray) -> np.ndarray:
        action, debug = self._single_control_with_debug(
            agent_id, vehicle, trajectory_local, None
        )
        trajectory_local = np.asarray(trajectory_local, dtype=np.float32)
        if trajectory_local.ndim == 2 and trajectory_local.shape[0] > 0:
            debug_index = min(max(self.lookahead_index, 0), trajectory_local.shape[0] - 1)
            save_pid_debug_plot(
                agent_id=agent_id,
                steering=float(action[0]),
                actual_heading=float(getattr(vehicle, "heading_theta", 0.0)),
                heading_error=_wrap_to_pi(float(trajectory_local[debug_index, 2])),
            )
        return action

    def _single_control_with_debug(
        self,
        agent_id: str,
        vehicle,
        trajectory_local: np.ndarray,
        longitudinal_reference: LongitudinalTrackingReference | None,
        *,
        gap_acceleration_mps2: float = 0.0,
        cross_track_error_m: float = 0.0,
        cross_track_kp: float | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        trajectory_local = np.asarray(trajectory_local, dtype=np.float32)
        if trajectory_local.ndim != 2 or trajectory_local.shape[0] == 0:
            return np.zeros((2,), dtype=np.float32), {
                "mode": "no_trajectory",
                "gap_feedback_mps2": 0.0,
            }
        current_speed_mps = signed_longitudinal_speed_mps(vehicle)
        lookahead_m = float(
            np.clip(
                self.preview_lookahead_time_s * abs(current_speed_mps),
                self.preview_lookahead_min_m,
                self.preview_lookahead_max_m,
            )
        )
        path_xy = np.concatenate(
            (
                np.zeros((1, 2), dtype=np.float64),
                trajectory_local[:, :2].astype(np.float64, copy=False),
            ),
            axis=0,
        )
        arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))
        )
        query = min(lookahead_m, float(arc[-1]))
        preview_direction_guarded = False
        first_path_heading = float(trajectory_local[0, 2])
        if query > 1.0e-6:
            preview_headings = np.concatenate(
                (
                    [0.0],
                    np.unwrap(
                        trajectory_local[:, 2].astype(
                            np.float64, copy=False
                        )
                    ),
                )
            )
            raw_preview_heading = float(
                np.interp(query, arc, preview_headings)
            )
            # At an exit transition the path can finish a shallow left bend
            # before beginning the requested right turn.  A long pure-pursuit
            # lookahead must not skip across that sign change and command the
            # later turn while the first reachable segment still turns the
            # other way.  Cap only that sign-changing case at the first fixed
            # waypoint; ordinary same-direction preview is unchanged.
            direction_threshold = 0.005
            if (
                abs(first_path_heading) >= direction_threshold
                and abs(raw_preview_heading) >= direction_threshold
                and first_path_heading * raw_preview_heading < 0.0
            ):
                query = min(query, float(arc[1]))
                preview_direction_guarded = True
        if query <= 1.0e-6:
            lateral_error = 0.0
            preview_heading = 0.0
        else:
            preview_x = float(np.interp(query, arc, path_xy[:, 0]))
            preview_y = float(np.interp(query, arc, path_xy[:, 1]))
            headings = np.concatenate(
                (
                    [0.0],
                    np.unwrap(
                        trajectory_local[:, 2].astype(
                            np.float64, copy=False
                        )
                    ),
                )
            )
            preview_heading = float(np.interp(query, arc, headings))
            lateral_error = _wrap_to_pi(
                float(np.arctan2(preview_y, max(preview_x, 1.0e-3)))
                + self.preview_heading_weight * preview_heading
            )
        steering = self._pid(
            agent_id,
            "lat",
            lateral_error,
            self.lateral_kp,
            self.lateral_ki,
            self.lateral_kd,
        )
        if not np.isfinite(cross_track_error_m):
            raise ValueError("cross_track_error_m must be finite")
        effective_cross_track_kp = (
            self.cross_track_kp
            if cross_track_kp is None
            else float(cross_track_kp)
        )
        cross_track_correction = -effective_cross_track_kp * float(
            cross_track_error_m
        )
        steering += cross_track_correction
        # Keep the first fixed-time heading visible in diagnostics.  It is not
        # an instantaneous heading target: on a high-curvature path a preview
        # tracker is expected to rotate before reaching that waypoint.  Adding
        # a second PID term here double-counts the same future curvature.
        tracking_heading_error = _wrap_to_pi(float(trajectory_local[0, 2]))
        heading_correction = 0.0
        actual_heading = float(getattr(vehicle, "heading_theta", 0.0))
        controller_state = self._state.setdefault(agent_id, {})
        previous_actual_heading = controller_state.get("actual_heading_rad")
        actual_yaw_rate = (
            0.0
            if previous_actual_heading is None
            else _wrap_to_pi(actual_heading - float(previous_actual_heading))
            / self.dt
        )
        controller_state["actual_heading_rad"] = actual_heading
        preview_reference_heading_world = _wrap_to_pi(
            actual_heading + float(preview_heading)
        )
        first_reference_heading_world = _wrap_to_pi(
            actual_heading + float(first_path_heading)
        )

        reference = longitudinal_reference or trajectory_to_longitudinal_reference(
            trajectory_local,
            max(current_speed_mps, 0.0),
            source="online_trajectory",
        )
        throttle, longitudinal_debug = self._longitudinal.compute(
            agent_id,
            current_speed_mps,
            reference,
            gap_acceleration_mps2=gap_acceleration_mps2,
        )
        action = np.asarray(
            [np.clip(steering, -1.0, 1.0), throttle],
            dtype=np.float32,
        )
        debug: dict[str, object] = {
            "mode": "trajectory_cascade",
            "current_speed_mps": float(current_speed_mps),
            "preview_lookahead_m": float(lookahead_m),
            "preview_query_m": float(query),
            "preview_heading_rad": float(preview_heading),
            "first_path_heading_rad": float(first_path_heading),
            "preview_direction_guarded": bool(
                preview_direction_guarded
            ),
            "preview_lateral_error_rad": float(lateral_error),
            "cross_track_error_m": float(cross_track_error_m),
            "cross_track_kp": float(effective_cross_track_kp),
            "cross_track_correction": float(cross_track_correction),
            "tracking_heading_error_rad": float(tracking_heading_error),
            "heading_correction": float(heading_correction),
            "actual_heading_rad": float(actual_heading),
            "actual_yaw_rate_rad_s": float(actual_yaw_rate),
            "preview_reference_heading_world_rad": float(
                preview_reference_heading_world
            ),
            "first_reference_heading_world_rad": float(
                first_reference_heading_world
            ),
            "raw_steering": float(steering),
            "clipped_steering": float(action[0]),
            "clipped_throttle": float(action[1]),
        }
        debug.update(longitudinal_debug)
        return action, debug
