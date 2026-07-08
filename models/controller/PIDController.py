"""PID-based trajectory tracking controller for platoon agents."""

from __future__ import annotations

import numpy as np

from models.controller.base_controller import BaseController
from models.controller.controller_helper import save_pid_debug_plot


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
        self.lookahead_index = int(cfg.get("pid_lookahead_index", cfg.get("lookahead_index", 2)))
        self.dt = float(cfg.get("pid_dt", 0.5))
        self.target_speed_km_h = float(cfg.get("target_speed_km_h", 30.0))
        self.lateral_kp = float(cfg.get("pid_lateral_kp", 0.85))
        self.lateral_ki = float(cfg.get("pid_lateral_ki", 0.0))
        self.lateral_kd = float(cfg.get("pid_lateral_kd", 0.08))
        self.heading_kp = float(cfg.get("pid_heading_kp", 0.35))
        self.heading_ki = float(cfg.get("pid_heading_ki", 0.0))
        self.heading_kd = float(cfg.get("pid_heading_kd", 0.04))
        self.speed_kp = float(cfg.get("pid_speed_kp", 0.10))
        self.speed_ki = float(cfg.get("pid_speed_ki", 0.0))
        self.speed_kd = float(cfg.get("pid_speed_kd", 0.02))
        self._state: dict[str, dict[str, float]] = {}

    def reset(self) -> None:
        self._state.clear()

    def compute_actions(self, env, trajectories_world: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        agents = getattr(env, "agents", {}) or {}
        for agent_id, trajectory_world in (trajectories_world or {}).items():
            vehicle = agents.get(agent_id)
            if vehicle is None:
                continue
            trajectory_local = _world_trajectory_to_ego_local(vehicle, trajectory_world)
            actions[agent_id] = self._single_control(agent_id, vehicle, trajectory_local)
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
        trajectory_local = np.asarray(trajectory_local, dtype=np.float32)
        if trajectory_local.ndim != 2 or trajectory_local.shape[0] == 0:
            return np.zeros((2,), dtype=np.float32)
        valid_idx = min(max(self.lookahead_index, 0), trajectory_local.shape[0] - 1)
        waypoint = trajectory_local[valid_idx]
        forward = max(float(waypoint[0]), 1e-3)
        lateral_angle_error = float(np.arctan2(float(waypoint[1]), forward))
        # heading_error = float(waypoint[2]) if waypoint.shape[0] > 2 else 0.0
        steering = self._pid(agent_id, "lat", lateral_angle_error, self.lateral_kp, self.lateral_ki, self.lateral_kd)
        # steering += self._pid(agent_id, "heading", heading_error, self.heading_kp, self.heading_ki, self.heading_kd)

        segment_distances = np.linalg.norm(np.diff(trajectory_local[:, :2], axis=0), axis=1)
        if segment_distances.size > 0:
            reference_speeds = np.concatenate([segment_distances, segment_distances[-1:]], axis=0) / max(self.dt, 1e-6)
            trajectory_target_km_h = float(reference_speeds[min(valid_idx, len(reference_speeds) - 1)] * 3.6)
        else:
            trajectory_target_km_h = 0.0
        target_speed_km_h = min(self.target_speed_km_h, trajectory_target_km_h)
        current_speed_km_h = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
        speed_error = (target_speed_km_h - current_speed_km_h) / 10.0
        throttle = self._pid(agent_id, "speed", speed_error, self.speed_kp, self.speed_ki, self.speed_kd)
        return np.asarray(
            [np.clip(steering, -1.0, 1.0), np.clip(throttle, -1.0, 1.0)],
            dtype=np.float32,
        )
