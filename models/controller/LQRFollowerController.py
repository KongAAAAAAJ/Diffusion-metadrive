"""LQR-based platoon follower controller.

Agent 0 (leader): PID trajectory tracking via PIDTrajectoryController.
Agent i (follower, i > 0): LQR spacing + lateral control relative to preceding vehicle.

Longitudinal LQR state: [gap_error, v_prev - v_ego]
  gap_error = (bumper-to-bumper gap) - desired_gap_m
  A = [[0,1],[0,0]], B = [[0],[1]], u = a_prev - a_ego
  -> a_ego = a_prev + K[0]*gap_error + K[1]*(v_prev - v_ego)

Lateral LQR state: [lat_error, heading_error]
  lat_error = y-offset of ego in preceding vehicle's local frame
  A = [[0,v],[0,0]], B = [[0],[v/L]], u = steering_angle
"""

from __future__ import annotations

import copy
import numpy as np
from scipy import linalg

from models.controller.base_controller import BaseController
from models.controller.PIDController import (
    PIDTrajectoryController,
    _world_trajectory_to_ego_local,
    _wrap_to_pi,
)


def _solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return LQR gain K for continuous-time system via ARE. Returns zeros on failure."""
    try:
        P = linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K
    except Exception:
        return np.zeros((R.shape[0], Q.shape[0]))


class LQRFollowerController(BaseController):
    """Platoon controller: PID leader + LQR followers."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config

        self._pid = PIDTrajectoryController(config)

        # Longitudinal LQR weights
        self.lqr_lon_q1 = float(cfg.get("lqr_lon_q1", 1.0))   # gap error weight
        self.lqr_lon_q2 = float(cfg.get("lqr_lon_q2", 2.0))   # speed error weight
        self.lqr_lon_r  = float(cfg.get("lqr_lon_r",  0.5))   # accel effort weight

        # Lateral LQR weights
        self.lqr_lat_q1 = float(cfg.get("lqr_lat_q1", 1.0))   # lateral error weight
        self.lqr_lat_q2 = float(cfg.get("lqr_lat_q2", 1.0))   # heading error weight
        self.lqr_lat_r  = float(cfg.get("lqr_lat_r",  0.1))   # steering effort weight

        self.desired_gap_m  = float(cfg.get("lqr_desired_gap_m",  10.0))
        self.max_accel_mps2 = float(cfg.get("lqr_max_accel_mps2",  4.0))
        self.wheelbase_m    = float(cfg.get("lqr_wheelbase_m",     3.0))
        self.dt             = float(cfg.get("pid_dt", 0.5))

        self._prev_speed_ms: dict[str, float] = {}
        self._last_debug: dict[str, dict] = {}

    def reset(self) -> None:
        self._pid.reset()
        self._prev_speed_ms.clear()
        self._last_debug = {}

    def get_last_debug(self) -> dict[str, dict]:
        return copy.deepcopy(self._last_debug)

    def compute_actions(
        self,
        env,
        trajectories_world: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        agents = getattr(env, "agents", {}) or {}
        all_ids: list[str] = list(getattr(env, "_agent_ids", sorted(agents.keys())))
        active_ids = [aid for aid in all_ids if aid in agents]

        actions: dict[str, np.ndarray] = {}
        debug: dict[str, dict] = {}

        for idx, agent_id in enumerate(active_ids):
            vehicle = agents[agent_id]
            if idx == 0:
                traj = trajectories_world.get(agent_id)
                if traj is not None:
                    traj_local = _world_trajectory_to_ego_local(vehicle, traj)
                    actions[agent_id] = self._pid._single_control(agent_id, vehicle, traj_local)
                else:
                    actions[agent_id] = np.zeros(2, dtype=np.float32)
                debug[agent_id] = {
                    "mode": "leader_pid",
                    "clipped_steering": float(actions[agent_id][0]),
                    "clipped_throttle": float(actions[agent_id][1]),
                }
            else:
                prev_id = active_ids[idx - 1]
                prev_vehicle = agents.get(prev_id)
                if prev_vehicle is None:
                    actions[agent_id] = np.zeros(2, dtype=np.float32)
                    debug[agent_id] = {
                        "mode": "missing_leader_zero",
                        "leader_id": prev_id,
                        "clipped_steering": 0.0,
                        "clipped_throttle": 0.0,
                    }
                else:
                    actions[agent_id], debug[agent_id] = self._lqr_follow(
                        agent_id, vehicle, prev_id, prev_vehicle
                    )

        for aid in active_ids:
            v = agents.get(aid)
            if v is not None:
                self._prev_speed_ms[aid] = float(getattr(v, "speed_km_h", 0.0) or 0.0) / 3.6

        self._last_debug = debug
        return actions

    def _lqr_follow(
        self,
        ego_id: str,
        ego_vehicle,
        prev_id: str,
        prev_vehicle,
    ) -> tuple[np.ndarray, dict]:
        ego_pos     = np.asarray(getattr(ego_vehicle,  "position",     (0., 0., 0.))[:2], dtype=float)
        ego_heading = float(getattr(ego_vehicle,  "heading_theta", 0.0))
        ego_spd_ms  = float(getattr(ego_vehicle,  "speed_km_h",   0.0) or 0.0) / 3.6
        ego_len     = float(getattr(ego_vehicle,  "LENGTH",        5.0))

        prev_pos     = np.asarray(getattr(prev_vehicle, "position",     (0., 0., 0.))[:2], dtype=float)
        prev_heading = float(getattr(prev_vehicle, "heading_theta", 0.0))
        prev_spd_ms  = float(getattr(prev_vehicle, "speed_km_h",   0.0) or 0.0) / 3.6
        prev_len     = float(getattr(prev_vehicle, "LENGTH",        5.0))
        prev_accel   = self._estimate_accel(prev_id, prev_spd_ms)

        # Relative state in preceding vehicle's local frame (x=forward, y=left)
        delta = ego_pos - prev_pos
        cos_h = np.cos(prev_heading)
        sin_h = np.sin(prev_heading)
        x_rel =  cos_h * delta[0] + sin_h * delta[1]   # <0 when ego is behind prev
        y_rel = -sin_h * delta[0] + cos_h * delta[1]   # >0 when ego is left of prev

        half_len   = 0.5 * (ego_len + prev_len)
        actual_gap = max(-x_rel - half_len, 0.0)        # bumper-to-bumper gap
        gap_error  = actual_gap - self.desired_gap_m    # >0: too far, <0: too close
        speed_diff = prev_spd_ms - ego_spd_ms           # v_prev - v_ego

        lat_error     = y_rel
        heading_error = _wrap_to_pi(ego_heading - prev_heading)

        raw_throttle = self._lon_lqr(gap_error, speed_diff, prev_accel)
        raw_steering, k_lat = self._lat_lqr_with_gain(lat_error, heading_error, ego_spd_ms)
        action = np.asarray(
            [np.clip(raw_steering, -1.0, 1.0), np.clip(raw_throttle, -1.0, 1.0)],
            dtype=np.float32,
        )
        debug = {
            "mode": "follower_lqr",
            "leader_id": prev_id,
            "lat_error": float(lat_error),
            "heading_error": float(heading_error),
            "gap_error": float(gap_error),
            "speed_diff": float(speed_diff),
            "ego_speed_mps": float(ego_spd_ms),
            "raw_steering": float(raw_steering),
            "clipped_steering": float(action[0]),
            "raw_throttle": float(raw_throttle),
            "clipped_throttle": float(action[1]),
            "K_lat": np.asarray(k_lat, dtype=np.float64).reshape(-1).tolist(),
            "q": {"lat": float(self.lqr_lat_q1), "heading": float(self.lqr_lat_q2)},
            "r": float(self.lqr_lat_r),
        }
        return action, debug

    def _lon_lqr(self, gap_error: float, speed_diff: float, prev_accel: float) -> float:
        """Longitudinal LQR: a_ego = a_prev + K @ [gap_error, v_prev - v_ego]."""
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.diag([self.lqr_lon_q1, self.lqr_lon_q2])
        R = np.array([[self.lqr_lon_r]])
        K = _solve_lqr(A, B, Q, R)
        u     = float((K @ np.array([gap_error, speed_diff]))[0])
        a_ego = prev_accel + u
        return a_ego / max(self.max_accel_mps2, 1e-3)

    def _lat_lqr(self, lat_error: float, heading_error: float, ego_spd_ms: float) -> float:
        """Lateral LQR: steering = -K @ [lat_error, heading_error]."""
        steering, _ = self._lat_lqr_with_gain(lat_error, heading_error, ego_spd_ms)
        return steering

    def _lat_lqr_with_gain(self, lat_error: float, heading_error: float, ego_spd_ms: float) -> tuple[float, np.ndarray]:
        """Lateral LQR with the solved gain for diagnostics."""
        v = max(abs(ego_spd_ms), 0.5)
        L = max(self.wheelbase_m, 1.0)
        A = np.array([[0.0, v  ], [0.0, 0.0]])
        B = np.array([[0.0     ], [v / L    ]])
        Q = np.diag([self.lqr_lat_q1, self.lqr_lat_q2])
        R = np.array([[self.lqr_lat_r]])
        K = _solve_lqr(A, B, Q, R)
        return -float((K @ np.array([lat_error, heading_error]))[0]), K

    def _estimate_accel(self, agent_id: str, current_spd_ms: float) -> float:
        prev = self._prev_speed_ms.get(agent_id, current_spd_ms)
        return (current_spd_ms - prev) / max(self.dt, 1e-6)
