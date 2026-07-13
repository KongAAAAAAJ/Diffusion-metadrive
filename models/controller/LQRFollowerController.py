"""LQR-based platoon follower controller.

Agent 0 (leader): PID trajectory tracking via PIDTrajectoryController.
Agent i (follower, i > 0): LQR spacing + lateral control relative to preceding vehicle.

Longitudinal LQR state: [desired_gap - actual_gap, v_ego - v_prev, a_ego]
  A = [[1,dt,0],[0,1,dt],[0,0,1-dt/Ts]], B = [[0],[0],[dt/Ts]]
  -> desired_accel = -K @ state

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

LQR_LON_Q1 = 10.0  # s
LQR_LON_Q2 = 1  # v
LQR_LON_Q3 = 0.01  # a
LQR_LON_R = 10  # a
LQR_LON_TS = 0.1  # engine/actuator lag

LQR_LAT_Q1 = 1.0
LQR_LAT_Q2 = 1.0
LQR_LAT_R = 0.1

LQR_DESIRED_GAP_M = 10.0
LQR_DESIRED_OUTGAP_M = 20.0   # target gap when following a background vehicle
TARGET_SPEED_KMH = 30.0        # free-flow target speed when no front vehicle (km/h)
LQR_MAX_ACCEL_MPS2 = 1.0
LQR_MIN_ACCEL_MPS2 = -2.0
LQR_WHEELBASE_M = 2.8  # xl车辆
LQR_PHYSICS_WORLD_STEP_SIZE = 2e-2
LQR_DECISION_REPEAT = 5


def _solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return LQR gain K for continuous-time system via ARE. Returns zeros on failure."""
    try:
        P = linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K
    except Exception:
        return np.zeros((R.shape[0], Q.shape[0]))


def _solve_discrete_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return discrete-time LQR gain K. Returns zeros on failure."""
    try:
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
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
        self.lqr_lon_q1 = LQR_LON_Q1  # gap error weight
        self.lqr_lon_q2 = LQR_LON_Q2  # speed error weight
        self.lqr_lon_q3 = LQR_LON_Q3  # acceleration state weight
        self.lqr_lon_r  = LQR_LON_R   # accel effort weight
        self.lqr_lon_Ts = LQR_LON_TS  # engine/actuator lag

        # Lateral LQR weights
        self.lqr_lat_q1 = LQR_LAT_Q1  # lateral error weight
        self.lqr_lat_q2 = LQR_LAT_Q2  # heading error weight
        self.lqr_lat_r  = LQR_LAT_R   # steering effort weight

        self.desired_gap_m    = LQR_DESIRED_GAP_M
        self.desired_outgap_m = LQR_DESIRED_OUTGAP_M
        self.target_speed_mps = TARGET_SPEED_KMH / 3.6
        self.max_accel_mps2   = LQR_MAX_ACCEL_MPS2
        self.min_accel_mps2   = LQR_MIN_ACCEL_MPS2
        self.wheelbase_m      = LQR_WHEELBASE_M
        physics_dt            = LQR_PHYSICS_WORLD_STEP_SIZE
        decision_repeat       = LQR_DECISION_REPEAT
        self.dt               = physics_dt * max(decision_repeat, 1)

        self._prev_speed_ms: dict[str, float] = {}
        self._last_debug: dict[str, dict] = {}

    def reset(self) -> None:
        self._pid.reset()
        self._prev_speed_ms.clear()
        self._last_debug = {}

    def get_last_debug(self) -> dict[str, dict]:
        return copy.deepcopy(self._last_debug)

    @staticmethod
    def _get_traffic_vehicles(env) -> list:
        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        if traffic_manager is None:
            return []
        vehicles = getattr(traffic_manager, "traffic_vehicles", None)
        if vehicles is not None:
            try:
                return list(vehicles)
            except TypeError:
                pass
        return list(getattr(traffic_manager, "_traffic_vehicles", []) or [])

    @staticmethod
    def _find_front_vehicle(
        ego_vehicle,
        agents: dict,
        traffic_vehicles: list,
    ) -> tuple:
        """Find closest front vehicle in the same lane.

        Returns (vehicle, bumper_gap_m, is_platoon_member) or (None, None, None).
        """
        ego_lane = getattr(ego_vehicle, "lane", None)
        if ego_lane is None:
            return None, None, None
        ego_pos = np.asarray(getattr(ego_vehicle, "position", (0.0, 0.0))[:2], dtype=float)
        ego_s, _ = ego_lane.local_coordinates(ego_pos)
        ego_len = float(getattr(ego_vehicle, "LENGTH", 5.0))
        lane_index = tuple(getattr(ego_lane, "index", ()) or ())
        lane_width = float(getattr(ego_lane, "width", 3.5) or 3.5)
        agent_set = set(agents.values())

        best_vehicle = None
        best_gap: float | None = None
        best_is_platoon = False

        for candidate in list(agents.values()) + list(traffic_vehicles):
            if candidate is None or candidate is ego_vehicle:
                continue
            c_lane = getattr(candidate, "lane", None)
            if c_lane is None:
                continue
            if tuple(getattr(c_lane, "index", ()) or ()) != lane_index:
                continue
            c_pos = np.asarray(getattr(candidate, "position", (0.0, 0.0))[:2], dtype=float)
            c_s, c_t = ego_lane.local_coordinates(c_pos)
            if abs(float(c_t)) > 0.5 * lane_width:
                continue
            c_len = float(getattr(candidate, "LENGTH", 5.0))
            gap = float(c_s) - float(ego_s) - 0.5 * c_len - 0.5 * ego_len
            if gap <= 0.0:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_vehicle = candidate
                best_is_platoon = candidate in agent_set

        return best_vehicle, best_gap, best_is_platoon

    def compute_actions(
        self,
        env,
        trajectories_world: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        agents = getattr(env, "agents", {}) or {}
        all_ids: list[str] = list(getattr(env, "_agent_ids", sorted(agents.keys())))
        active_ids = [aid for aid in all_ids if aid in agents]
        traffic_vehicles = self._get_traffic_vehicles(env)

        actions: dict[str, np.ndarray] = {}
        debug: dict[str, dict] = {}

        for agent_id in active_ids:
            vehicle = agents[agent_id]
            traj_world = (trajectories_world or {}).get(agent_id)
            traj_local = (
                _world_trajectory_to_ego_local(vehicle, traj_world)
                if traj_world is not None
                else np.zeros((0, 3), dtype=np.float32)
            )
            front_veh, gap_m, is_platoon = self._find_front_vehicle(vehicle, agents, traffic_vehicles)

            if front_veh is None:
                ego_spd_ms = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6
                speed_diff = self.target_speed_mps - ego_spd_ms
                raw_throttle, lon_debug = self._lon_lqr(0.0, speed_diff, agent_id, ego_spd_ms)
                raw_steering = self._pid._single_control(agent_id, vehicle, traj_local)[0]
                action = np.asarray(
                    [np.clip(raw_steering, -1.0, 1.0), np.clip(raw_throttle, -1.0, 1.0)],
                    dtype=np.float32,
                )
                actions[agent_id] = action
                debug[agent_id] = {
                    "mode": "free_speed_tracking",
                    "target_speed_mps": float(self.target_speed_mps),
                    "ego_speed_mps": float(ego_spd_ms),
                    "raw_steering": float(raw_steering),
                    "clipped_steering": float(action[0]),
                    "raw_throttle": float(raw_throttle),
                    "clipped_throttle": float(action[1]),
                }
                debug[agent_id].update(lon_debug)
            else:
                desired_gap = self.desired_gap_m if is_platoon else self.desired_outgap_m
                front_id = next(
                    (k for k, v in agents.items() if v is front_veh), "background"
                )
                mode = "follower_platoon" if is_platoon else "follower_background"
                actions[agent_id], debug[agent_id] = self._lqr_follow(
                    agent_id, vehicle, front_id, front_veh, desired_gap, mode, traj_local
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
        desired_gap_m: float,
        mode: str = "follower_platoon",
        traj_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        ego_pos     = np.asarray(getattr(ego_vehicle,  "position",     (0., 0., 0.))[:2], dtype=float)
        ego_heading = float(getattr(ego_vehicle,  "heading_theta", 0.0))
        ego_spd_ms  = float(getattr(ego_vehicle,  "speed_km_h",   0.0) or 0.0) / 3.6
        ego_len     = float(getattr(ego_vehicle,  "LENGTH",        5.0))

        prev_pos     = np.asarray(getattr(prev_vehicle, "position",     (0., 0., 0.))[:2], dtype=float)
        prev_heading = float(getattr(prev_vehicle, "heading_theta", 0.0))
        prev_spd_ms  = float(getattr(prev_vehicle, "speed_km_h",   0.0) or 0.0) / 3.6
        prev_len     = float(getattr(prev_vehicle, "LENGTH",        5.0))

        # Relative state in preceding vehicle's local frame (x=forward, y=left)
        delta = ego_pos - prev_pos
        cos_h = np.cos(prev_heading)
        sin_h = np.sin(prev_heading)
        x_rel =  cos_h * delta[0] + sin_h * delta[1]   # <0 when ego is behind prev
        y_rel = -sin_h * delta[0] + cos_h * delta[1]   # >0 when ego is left of prev

        half_len   = 0.5 * (ego_len + prev_len)
        actual_gap = max(-x_rel - half_len, 0.0)        # bumper-to-bumper gap
        gap_error  = actual_gap - desired_gap_m         # >0: too far, <0: too close
        speed_diff = prev_spd_ms - ego_spd_ms           # v_prev - v_ego

        lat_error     = y_rel
        heading_error = _wrap_to_pi(ego_heading - prev_heading)

        'lqr lon'
        raw_throttle, lon_debug = self._lon_lqr(gap_error, speed_diff, ego_id, ego_spd_ms)
        _traj = traj_local if traj_local is not None else np.zeros((0, 3), dtype=np.float32)

        'lqr lat'
        raw_steering = self._pid._single_control(ego_id, ego_vehicle, _traj)[0]
        
        action = np.asarray(
            [np.clip(raw_steering, -1.0, 1.0), np.clip(raw_throttle, -1.0, 1.0)],
            dtype=np.float32,
        )
        debug = {
            "mode": mode,
            "leader_id": prev_id,
            "desired_gap_m": float(desired_gap_m),
            "lat_error": float(lat_error),
            "heading_error": float(heading_error),
            "gap_error": float(gap_error),
            "speed_diff": float(speed_diff),
            "ego_speed_mps": float(ego_spd_ms),
            "raw_steering": float(raw_steering),
            "clipped_steering": float(action[0]),
            "raw_throttle": float(raw_throttle),
            "clipped_throttle": float(action[1]),
        }
        debug.update(lon_debug)
        return action, debug

    def _lon_lqr(
        self,
        gap_error: float,
        speed_diff: float,
        ego_id: str,
        ego_spd_ms: float,
    ) -> tuple[float, dict]:
        """Third-order longitudinal LQR. Returns normalized throttle and diagnostics."""
        dt = max(float(self.dt), 1e-6)
        Ts = max(float(self.lqr_lon_Ts), 1e-6)
        A = np.array(
            [
                [1.0, dt, 0.0],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0 - dt / Ts],
            ],
            dtype=np.float64,
        )
        B = np.array([[0.0], [0.0], [dt / Ts]], dtype=np.float64)
        Q = np.diag([self.lqr_lon_q1, self.lqr_lon_q2, self.lqr_lon_q3])
        R = np.array([[self.lqr_lon_r]])
        K = _solve_discrete_lqr(A, B, Q, R)

        spacing_error = -float(gap_error)
        velocity_error = -float(speed_diff)
        ego_accel = self._estimate_accel(ego_id, ego_spd_ms)
        state = np.asarray([spacing_error, velocity_error, ego_accel], dtype=np.float64)
        desired_accel = -float((K @ state)[0])
        if desired_accel >= 0.0:
            accel_scale = max(self.max_accel_mps2, 1e-3)
        else:
            accel_scale = max(abs(self.min_accel_mps2), 1e-3)
        raw_throttle = desired_accel / accel_scale
        debug = {
            "desired_accel_mps2": float(desired_accel),
            "ego_accel_mps2": float(ego_accel),
            "max_accel_mps2": float(self.max_accel_mps2),
            "min_accel_mps2": float(self.min_accel_mps2),
            "lon_state": state.tolist(),
            "K_lon": np.asarray(K, dtype=np.float64).reshape(-1).tolist(),
            "lon_q": {
                "spacing": float(self.lqr_lon_q1),
                "velocity": float(self.lqr_lon_q2),
                "accel": float(self.lqr_lon_q3),
            },
            "lon_r": float(self.lqr_lon_r),
            "lon_dt": float(dt),
            "lon_Ts": float(Ts),
        }
        return raw_throttle, debug

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
