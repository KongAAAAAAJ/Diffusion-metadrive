"""platoon_lqr_expert.py

Complete platoon expert: longitudinal LQR + lateral dual-PID (matching IDM style).

Topology: agent0 = leader (tracks desired speed); agent1…N-1 = followers
          (each follows its immediate predecessor).

Longitudinal control (LQR)
--------------------------
Leader  — 1-state LQR on speed error:
    state  x = [v_desired - v_ego]
    model  x[t+1] = x[t] - dt * u[t]
    u*     = -K_leader @ x          (positive u = accelerate)

Follower — 3-state LQR on gap + relative velocity + speed error:
    state  x = [e_gap, e_rel_vel, e_speed]
               e_gap     = d_actual - d_desired   (+ → too far behind → accelerate)
               e_rel_vel = v_front - v_ego         (+ → front faster   → accelerate)
               e_speed   = v_desired - v_ego       (+ → below desired  → accelerate)
    model  x[t+1] ≈ A @ x[t] + B * u[t]
               A = [[1, dt, 0], [0, 1, 0], [0, 0, 1]]
               B = [[0], [-dt], [-dt]]
    u*     = -K_follower @ x

Desired gap: d_desired = standstill_gap_m + headway_time_s * v_ego

Lateral control (dual-PID, identical to IDM steering_control)
-------------------------------------------------------------
Both leader and followers track the reference lane (veh.lane) using:
    long, lat = lane.local_coordinates(veh.position)
    lane_hdg   = lane.heading_theta_at(long + lateral_lookahead_m)
    hdg_err    = wrap_to_pi(lane_hdg - veh.heading_theta)
    steering   = heading_pid(-hdg_err) + lateral_pid(-lat)

PID controllers are stateful (integrators); call expert.reset() at each episode start.
If veh.lane is None (e.g. in unit tests with fake vehicles), steering = 0.0 gracefully.

Public interface
----------------
    cfg    = PlatoonLQRConfig(desired_speed_km_h=30.0, headway_time_s=0.5)
    expert = PlatoonLQRExpert(cfg)
    expert.reset()                             # call at episode start

    # Every step:
    actions = expert.compute_actions(env)      # → {agent_id: np.array([steer, throttle])}
    env.low_level_step(actions)

    # With diagnostics:
    out = expert.compute_actions_with_state(env)
    for aid, st in out.agent_states.items():
        print(aid, st.e_gap, st.steering, st.lateral_error_m)

Output format
-------------
actions[agent_id] = np.array([steering, throttle], dtype=float32)
  • steering ∈ [-1, 1]: positive = steer right (MetaDrive convention)
  • throttle ∈ [-1, 1]: positive = accelerate, negative = brake
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PIDController:
    """Minimal stateful PID controller (no MetaDrive dependency)."""

    __slots__ = ("kp", "ki", "kd", "p_error", "i_error", "d_error")

    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.p_error = self.i_error = self.d_error = 0.0

    def get_result(self, error: float) -> float:
        self.i_error += error
        self.d_error = error - self.p_error
        self.p_error = error
        return -(self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error)

    def reset(self) -> None:
        self.p_error = self.i_error = self.d_error = 0.0


class _AgentLateralState:
    """Per-agent PID controllers for lateral tracking."""

    __slots__ = ("heading_pid", "lateral_pid")

    def __init__(self, cfg: "PlatoonLQRConfig") -> None:
        self.heading_pid = _PIDController(cfg.heading_pid_kp, cfg.heading_pid_ki, cfg.heading_pid_kd)
        self.lateral_pid = _PIDController(cfg.lateral_pid_kp, cfg.lateral_pid_ki, cfg.lateral_pid_kd)

    def reset(self) -> None:
        self.heading_pid.reset()
        self.lateral_pid.reset()


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlatoonLQRConfig:
    """All tunable parameters for the platoon LQR expert."""

    # Speed reference
    desired_speed_km_h: float = 30.0   # target cruise speed for the leader

    # Spacing policy:  d_desired = standstill_gap_m + headway_time_s * v_ego
    standstill_gap_m: float = 4.0      # minimum bumper-to-bumper gap at standstill [m]
    headway_time_s: float = 0.5        # time-gap headway [s]

    # LQR weights (state cost)
    q_gap: float = 1.0       # gap error weight         (e_gap)
    q_rel_vel: float = 2.0   # relative-velocity weight (e_rel_vel)
    q_speed: float = 0.5     # absolute-speed weight    (e_speed)

    # LQR weight (control cost)
    r_accel: float = 0.5     # control-effort weight

    # Model time step used when solving Riccati equation [s]
    # Should match the simulation step (MetaDrive default = 0.1 s)
    lqr_dt: float = 0.1

    # LQR output saturation [m/s²] before throttle mapping
    accel_clip_mps2: float = 3.0   # max forward acceleration command
    decel_clip_mps2: float = 4.0   # max deceleration command (positive magnitude)

    # Throttle normalisation: throttle=±1 corresponds to these accelerations
    max_accel_mps2: float = 3.0
    max_decel_mps2: float = 4.0

    # Fallback vehicle length when not available from vehicle object [m]
    vehicle_length_m: float = 5.74

    # ID of the leader agent (controls only by speed tracking)
    leader_agent_id: str = "agent0"
    leader_longitudinal_mode: str = "idm"  # "idm" | "speed_lqr"
    leader_allow_lane_change: bool = False
    leader_idm_min_gap_m: float = 10.0
    leader_idm_headway_time_s: float = 1.2
    leader_idm_accel_clip_mps2: float = 3.0
    leader_idm_decel_clip_mps2: float = 4.5
    leader_idm_delta: float = 4.0

    # --- Lateral control (dual-PID, matching IDM steering_control) ---

    # Heading PID gains (yaw alignment)
    heading_pid_kp: float = 1.7
    heading_pid_ki: float = 0.01
    heading_pid_kd: float = 3.5

    # Lateral PID gains (cross-track error)
    lateral_pid_kp: float = 0.3
    lateral_pid_ki: float = 0.002
    lateral_pid_kd: float = 0.05

    # Lookahead distance for lane heading estimate [m]
    lateral_lookahead_m: float = 1.0

    # Steering output clip (before applying to env)
    max_steering: float = 1.0


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class PlatoonAgentState(NamedTuple):
    """Per-agent observable state and applied control — used for logging / imitation."""
    agent_id: str
    speed_mps: float
    gap_to_front_m: float      # bumper-to-bumper distance; float('inf') for leader
    e_gap: float               # gap error  d_actual - d_desired;  0 for leader
    e_rel_vel: float           # v_front - v_ego;                  0 for leader
    e_speed: float             # v_desired - v_ego
    accel_cmd_mps2: float      # raw LQR output (before clipping)
    accel_applied_mps2: float  # after clipping
    throttle: float            # longitudinal output in [-1, 1]
    steering: float            # lateral output in [-1, 1]
    lateral_error_m: float     # cross-track error [m] (+ = left of lane center)
    heading_error_rad: float   # heading error [rad]  (+ = lane heading > vehicle heading)
    longitudinal_mode: str     # "leader_idm" | "leader_speed_lqr" | "follower_lqr"


class PlatoonExpertOutput(NamedTuple):
    """Full output from one compute_actions_with_state() call."""
    actions: dict        # {agent_id: np.array([steering, throttle], float32)}
    agent_states: dict   # {agent_id: PlatoonAgentState}


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class PlatoonLQRExpert:
    """
    Platoon expert: longitudinal LQR + lateral dual-PID (IDM-style lane tracking).

    Parameters
    ----------
    config : PlatoonLQRConfig, optional
        If None, uses default configuration.

    Notes
    -----
    Longitudinal control is stateless (reads env.agents each call).
    Lateral PID controllers are stateful (integrators). Call reset() at each
    episode boundary to clear PID integrator state.

    If a vehicle has no .lane attribute (e.g. in unit tests with fake vehicles),
    steering falls back to 0.0 gracefully — longitudinal control is unaffected.
    """

    def __init__(self, config: Optional[PlatoonLQRConfig] = None) -> None:
        self.config = config or PlatoonLQRConfig()
        self._K_leader, self._K_follower = self._solve_gains()
        self._lateral: dict[str, _AgentLateralState] = {}

    # ------------------------------------------------------------------
    # Gain computation
    # ------------------------------------------------------------------

    def _solve_gains(self) -> tuple[np.ndarray, np.ndarray]:
        dt = float(self.config.lqr_dt)
        q_g = float(self.config.q_gap)
        q_v = float(self.config.q_rel_vel)
        q_s = float(self.config.q_speed)
        r = float(self.config.r_accel)

        # Leader: 1-state   x = [e_speed],  model: x[t+1] = x[t] - dt*u
        K_leader = _dare_gain(
            A=np.array([[1.0]]),
            B=np.array([[-dt]]),
            Q=np.array([[q_s]]),
            R=np.array([[r]]),
        )  # shape (1, 1)

        # Follower: 3-state  x = [e_gap, e_rel_vel, e_speed]
        K_follower = _dare_gain(
            A=np.array([[1.0, dt, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]]),
            B=np.array([[0.0], [-dt], [-dt]]),
            Q=np.diag([q_g, q_v, q_s]),
            R=np.array([[r]]),
        )  # shape (1, 3)

        return K_leader, K_follower

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, agent_id: Optional[str] = None) -> None:
        """Reset lateral PID integrators. Call at each episode boundary.

        Parameters
        ----------
        agent_id : str, optional
            If given, reset only that agent's PID state. If None, reset all.
        """
        if agent_id is None:
            self._lateral.clear()
        else:
            ls = self._lateral.get(agent_id)
            if ls is not None:
                ls.reset()

    def compute_actions(self, env) -> dict[str, np.ndarray]:
        """
        Compute actions for all active platoon agents.

        Returns
        -------
        dict[str, np.ndarray]
            ``{agent_id: np.array([steering, throttle], dtype=float32)}``
            ready to pass directly to ``PlatoonEnv.low_level_step()``.
        """
        return self.compute_actions_with_state(env).actions

    def compute_actions_with_state(self, env) -> PlatoonExpertOutput:
        """
        Compute actions and return full diagnostic state.

        Returns
        -------
        PlatoonExpertOutput
            ``.actions``      — dict ready for ``low_level_step``
            ``.agent_states`` — dict of ``PlatoonAgentState`` for logging/imitation
        """
        agents = getattr(env, "agents", {}) or {}
        if not agents:
            return PlatoonExpertOutput(actions={}, agent_states={})

        agent_ids = _sort_agent_ids(list(agents.keys()))
        v_desired_mps = float(self.config.desired_speed_km_h) / 3.6

        speeds:    dict[str, float]      = {}
        positions: dict[str, np.ndarray] = {}
        lengths:   dict[str, float]      = {}
        for aid in agent_ids:
            veh = agents[aid]
            speeds[aid]    = float(veh.speed_km_h) / 3.6
            positions[aid] = np.asarray(veh.position[:2], dtype=np.float64)
            lengths[aid]   = float(getattr(veh, "LENGTH", self.config.vehicle_length_m))

        actions: dict[str, np.ndarray]        = {}
        states:  dict[str, PlatoonAgentState] = {}

        for i, aid in enumerate(agent_ids):
            veh     = agents[aid]
            v_ego   = speeds[aid]
            e_speed = v_desired_mps - v_ego

            if aid == self.config.leader_agent_id or i == 0:
                if self.config.leader_longitudinal_mode == "idm":
                    accel_raw, gap, e_gap, e_rv = self._compute_leader_longitudinal_idm(
                        env=env,
                        leader=veh,
                        leader_id=aid,
                        leader_speed_mps=v_ego,
                        desired_speed_mps=v_desired_mps,
                        lengths=lengths,
                    )
                    longitudinal_mode = "leader_idm"
                else:
                    # ---- Leader: 1-state LQR ----
                    x = np.array([e_speed])
                    accel_raw = float((-self._K_leader @ x).item())
                    gap   = float("inf")
                    e_gap = 0.0
                    e_rv  = 0.0
                    longitudinal_mode = "leader_speed_lqr"
            else:
                # ---- Follower: 3-state LQR ----
                front_id  = agent_ids[i - 1]
                v_front   = speeds[front_id]
                pos_ego   = positions[aid]
                pos_front = positions[front_id]

                center_dist = float(np.linalg.norm(pos_front - pos_ego))
                half_sum    = 0.5 * (lengths[aid] + lengths[front_id])
                gap         = max(center_dist - half_sum, 0.0)

                d_desired = (float(self.config.standstill_gap_m)
                             + float(self.config.headway_time_s) * v_ego)
                e_gap = gap - d_desired
                e_rv  = v_front - v_ego

                x = np.array([e_gap, e_rv, e_speed])
                accel_raw = float((-self._K_follower @ x).item())
                longitudinal_mode = "follower_lqr"

            # Clip and normalise to throttle ∈ [-1, 1]
            accel_clip_pos = float(self.config.accel_clip_mps2)
            accel_clip_neg = float(self.config.decel_clip_mps2)
            if longitudinal_mode == "leader_idm":
                accel_clip_pos = float(self.config.leader_idm_accel_clip_mps2)
                accel_clip_neg = float(self.config.leader_idm_decel_clip_mps2)
            accel_applied = float(np.clip(
                accel_raw,
                -accel_clip_neg,
                accel_clip_pos,
            ))
            throttle = _accel_to_throttle(
                accel_applied,
                float(self.config.max_accel_mps2),
                float(self.config.max_decel_mps2),
            )

            # Lateral control
            ls = self._lateral.setdefault(aid, _AgentLateralState(self.config))
            steering, lat_err, hdg_err = self._compute_steering(veh, ls)

            actions[aid] = np.array([steering, throttle], dtype=np.float32)
            states[aid]  = PlatoonAgentState(
                agent_id=aid,
                speed_mps=v_ego,
                gap_to_front_m=gap,
                e_gap=e_gap,
                e_rel_vel=e_rv,
                e_speed=e_speed,
                accel_cmd_mps2=accel_raw,
                accel_applied_mps2=accel_applied,
                throttle=throttle,
                steering=steering,
                lateral_error_m=lat_err,
                heading_error_rad=hdg_err,
                longitudinal_mode=longitudinal_mode,
            )

        return PlatoonExpertOutput(actions=actions, agent_states=states)

    def _compute_leader_longitudinal_idm(
        self,
        env,
        leader,
        leader_id: str,
        leader_speed_mps: float,
        desired_speed_mps: float,
        lengths: dict[str, float],
    ) -> tuple[float, float, float, float]:
        front = self._find_front_object_for_leader(env, leader, leader_id)
        if front is None:
            x = np.array([desired_speed_mps - leader_speed_mps])
            accel = float((-self._K_leader @ x).item())
            return accel, float("inf"), 0.0, 0.0

        front_vehicle, front_long_dist = front
        front_speed_mps = _vehicle_speed_mps(front_vehicle)
        front_length = float(getattr(front_vehicle, "LENGTH", self.config.vehicle_length_m))
        half_sum = 0.5 * (lengths.get(leader_id, self.config.vehicle_length_m) + front_length)
        gap = max(float(front_long_dist) - half_sum, 0.0)

        min_gap = float(self.config.leader_idm_min_gap_m)
        headway = float(self.config.leader_idm_headway_time_s)
        accel_max = float(self.config.leader_idm_accel_clip_mps2)
        decel_max = max(float(self.config.leader_idm_decel_clip_mps2), 1e-6)
        delta = float(self.config.leader_idm_delta)

        rel_speed = float(leader_speed_mps - front_speed_mps)
        dynamic_gap = min_gap + max(leader_speed_mps, 0.0) * headway
        braking_term = (leader_speed_mps * rel_speed) / (2.0 * np.sqrt(accel_max * decel_max))
        desired_gap = max(min_gap, dynamic_gap + max(braking_term, 0.0))
        safe_gap = max(gap, 0.1)

        cruise_term = 1.0 - np.power(max(leader_speed_mps, 0.0) / max(desired_speed_mps, 1e-3), delta)
        interaction_term = np.power(desired_gap / safe_gap, 2.0)
        accel = accel_max * (cruise_term - interaction_term)
        e_gap = gap - desired_gap
        e_rel_vel = front_speed_mps - leader_speed_mps
        return float(accel), gap, float(e_gap), float(e_rel_vel)

    def _find_front_object_for_leader(self, env, leader, leader_id: str):
        lane = getattr(leader, "lane", None)
        if lane is None:
            return None
        try:
            leader_long, _leader_lat = lane.local_coordinates(np.asarray(leader.position[:2], dtype=np.float64))
        except Exception:
            return None

        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        traffic_vehicles = list(getattr(traffic_manager, "_traffic_vehicles", []) or [])
        if not traffic_vehicles:
            return None

        best_obj = None
        best_delta_long = None
        for obj in traffic_vehicles:
            if obj is leader or getattr(obj, "name", None) == leader_id:
                continue
            obj_lane = getattr(obj, "lane", None)
            if obj_lane is not lane:
                continue
            try:
                obj_long, _obj_lat = lane.local_coordinates(np.asarray(obj.position[:2], dtype=np.float64))
            except Exception:
                continue
            delta_long = float(obj_long - leader_long)
            if delta_long <= 0.0:
                continue
            if best_delta_long is None or delta_long < best_delta_long:
                best_delta_long = delta_long
                best_obj = (obj, delta_long)
        return best_obj

    # ------------------------------------------------------------------
    # Lateral control
    # ------------------------------------------------------------------

    def _compute_steering(
        self,
        veh,
        lat_state: _AgentLateralState,
    ) -> tuple[float, float, float]:
        """Dual-PID steering from vehicle lane reference.

        Returns (steering, lateral_error_m, heading_error_rad).
        Falls back to (0.0, 0.0, 0.0) if veh has no .lane attribute.
        """
        lane = getattr(veh, "lane", None)
        if lane is None:
            return 0.0, 0.0, 0.0

        pos  = np.asarray(veh.position[:2], dtype=np.float64)
        long, lat = lane.local_coordinates(pos)
        lane_hdg  = lane.heading_theta_at(long + self.config.lateral_lookahead_m)
        v_hdg     = float(getattr(veh, "heading_theta", 0.0))
        hdg_err   = _wrap_to_pi(lane_hdg - v_hdg)

        steer  = lat_state.heading_pid.get_result(-hdg_err)
        steer += lat_state.lateral_pid.get_result(-lat)
        steer  = float(np.clip(steer, -self.config.max_steering, self.config.max_steering))
        return steer, float(lat), float(hdg_err)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def K_leader(self) -> np.ndarray:
        """LQR gain for leader: u = -K_leader @ [e_speed], shape (1, 1)."""
        return self._K_leader.copy()

    @property
    def K_follower(self) -> np.ndarray:
        """LQR gain for follower: u = -K_follower @ [e_gap, e_rel_vel, e_speed], shape (1, 3)."""
        return self._K_follower.copy()

    def desired_gap_m(self, speed_mps: float) -> float:
        """Compute desired bumper-to-bumper gap at given speed."""
        return float(self.config.standstill_gap_m) + float(self.config.headway_time_s) * speed_mps


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sort_agent_ids(agent_ids: list[str]) -> list[str]:
    """Sort agent IDs by numeric suffix: agent0 < agent1 < ... < agent10."""
    def _key(k: str) -> int:
        suffix = k.replace("agent", "")
        return int(suffix) if suffix.isdigit() else hash(k)
    try:
        return sorted(agent_ids, key=_key)
    except Exception:
        return sorted(agent_ids)


def _dare_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Solve discrete Riccati equation and return gain K (u* = -K @ x).

    Uses scipy.linalg.solve_discrete_are when available; falls back to
    Riccati iteration otherwise.
    """
    try:
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return K.astype(np.float64)
    except Exception:
        return _dare_gain_iteration(A, B, Q, R)


def _dare_gain_iteration(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    """Riccati iteration fallback (no scipy dependency)."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    Q = Q.astype(np.float64)
    R = R.astype(np.float64)
    P = Q.copy()
    for _ in range(max_iter):
        BtP  = B.T @ P
        schur = np.linalg.inv(R + BtP @ B)
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ schur @ BtP @ A
        if np.max(np.abs(P_new - P)) < tol:
            break
        P = P_new
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K.astype(np.float64)


def _accel_to_throttle(accel_mps2: float, max_accel: float, max_decel: float) -> float:
    """Map acceleration [m/s²] to throttle ∈ [-1, 1]."""
    if accel_mps2 >= 0.0:
        return float(np.clip(accel_mps2 / max(max_accel, 1e-6), 0.0, 1.0))
    else:
        return float(np.clip(accel_mps2 / max(max_decel, 1e-6), -1.0, 0.0))


def _vehicle_speed_mps(vehicle) -> float:
    """Best-effort vehicle speed extraction in m/s."""
    if hasattr(vehicle, "speed"):
        try:
            return float(vehicle.speed)
        except Exception:
            pass
    if hasattr(vehicle, "speed_km_h"):
        try:
            return float(vehicle.speed_km_h) / 3.6
        except Exception:
            pass
    return 0.0
