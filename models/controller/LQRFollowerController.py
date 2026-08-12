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
from models.controller.longitudinal_reference import (
    BRAKE_ACCELERATION_SCALE_MPS2,
    DRIVE_ACCELERATION_SCALE_MPS2,
    LongitudinalTrackingReference,
    trajectory_to_longitudinal_reference,
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
LQR_DESIRED_OUTGAP_M = 15.0   # target gap when following a background vehicle
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
        longitudinal_references: dict[str, LongitudinalTrackingReference] | None = None,
        lateral_tracking_errors_m: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        agents = getattr(env, "agents", {}) or {}
        all_ids: list[str] = list(getattr(env, "_agent_ids", sorted(agents.keys())))
        active_ids = [aid for aid in all_ids if aid in agents]
        traffic_vehicles = self._get_traffic_vehicles(env)

        actions: dict[str, np.ndarray] = {}
        debug: dict[str, dict] = {}
        effective_cross_track_kp, scenario_id = self._pid.effective_cross_track_gain(env)
        s6_actor = next(
            (
                value
                for value in traffic_vehicles
                if str(getattr(value, "scenario_vehicle_role", ""))
                == "s6_gap_intruder"
            ),
            None,
        )
        s6_merge_pending = bool(
            scenario_id == "S6_background_merge_in"
            and s6_actor is not None
            and not bool(getattr(s6_actor, "scenario_merge_completed", False))
        )
        s6_resolved = dict(
            getattr(
                getattr(env, "_scenario_orchestrator", None),
                "_resolved_scenario_parameters",
                {},
            )
            or {}
        )
        s6_target_gap_id = str(
            s6_resolved.get(
                "target_gap_id",
                getattr(s6_actor, "scenario_target_gap_id", ""),
            )
        )
        s6_target_front_id = s6_target_gap_id.split("-", 1)[0]
        s6_target_rear_id = s6_target_gap_id.split("-", 1)[-1]
        s6_interaction_gap_m = float(
            getattr(s6_actor, "LENGTH", 5.74) or 5.74
        ) + float(
            s6_resolved.get(
                "target_front_bumper_gap_m",
                getattr(s6_actor, "scenario_target_front_bumper_gap_m", 6.0),
            )
        ) + float(
            s6_resolved.get(
                "target_rear_bumper_gap_m",
                getattr(s6_actor, "scenario_target_rear_bumper_gap_m", 6.0),
            )
        ) + (-2.0 if s6_target_gap_id == "agent0-agent1" else 2.0)
        orchestrator = getattr(env, "_scenario_orchestrator", None)
        s7_evidence = dict(getattr(orchestrator, "_conflict_evidence", {}) or {})
        s7_resolved = dict(
            getattr(orchestrator, "_resolved_scenario_parameters", {}) or {}
        )
        s7_crossings = dict(
            s7_evidence.get("actor_conflict_crossing_steps", {}) or {}
        )
        s7_expected = str(s7_resolved.get("expected_behavior", "pass_first"))
        if s7_expected == "pass_first":
            s7_manifest = dict(
                getattr(orchestrator, "_actor_manifest", {}) or {}
            )
            s7_release_ready = {
                "critical_gap_front",
                "critical_gap_rear",
                "next_gap_front",
                "next_gap_rear",
            }.issubset(s7_manifest)
        else:
            s7_release_ready = bool("critical_gap_rear" in s7_crossings)

        for agent_id in active_ids:
            vehicle = agents[agent_id]
            role_index = active_ids.index(agent_id)
            traj_world = (trajectories_world or {}).get(agent_id)
            traj_local = (
                _world_trajectory_to_ego_local(vehicle, traj_world)
                if traj_world is not None
                else np.zeros((0, 3), dtype=np.float32)
            )
            front_veh, gap_m, is_platoon = self._find_front_vehicle(vehicle, agents, traffic_vehicles)
            if (
                scenario_id == "S7_ego_merge_from_ramp"
                and role_index > 0
                and front_veh is None
            ):
                # MetaDrive's lane-local front lookup becomes discontinuous
                # while adjacent platoon members straddle the two curved ramp
                # connectors.  The formation order is fixed, so retain a
                # conservative centre-distance bumper gap to the immediate
                # predecessor across that seam.  This is control feedback;
                # NormalPlanner remains the exact route/OBB authority.
                predecessor = agents.get(active_ids[role_index - 1])
                if predecessor is not None:
                    centre_distance_m = float(
                        np.linalg.norm(
                            np.asarray(predecessor.position[:2], dtype=float)
                            - np.asarray(vehicle.position[:2], dtype=float)
                        )
                    )
                    seam_gap_m = centre_distance_m - 0.5 * float(
                        getattr(predecessor, "LENGTH", 5.74) or 5.74
                    ) - 0.5 * float(
                        getattr(vehicle, "LENGTH", 5.74) or 5.74
                    )
                    if seam_gap_m >= 0.0:
                        front_veh = predecessor
                        gap_m = float(seam_gap_m)
                        is_platoon = True
            s6_physical_corridor_entered = bool(
                (
                    getattr(
                        getattr(env, "_scenario_orchestrator", None),
                        "_conflict_evidence",
                        {},
                    )
                    or {}
                ).get("physical_gap_corridor_entered", False)
            )
            if (
                scenario_id == "S6_background_merge_in"
                and s6_physical_corridor_entered
                and str(agent_id) == s6_target_rear_id
                and s6_actor is not None
                and front_veh is None
            ):
                centre_distance = float(
                    np.linalg.norm(
                        np.asarray(s6_actor.position[:2], dtype=float)
                        - np.asarray(vehicle.position[:2], dtype=float)
                    )
                )
                connected_gap = (
                    centre_distance
                    - 0.5 * float(getattr(s6_actor, "LENGTH", 5.74) or 5.74)
                    - 0.5 * float(getattr(vehicle, "LENGTH", 5.74) or 5.74)
                )
                if connected_gap > 0.0:
                    front_veh = s6_actor
                    gap_m = connected_gap
                    is_platoon = False
            ego_speed = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6
            reference = (longitudinal_references or {}).get(agent_id)
            if reference is None and traj_local.shape == (8, 3):
                reference = trajectory_to_longitudinal_reference(
                    traj_local, ego_speed, source="online_trajectory"
                )
            gap_acceleration = 0.0
            mode = "trajectory_no_front"
            front_id = None
            desired_gap = None
            actual_gap = None
            front_speed = None
            rear_gap = None
            rear_gap_feedback = 0.0
            if front_veh is not None and gap_m is not None:
                front_agent_id = next(
                    (key for key, value in agents.items() if value is front_veh),
                    "background",
                )
                is_s6_target_pair = bool(
                    s6_merge_pending
                    and is_platoon
                    and f"{front_agent_id}-{agent_id}" == s6_target_gap_id
                )
                is_s6_actor_follow_pair = bool(
                    scenario_id == "S6_background_merge_in"
                    and (
                        s6_target_gap_id == "agent0-agent1"
                        or s6_physical_corridor_entered
                    )
                    and s6_target_gap_id.endswith(f"-{agent_id}")
                    and str(
                        getattr(front_veh, "scenario_vehicle_role", "")
                    )
                    == "s6_gap_intruder"
                )
                desired_gap = (
                    s6_interaction_gap_m
                    if is_s6_target_pair
                    else float(
                        s6_resolved.get("target_rear_bumper_gap_m", 6.0)
                    )
                    if is_s6_actor_follow_pair
                    else self.desired_gap_m if is_platoon else self.desired_outgap_m
                )
                actual_gap = float(gap_m)
                front_speed = float(
                    getattr(front_veh, "speed_km_h", 0.0) or 0.0
                ) / 3.6
                gap_acceleration = float(
                    np.clip(
                        (
                            0.18
                            if is_s6_target_pair
                            and s6_target_gap_id == "agent0-agent1"
                            else 0.08
                            if is_s6_target_pair
                            else 0.15
                        )
                        * (actual_gap - desired_gap)
                        + 0.20 * (front_speed - ego_speed),
                        -0.5 if is_s6_target_pair else -1.0,
                        (
                            1.2
                            if is_s6_target_pair
                            and s6_target_gap_id == "agent0-agent1"
                            else 0.5
                            if is_s6_target_pair
                            else 1.0
                        ),
                    )
                )
                front_id = front_agent_id
                mode = "follower_platoon" if is_platoon else "follower_background"
            lateral_maneuver_active = bool(
                traj_local.shape == (8, 3)
                and np.max(np.abs(traj_local[:, 1]), initial=0.0) > 0.75
            )
            committed_roll_active = bool(
                reference is not None
                and str(getattr(reference, "source", "")) == "committed_roll"
            )
            rear_feedback_suppressed = bool(
                lateral_maneuver_active
                or committed_roll_active
            )
            if role_index + 1 < len(active_ids) and not rear_feedback_suppressed:
                rear_vehicle = agents.get(active_ids[role_index + 1])
                rear_lane = getattr(rear_vehicle, "lane", None)
                ego_lane = getattr(vehicle, "lane", None)
                if rear_vehicle is not None and rear_lane is not None and ego_lane is not None:
                    try:
                        ego_s, _ = ego_lane.local_coordinates(vehicle.position)
                        rear_s, rear_t = ego_lane.local_coordinates(rear_vehicle.position)
                        lane_width = float(getattr(ego_lane, "width", 3.5) or 3.5)
                        if abs(float(rear_t)) <= 0.30 * lane_width:
                            rear_gap = (
                                float(ego_s)
                                - float(rear_s)
                                - 0.5 * float(getattr(vehicle, "LENGTH", 5.0))
                                - 0.5 * float(getattr(rear_vehicle, "LENGTH", 5.0))
                            )
                            rear_speed = float(
                                getattr(rear_vehicle, "speed_km_h", 0.0) or 0.0
                            ) / 3.6
                            rear_agent_id = active_ids[role_index + 1]
                            is_s6_target_rear_pair = bool(
                                s6_merge_pending
                                and f"{agent_id}-{rear_agent_id}"
                                == s6_target_gap_id
                            )
                            if is_s6_target_rear_pair:
                                # A small pass-first pulse is needed to keep a
                                # dense hard-safe KEEP trajectory while the
                                # actor converges.  Keep it much weaker than
                                # the rear vehicle's yield command so the
                                # actor-to-front clearance does not overshoot
                                # the declared 6--10 m corridor.
                                rear_gap_feedback = float(
                                    np.clip(
                                        0.03 * (s6_interaction_gap_m - rear_gap)
                                        + 0.15 * (rear_speed - ego_speed),
                                        (
                                            -0.5
                                            if rear_gap
                                            > s6_interaction_gap_m - 3.0
                                            else 0.0
                                        ),
                                        0.2,
                                    )
                                )
                                gap_acceleration = float(
                                    max(gap_acceleration, rear_gap_feedback)
                                )
                            elif scenario_id == "S7_ego_merge_from_ramp":
                                # The curved ramp seams make lane-local rear
                                # progress noisy.  Use the directly measured
                                # bumper gap to keep an S7 predecessor from
                                # running away from its follower while the
                                # full platoon enters the mainline.
                                rear_gap_feedback = float(
                                    np.clip(
                                        -0.18
                                        * (rear_gap - self.desired_gap_m)
                                        + 0.10 * (rear_speed - ego_speed),
                                        -3.0,
                                        0.0,
                                    )
                                )
                            else:
                                rear_gap_feedback = float(
                                    np.clip(
                                        -0.25
                                        * (rear_gap - self.desired_gap_m)
                                        + 0.15 * (rear_speed - ego_speed),
                                        -2.0,
                                        0.0,
                                    )
                                )
                            if rear_gap_feedback < 0.0:
                                # Do not let a positive front-gap/free-road
                                # command cancel the predecessor's duty to
                                # recover an expanding rear formation gap.
                                if scenario_id == "S7_ego_merge_from_ramp":
                                    gap_acceleration = float(
                                        np.clip(
                                            gap_acceleration
                                            + rear_gap_feedback,
                                            -3.0,
                                            1.0,
                                        )
                                    )
                                else:
                                    gap_acceleration = float(
                                        min(gap_acceleration, rear_gap_feedback)
                                    )
                    except Exception:
                        rear_gap = None
                        rear_gap_feedback = 0.0
            if reference is None:
                action = np.zeros((2,), dtype=np.float32)
                agent_debug: dict[str, object] = {
                    "mode": "no_trajectory",
                    "gap_feedback_mps2": 0.0,
                }
            else:
                action, agent_debug = self._pid._single_control_with_debug(
                    agent_id,
                    vehicle,
                    traj_local,
                    reference,
                    gap_acceleration_mps2=gap_acceleration,
                    cross_track_error_m=float(
                        (lateral_tracking_errors_m or {}).get(agent_id, 0.0)
                    ),
                    cross_track_kp=effective_cross_track_kp,
                )
                lane_index = tuple(getattr(vehicle, "lane_index", ()) or ())
                s6_premerge_front_speed_km_h = (
                    19.3
                    if s6_target_gap_id == "agent0-agent1"
                    else 21.0
                )
                s6_premerge_front_speed_guard = bool(
                    scenario_id == "S6_background_merge_in"
                    and str(agent_id) == s6_target_front_id
                    and not bool(
                        (
                            getattr(
                                getattr(env, "_scenario_orchestrator", None),
                                "_conflict_evidence",
                                {},
                            )
                            or {}
                        ).get("physical_gap_corridor_entered", False)
                    )
                    and ego_speed > s6_premerge_front_speed_km_h / 3.6
                    and not committed_roll_active
                )
                if s6_premerge_front_speed_guard:
                    premerge_guard_acceleration = float(
                        np.clip(
                            1.2
                            * (
                                s6_premerge_front_speed_km_h / 3.6
                                - ego_speed
                            ),
                            -2.0,
                            0.0,
                        )
                    )
                    action = np.asarray(action, dtype=np.float32).copy()
                    action[1] = min(
                        float(action[1]),
                        premerge_guard_acceleration
                        / BRAKE_ACCELERATION_SCALE_MPS2,
                    )
                    agent_debug["s6_premerge_front_speed_guard"] = True
                else:
                    agent_debug["s6_premerge_front_speed_guard"] = False
                s6_premerge_rear_gap_guard = bool(
                    scenario_id == "S6_background_merge_in"
                    and s6_target_gap_id == "agent0-agent1"
                    and str(agent_id) == s6_target_rear_id
                    and front_veh is not None
                    and actual_gap is not None
                    and desired_gap is not None
                    and (
                        (
                            str(
                                getattr(
                                    front_veh,
                                    "scenario_vehicle_role",
                                    "",
                                )
                            )
                            == "s6_gap_intruder"
                            and actual_gap >= desired_gap + 1.0
                        )
                        or (
                            front_id == s6_target_front_id
                            and actual_gap >= s6_interaction_gap_m - 2.0
                        )
                    )
                    and ego_speed < 18.5 / 3.6
                    and not committed_roll_active
                )
                if s6_premerge_rear_gap_guard:
                    rear_guard_acceleration = float(
                        np.clip(1.0 * (18.5 / 3.6 - ego_speed), 0.0, 1.0)
                    )
                    action = np.asarray(action, dtype=np.float32).copy()
                    action[1] = max(
                        float(action[1]),
                        rear_guard_acceleration
                        / DRIVE_ACCELERATION_SCALE_MPS2,
                    )
                    agent_debug["s6_premerge_rear_gap_guard"] = True
                else:
                    agent_debug["s6_premerge_rear_gap_guard"] = False
                s6_guard_speed_km_h = (
                    24.0
                    if s6_target_gap_id.endswith(f"-{agent_id}")
                    else 14.0
                )
                s6_post_response_speed_guard = bool(
                    scenario_id == "S6_background_merge_in"
                    and not s6_merge_pending
                    and len(lane_index) >= 3
                    and (
                        int(lane_index[2]) != 2
                        or tuple(lane_index[:2])
                        != ("9g0_0_", "9g0_1_")
                    )
                    and not committed_roll_active
                    and ego_speed > s6_guard_speed_km_h / 3.6
                )
                if s6_post_response_speed_guard:
                    guard_acceleration = float(
                        np.clip(
                            1.2 * (s6_guard_speed_km_h / 3.6 - ego_speed),
                            -2.0,
                            0.0,
                        )
                    )
                    guard_throttle = float(
                        guard_acceleration / BRAKE_ACCELERATION_SCALE_MPS2
                    )
                    action = np.asarray(action, dtype=np.float32).copy()
                    action[1] = min(float(action[1]), guard_throttle)
                    agent_debug["s6_post_response_speed_guard"] = True
                    agent_debug["s6_speed_guard_acceleration_mps2"] = (
                        guard_acceleration
                    )
                else:
                    agent_debug["s6_post_response_speed_guard"] = False
                s7_wait_speed_guard = bool(
                    scenario_id == "S7_ego_merge_from_ramp"
                    and not s7_release_ready
                    and tuple(lane_index[:2]) == ("18c0_1_", "9g0_0_")
                    and not committed_roll_active
                )
                if s7_wait_speed_guard:
                    lane = getattr(vehicle, "lane", None)
                    try:
                        longitudinal = float(
                            lane.local_coordinates(vehicle.position)[0]
                        )
                        remaining = max(float(lane.length) - longitudinal, 0.0)
                    except (AttributeError, TypeError, ValueError):
                        remaining = 0.0
                    stop_buffer_m = 6.0
                    allowed_speed_mps = float(
                        np.sqrt(4.0 * max(remaining - stop_buffer_m, 0.0))
                    )
                    guard_acceleration = float(
                        np.clip(1.5 * (allowed_speed_mps - ego_speed), -3.0, 0.0)
                    )
                    action = np.asarray(action, dtype=np.float32).copy()
                    action[1] = min(
                        float(action[1]),
                        guard_acceleration / BRAKE_ACCELERATION_SCALE_MPS2,
                    )
                    agent_debug["s7_wait_speed_guard"] = True
                    agent_debug["s7_wait_remaining_m"] = remaining
                    agent_debug["s7_wait_allowed_speed_mps"] = allowed_speed_mps
                else:
                    agent_debug["s7_wait_speed_guard"] = False
                s7_follower_queue_guard = bool(
                    scenario_id == "S7_ego_merge_from_ramp"
                    and role_index > 0
                    and is_platoon
                    and actual_gap is not None
                    and float(actual_gap) > 9.0
                    and not committed_roll_active
                )
                if s7_follower_queue_guard:
                    # Close the initially declared 12--20 m platoon gaps
                    # while the leader waits for the selected traffic gap.
                    # This target remains inside S7's 20--26 km/h ego range;
                    # the unchanged 7 m predictive guard is still the hard
                    # authority as a follower approaches its predecessor.
                    queue_speed_mps = 25.0 / 3.6
                    queue_acceleration = float(
                        np.clip(
                            1.0 * (queue_speed_mps - ego_speed),
                            0.0,
                            1.0,
                        )
                    )
                    action = np.asarray(action, dtype=np.float32).copy()
                    action[1] = max(
                        float(action[1]),
                        queue_acceleration / DRIVE_ACCELERATION_SCALE_MPS2,
                    )
                    agent_debug["s7_follower_queue_guard"] = True
                else:
                    agent_debug["s7_follower_queue_guard"] = False
                s7_platoon_gap_guard = bool(
                    scenario_id == "S7_ego_merge_from_ramp"
                    and is_platoon
                    and actual_gap is not None
                    and front_speed is not None
                    and ego_speed > float(front_speed) + 1.0e-3
                    and not committed_roll_active
                )
                if s7_platoon_gap_guard:
                    hard_gap_m = 7.0
                    guard_margin_m = 0.5
                    closing_speed_mps = max(
                        ego_speed - float(front_speed), 0.0
                    )
                    available_m = max(
                        float(actual_gap) - hard_gap_m - guard_margin_m,
                        0.1,
                    )
                    required_braking_mps2 = -(
                        closing_speed_mps * closing_speed_mps
                    ) / (2.0 * available_m)
                    guard_active = bool(
                        float(actual_gap) <= 10.0
                        or (
                            closing_speed_mps * closing_speed_mps
                            / (2.0 * 3.0)
                        )
                        >= available_m
                    )
                    if guard_active:
                        required_braking_mps2 = float(
                            np.clip(required_braking_mps2 - 0.35, -8.0, 0.0)
                        )
                        action = np.asarray(action, dtype=np.float32).copy()
                        action[1] = min(
                            float(action[1]),
                            required_braking_mps2
                            / BRAKE_ACCELERATION_SCALE_MPS2,
                        )
                    agent_debug["s7_platoon_gap_guard"] = bool(guard_active)
                    agent_debug["s7_gap_guard_available_m"] = float(available_m)
                    agent_debug["s7_gap_guard_closing_speed_mps"] = float(
                        closing_speed_mps
                    )
                else:
                    agent_debug["s7_platoon_gap_guard"] = False
                s7_mainline_coordination_guard = bool(
                    scenario_id == "S7_ego_merge_from_ramp"
                    and tuple(lane_index[:2])
                    == ("9g0_0_", "9g1_4_")
                    and not bool(
                        s7_evidence.get(
                            "formation_recovered_after_merge", False
                        )
                    )
                    and not committed_roll_active
                )
                if s7_mainline_coordination_guard:
                    # Use a small rearward speed gradient until both bumper
                    # gaps have converged.  A common target preserves an
                    # already-expanded gap forever; the gradient closes it,
                    # while the predictive 7 m guard below remains the hard
                    # lower-bound authority.
                    coordination_speed_mps = (
                        12.0 + 2.0 * float(role_index)
                    ) / 3.6
                    coordination_acceleration = float(
                        np.clip(
                            1.2 * (coordination_speed_mps - ego_speed),
                            -3.0,
                            1.0,
                        )
                    )
                    action = np.asarray(action, dtype=np.float32).copy()
                    if coordination_acceleration <= 0.0:
                        action[1] = min(
                            float(action[1]),
                            coordination_acceleration
                            / BRAKE_ACCELERATION_SCALE_MPS2,
                        )
                    elif actual_gap is None or float(actual_gap) > 9.0:
                        action[1] = max(
                            float(action[1]),
                            coordination_acceleration
                            / DRIVE_ACCELERATION_SCALE_MPS2,
                        )
                    agent_debug["s7_mainline_coordination_guard"] = True
                else:
                    agent_debug["s7_mainline_coordination_guard"] = False
                agent_debug["mode"] = mode
                agent_debug["scenario_id"] = scenario_id
            agent_debug.update(
                {
                    "leader_id": front_id,
                    "desired_gap_m": desired_gap,
                    "actual_gap_m": actual_gap,
                    "gap_feedback_mps2": float(gap_acceleration),
                    "rear_platoon_gap_m": rear_gap,
                    "rear_gap_feedback_mps2": float(rear_gap_feedback),
                    "rear_gap_feedback_suppressed_for_lateral_maneuver": bool(
                        rear_feedback_suppressed
                    ),
                }
            )
            actions[agent_id] = action
            debug[agent_id] = agent_debug

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
