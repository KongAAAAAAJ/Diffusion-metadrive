from __future__ import annotations

import copy
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from models.bev_planner.mode_contract import (
    HardModeMaskConfig,
    ModeContractError,
    validate_trajectory_kinematics,
)
from models.controller.longitudinal_reference import (
    LongitudinalReferenceError,
    LongitudinalTrackingReference,
    build_feedback_executable_profile,
    project_point_to_path_arc,
    sample_path_at_arc,
)
from models.platoon_planner.collision_geometry import (
    obb_overlap_series,
    world_trajectory_to_ego_local,
)


def audit_dense_footprint_on_lanes(
    trajectory: np.ndarray,
    lanes: list[object] | tuple[object, ...],
    dimensions: tuple[float, float],
    *,
    dense_dt_s: float,
) -> tuple[bool, dict]:
    """Check every vehicle footprint point against the same execution lanes.

    This is the single road-footprint contract shared by candidate generation
    and committed execution.  A point is valid when it lies in the union of
    the source, target and continuation lane surfaces.
    """

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("road footprint trajectory must have shape [N,3]")
    if not np.isfinite(trajectory).all():
        raise ValueError("road footprint trajectory must be finite")
    valid_lanes = []
    seen_lane_keys = set()
    for lane in lanes:
        if lane is None:
            continue
        lane_index = tuple(getattr(lane, "index", ()) or ())
        lane_key = lane_index if lane_index else ("object", id(lane))
        if lane_key in seen_lane_keys:
            continue
        seen_lane_keys.add(lane_key)
        valid_lanes.append(lane)
    if not valid_lanes:
        return False, {"reason": "no_execution_lanes"}
    length, width = (float(dimensions[0]), float(dimensions[1]))
    if not np.isfinite([length, width]).all() or length <= 0.0 or width <= 0.0:
        raise ValueError("vehicle footprint dimensions must be positive and finite")
    offsets = np.asarray(
        [
            [0.5 * length, 0.5 * width],
            [0.5 * length, -0.5 * width],
            [-0.5 * length, 0.5 * width],
            [-0.5 * length, -0.5 * width],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    for pose_index, pose in enumerate(trajectory):
        cosine, sine = math.cos(float(pose[2])), math.sin(float(pose[2]))
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        points = pose[:2][None, :] + offsets @ rotation.T
        for corner_index, point in enumerate(points):
            inside = False
            lane_coordinates = []
            for lane in valid_lanes:
                try:
                    longitudinal, lateral = lane.local_coordinates(point)
                    lane_length = float(getattr(lane, "length", 0.0) or 0.0)
                    lane_width = float(getattr(lane, "width", 3.5) or 3.5)
                except Exception:
                    continue
                lane_coordinates.append(
                    {
                        "lane_index": list(getattr(lane, "index", ())),
                        "longitudinal_m": float(longitudinal),
                        "lateral_m": float(lateral),
                        "length_m": lane_length,
                        "width_m": lane_width,
                    }
                )
                if (
                    -1e-3 <= float(longitudinal) <= lane_length + 1e-3
                    and abs(float(lateral)) <= 0.5 * lane_width + 1e-3
                ):
                    inside = True
                    break
            if not inside:
                return False, {
                    "reason": "footprint_point_outside_execution_lanes",
                    "trajectory_index": int(pose_index),
                    "time_offset_s": float(pose_index * dense_dt_s),
                    "corner_index": int(corner_index),
                    "pose": [float(value) for value in pose],
                    "point": [float(value) for value in point],
                    "lane_coordinates": lane_coordinates,
                }
    return True, {"reason": "passed"}


class NormalPlannerKinematicError(RuntimeError):
    """Raised when a selected native trajectory violates the fixed-time contract."""


class NormalPlannerNoFeasiblePlan(RuntimeError):
    """Raised when no ranked RuleMaker proposal has a native joint trajectory."""

    def __init__(self, message: str, *, reason_code: str, debug: dict) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.debug = copy.deepcopy(debug)


class CommittedTrajectoryError(RuntimeError):
    """Raised when an accepted joint trajectory can no longer be executed safely."""

    def __init__(self, message: str, *, reason_code: str, debug: dict) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.debug = copy.deepcopy(debug)


@dataclass(frozen=True)
class TrajectoryExecutionSpec:
    """Immutable parameterization of one selected native trajectory."""

    agent_id: str
    source_lane_index: tuple
    continuation_lane_index: tuple
    target_lane_index: tuple
    start_s: float
    start_d: float
    end_d: float
    initial_speed_mps: float
    acceleration_mps2: float
    acceleration_duration_s: float
    recovery_acceleration_mps2: float
    lane_change_duration_s: float
    lane_change_start_delay_s: float
    default_heading: float
    selected_candidate_index: int
    rule_target_point: tuple[float, float]
    sample_times_s: np.ndarray
    trajectory_world: np.ndarray
    path_arc_m: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        times = np.ascontiguousarray(self.sample_times_s, dtype=np.float64)
        trajectory = np.ascontiguousarray(self.trajectory_world, dtype=np.float64)
        path_arc = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
                ),
            )
        )
        if times.ndim != 1 or trajectory.shape != (times.size, 3):
            raise ValueError("execution trajectory/time shape mismatch")
        if path_arc.shape != times.shape:
            raise ValueError("execution path arc/time shape mismatch")
        if times.size < 2 or abs(float(times[0])) > 1e-9:
            raise ValueError("execution trajectory must start at t=0")
        if (
            not np.isfinite(times).all()
            or not np.isfinite(trajectory).all()
            or not np.isfinite(path_arc).all()
        ):
            raise ValueError("execution trajectory must be finite")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("execution trajectory times must increase")
        if abs(float(path_arc[0])) > 1e-9 or np.any(np.diff(path_arc) < -1e-9):
            raise ValueError("execution path arc must start at zero and be monotonic")
        times.setflags(write=False)
        trajectory.setflags(write=False)
        path_arc.setflags(write=False)
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "trajectory_world", trajectory)
        object.__setattr__(self, "path_arc_m", path_arc)


@dataclass(frozen=True)
class JointTrajectoryExecutionPlan:
    """Atomic three-agent trajectory retained for a lane-change commitment."""

    execution_id: int
    proposal_id: int
    proposal_rank: int
    start_step: int
    start_time_s: float
    rule_actions: Mapping[str, int]
    agent_specs: Mapping[str, TrajectoryExecutionSpec]
    committed_agents: tuple[str, ...]
    completion_deadline_s: float


@dataclass(frozen=True)
class RolledJointTrajectory:
    execution_id: int
    elapsed_s: float
    trajectories_world: Mapping[str, np.ndarray]
    trajectories_local: Mapping[str, np.ndarray]
    longitudinal_references: Mapping[str, LongitudinalTrackingReference]
    rule_actions: Mapping[str, int]
    debug: Mapping[str, object]


@dataclass(frozen=True)
class RankedJointPlan:
    proposal_id: int
    proposal_rank: int
    rule_score: float
    decisions: Mapping[str, Mapping[str, object]]
    trajectories_world: Mapping[str, np.ndarray]
    trajectories_local: Mapping[str, np.ndarray]
    selected_candidate_indices: Mapping[str, int]
    execution_plan: JointTrajectoryExecutionPlan | None = None


@dataclass(frozen=True)
class _Neighbor:
    name: str
    vehicle: object
    is_platoon: bool
    delta_s_m: float
    bumper_gap_m: float
    speed_mps: float
    ttc_s: float


@dataclass(frozen=True)
class _TrafficEnvelope:
    front: _Neighbor | None
    rear: _Neighbor | None


@dataclass
class _TrajectoryCandidate:
    dense: np.ndarray
    output: np.ndarray
    score: float
    acceleration_mps2: float
    acceleration_duration_s: float
    recovery_acceleration_mps2: float
    lane_change_duration_s: float
    lane_change_start_delay_s: float
    stop_time_s: float | None
    terminal_progress_m: float
    execution_spec: TrajectoryExecutionSpec | None = None
    execution_parameters: Mapping[str, object] | None = None


class PlatoonNormalPlanner:
    """Gap-aware four-second expert planner with joint platoon selection."""

    OUTPUT_DT_S = 0.5
    HORIZON_S = 4.0
    DENSE_DT_S = 0.1
    MAX_SPEED_MPS = 100.0 / 3.6
    MIN_ACCEL_MPS2 = -8.0
    MAX_ACCEL_MPS2 = 5.0
    LANE_CHANGE_DURATIONS_S = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
    URGENT_LANE_CHANGE_DURATIONS_S = (1.0, 1.5, 2.0)
    LANE_END_CLEARANCE_M = 0.5

    def __init__(
        self,
        *,
        num_output_points: int = 8,
        lane_change_target_margin_m: float = 0.15,
        keep_lateral_margin_m: float = 0.25,
        background_safe_gap_m: float = 5.0,
        platoon_safe_gap_m: float = 7.0,
        safety_distance_m: float = 6.0,
        safety_weight: float = 8.0,
        ttc_threshold_s: float = 3.0,
        ttc_weight: float = 4.0,
        hard_collision_check_enabled: bool = True,
        collision_margin_m: float = 0.2,
        candidate_pool_size: int = 12,
        joint_pair_weight: float = 0.25,
    ) -> None:
        if int(num_output_points) != 8:
            raise ValueError("PlatoonNormalPlanner requires exactly eight future points")
        if int(candidate_pool_size) <= 0:
            raise ValueError("candidate_pool_size must be positive")
        self.num_output_points = 8
        self.lane_change_target_margin_m = float(lane_change_target_margin_m)
        self.keep_lateral_margin_m = float(keep_lateral_margin_m)
        self.background_safe_gap_m = float(background_safe_gap_m)
        self.platoon_safe_gap_m = float(platoon_safe_gap_m)
        self.safety_distance_m = float(safety_distance_m)
        self.safety_weight = float(safety_weight)
        self.ttc_threshold_s = float(ttc_threshold_s)
        self.ttc_weight = float(ttc_weight)
        self.hard_collision_check_enabled = bool(hard_collision_check_enabled)
        self.collision_margin_m = float(collision_margin_m)
        self.candidate_pool_size = int(candidate_pool_size)
        self.joint_pair_weight = float(joint_pair_weight)
        self._dense_times = np.arange(
            0.0,
            self.HORIZON_S + 0.5 * self.DENSE_DT_S,
            self.DENSE_DT_S,
            dtype=np.float64,
        )
        self._output_indices = np.asarray(
            [round(value / self.DENSE_DT_S) for value in np.arange(1, 9) * self.OUTPUT_DT_S],
            dtype=np.int64,
        )
        self._last_debug: dict | None = None
        self._execution_counter = 0
        self._last_selected_candidates: dict[str, _TrajectoryCandidate] = {}

    def plan(
        self,
        env,
        agent_decisions,
        *,
        _pool_cache: dict | None = None,
    ) -> dict[str, np.ndarray]:
        planning_started_at = time.perf_counter()
        self._last_selected_candidates = {}
        agents = getattr(env, "agents", {}) or {}
        ordered_ids = [
            str(agent_id)
            for agent_id in (agent_decisions or {})
            if agent_id in agents
        ]
        pools: dict[str, list[_TrajectoryCandidate]] = {}
        debug: dict[str, dict] = {}
        formation_flags = {
            bool((decision or {}).get("formation_constraint_enabled", True))
            for decision in (agent_decisions or {}).values()
        }
        if len(formation_flags) != 1:
            raise ValueError(
                "all joint decisions must agree on formation_constraint_enabled"
            )
        formation_constraint_enabled = formation_flags.pop()

        for agent_id in ordered_ids:
            decision = (agent_decisions or {})[agent_id] or {}
            action = int(decision.get("action", 0))
            target_point = np.asarray(
                decision.get("target_point", [15.0, 0.0]), dtype=np.float32
            ).reshape(2)
            target_lane_index = tuple(
                decision.get("target_lane_index", ()) or ()
            )
            commitment_elapsed_s = decision.get("commitment_elapsed_s")
            commitment_elapsed_s = (
                None
                if commitment_elapsed_s is None
                else float(commitment_elapsed_s)
            )
            cache_key = (
                str(agent_id),
                int(action),
                target_lane_index,
                commitment_elapsed_s,
                np.ascontiguousarray(target_point).tobytes(),
            )
            cached = None if _pool_cache is None else _pool_cache.get(cache_key)
            if cached is None:
                pool, agent_debug = self._generate_candidate_pool(
                    env,
                    agents[agent_id],
                    action,
                    target_point,
                    target_lane_index=target_lane_index,
                    commitment_elapsed_s=commitment_elapsed_s,
                )
                if _pool_cache is not None:
                    _pool_cache[cache_key] = (pool, copy.deepcopy(agent_debug))
                agent_debug["pool_cache_hit"] = False
            else:
                pool, cached_debug = cached
                agent_debug = copy.deepcopy(cached_debug)
                agent_debug["pool_cache_hit"] = True
            agent_debug["rule_action"] = action
            agent_debug["rule_target_point"] = target_point.astype(
                np.float32,
                copy=False,
            ).tolist()
            pools[agent_id] = pool
            debug[agent_id] = agent_debug

        missing = [agent_id for agent_id in ordered_ids if not pools[agent_id]]
        if missing:
            results = {}
            for agent_id in ordered_ids:
                reason = debug[agent_id].get("fallback_reason")
                if agent_id not in missing:
                    reason = "peer_has_no_safe_candidate"
                debug[agent_id].update(
                    fallback_used=True,
                    fallback_reason=reason or "no_safe_candidate",
                    best_index=None,
                )
                results[agent_id] = self._fallback_keep_trajectory(agents[agent_id])
            debug["_joint"] = {
                "fallback_used": True,
                "fallback_reason": "single_agent_no_safe_candidate",
                "missing_agents": missing,
                "combination_count": 0,
                "pairwise_conflict_count": 0,
                "pairwise_conflict_pair_count": 0,
                "formation_constraint_enabled": formation_constraint_enabled,
                "formation_penalty_applied": False,
                "prefix_counts": [],
                "planning_time_ms": (
                    time.perf_counter() - planning_started_at
                ) * 1000.0,
            }
            self._last_debug = debug
            return results

        selection, joint_debug = self._select_joint_candidates(
            env,
            ordered_ids,
            agents,
            pools,
            formation_constraint_enabled=formation_constraint_enabled,
        )
        debug["_joint"] = joint_debug
        if selection is None:
            results = {}
            for agent_id in ordered_ids:
                debug[agent_id].update(
                    fallback_used=True,
                    fallback_reason="no_safe_joint_combination",
                    best_index=None,
                )
                results[agent_id] = self._fallback_keep_trajectory(agents[agent_id])
            debug["_joint"]["planning_time_ms"] = (
                time.perf_counter() - planning_started_at
            ) * 1000.0
            self._last_debug = debug
            return results

        results = {}
        for agent_id, selected_index in zip(ordered_ids, selection):
            candidate = pools[agent_id][selected_index]
            final_audit = self._candidate_kinematics_audit(
                candidate.output, agents[agent_id]
            )
            if not final_audit.valid:
                raise NormalPlannerKinematicError(
                    f"{agent_id} selected trajectory is dynamically invalid: "
                    + ",".join(final_audit.violations)
                )
            results[agent_id] = candidate.output.astype(np.float32, copy=False)
            self._last_selected_candidates[agent_id] = candidate
            debug[agent_id].update(
                fallback_used=False,
                fallback_reason=None,
                best_index=int(selected_index),
                selected_acceleration_mps2=float(candidate.acceleration_mps2),
                selected_acceleration_duration_s=float(
                    candidate.acceleration_duration_s
                ),
                selected_recovery_acceleration_mps2=float(
                    candidate.recovery_acceleration_mps2
                ),
                selected_lane_change_duration_s=float(candidate.lane_change_duration_s),
                selected_lane_change_start_delay_s=float(
                    candidate.lane_change_start_delay_s
                ),
                selected_stop_time_s=(
                    None if candidate.stop_time_s is None else float(candidate.stop_time_s)
                ),
            )
            for index, item in enumerate(debug[agent_id]["candidates"]):
                item["selected"] = index == selected_index
        debug["_joint"]["planning_time_ms"] = (
            time.perf_counter() - planning_started_at
        ) * 1000.0
        self._last_debug = debug
        return results

    def plan_ranked(self, env, proposals) -> RankedJointPlan:
        """Select the first RuleMaker-ranked proposal with a native joint plan."""

        ordered = sorted(
            tuple(proposals),
            key=lambda value: (int(value.rank), int(value.proposal_id)),
        )
        if not ordered:
            debug = {"proposal_attempts": [], "selected_proposal_id": None}
            self._last_debug = debug
            raise NormalPlannerNoFeasiblePlan(
                "RuleMaker produced no joint action proposal",
                reason_code="all_rule_proposals_infeasible",
                debug=debug,
            )
        pool_cache: dict = {}
        attempts: list[dict] = []
        pool_cache_hit_count = 0
        pool_request_count = 0
        last_attempt_debug: dict = {}
        for proposal in ordered:
            trajectories = self.plan(
                env,
                proposal.decisions,
                _pool_cache=pool_cache,
            )
            attempt_debug = self.get_last_debug() or {}
            last_attempt_debug = attempt_debug
            joint_debug = attempt_debug.get("_joint", {})
            per_agent_debug = [
                value
                for key, value in attempt_debug.items()
                if key != "_joint" and isinstance(value, Mapping)
            ]
            pool_request_count += len(per_agent_debug)
            pool_cache_hit_count += sum(
                int(bool(value.get("pool_cache_hit", False)))
                for value in per_agent_debug
            )
            attempt = {
                "proposal_id": int(proposal.proposal_id),
                "rank": int(proposal.rank),
                "rule_score": float(proposal.rule_score),
                "actions": {
                    agent_id: int(decision["action"])
                    for agent_id, decision in proposal.decisions.items()
                },
                "fallback_reason": joint_debug.get("fallback_reason"),
                "missing_agents": list(joint_debug.get("missing_agents", ())),
                "native_feasible": not bool(
                    joint_debug.get("fallback_used", True)
                ),
            }
            attempts.append(attempt)
            if not attempt["native_feasible"]:
                continue

            local: dict[str, np.ndarray] = {}
            indices: dict[str, int] = {}
            for agent_id, trajectory in trajectories.items():
                vehicle = env.agents[agent_id]
                origin = np.asarray(
                    [
                        float(vehicle.position[0]),
                        float(vehicle.position[1]),
                        float(getattr(vehicle, "heading_theta", 0.0)),
                    ],
                    dtype=np.float64,
                )
                local[agent_id] = world_trajectory_to_ego_local(
                    trajectory, origin
                )
                indices[agent_id] = int(attempt_debug[agent_id]["best_index"])
            attempt_debug["_ranked"] = {
                "proposal_attempts": attempts,
                "selected_proposal_id": int(proposal.proposal_id),
                "selected_proposal_rank": int(proposal.rank),
                "pool_cache_entry_count": len(pool_cache),
                "pool_request_count": int(pool_request_count),
                "pool_cache_hit_count": int(pool_cache_hit_count),
            }
            self._last_debug = attempt_debug
            committed_agents = tuple(
                agent_id
                for agent_id, decision in proposal.decisions.items()
                if int(decision.get("action", 0)) != 0
            )
            execution_plan = None
            if committed_agents:
                self._execution_counter += 1
                config = getattr(env, "config", {}) or {}
                decision_dt_s = float(
                    config.get("physics_world_step_size", 0.02)
                ) * float(config.get("decision_repeat", 5))
                deadline = max(
                    float(self._last_selected_candidates[agent_id].lane_change_start_delay_s)
                    + float(self._last_selected_candidates[agent_id].lane_change_duration_s)
                    for agent_id in committed_agents
                )
                specs = {
                    agent_id: self._build_execution_spec(
                        env,
                        agent_id,
                        self._last_selected_candidates[agent_id],
                        selected_candidate_index=indices[agent_id],
                        maximum_time_s=deadline + self.HORIZON_S + decision_dt_s,
                    )
                    for agent_id in trajectories
                }
                start_step = int(getattr(env, "_scenario_step_count", 0) or 0)
                execution_plan = JointTrajectoryExecutionPlan(
                    execution_id=int(self._execution_counter),
                    proposal_id=int(proposal.proposal_id),
                    proposal_rank=int(proposal.rank),
                    start_step=start_step,
                    start_time_s=float(start_step * decision_dt_s),
                    rule_actions={
                        agent_id: int(decision.get("action", 0))
                        for agent_id, decision in proposal.decisions.items()
                    },
                    agent_specs=specs,
                    committed_agents=committed_agents,
                    completion_deadline_s=float(deadline),
                )
            return RankedJointPlan(
                proposal_id=int(proposal.proposal_id),
                proposal_rank=int(proposal.rank),
                rule_score=float(proposal.rule_score),
                decisions=proposal.decisions,
                trajectories_world={
                    key: np.ascontiguousarray(value, dtype=np.float32)
                    for key, value in trajectories.items()
                },
                trajectories_local=local,
                selected_candidate_indices=indices,
                execution_plan=execution_plan,
            )

        committed = any(
            bool(decision.get("maneuver_committed", False))
            for proposal in ordered
            for decision in proposal.decisions.values()
        )
        reason_code = (
            "committed_action_infeasible"
            if committed
            else "all_rule_proposals_infeasible"
        )
        debug = {
            "proposal_attempts": attempts,
            "selected_proposal_id": None,
            "pool_cache_entry_count": len(pool_cache),
            "pool_request_count": int(pool_request_count),
            "pool_cache_hit_count": int(pool_cache_hit_count),
            "reason_code": reason_code,
        }
        last_attempt_debug["_ranked"] = debug
        self._last_debug = last_attempt_debug
        raise NormalPlannerNoFeasiblePlan(
            "no ranked RuleMaker proposal has a native joint trajectory",
            reason_code=reason_code,
            debug=debug,
        )

    def _build_execution_spec(
        self,
        env,
        agent_id: str,
        candidate: _TrajectoryCandidate,
        *,
        selected_candidate_index: int,
        maximum_time_s: float,
    ) -> TrajectoryExecutionSpec:
        params = dict(candidate.execution_parameters or {})
        required = {
            "source_lane_index",
            "target_lane_index",
            "start_s",
            "start_d",
            "end_d",
            "initial_speed_mps",
            "default_heading",
            "rule_target_point",
        }
        if not required.issubset(params):
            raise NormalPlannerKinematicError(
                f"{agent_id} selected candidate lacks execution parameters"
            )
        source_lane = self._lane_from_index(env, tuple(params["source_lane_index"]))
        continuation_index = tuple(params.get("continuation_lane_index", ()) or ())
        continuation_lane = (
            self._lane_from_index(env, continuation_index)
            if continuation_index
            else None
        )
        if source_lane is None:
            raise NormalPlannerKinematicError(
                f"{agent_id} execution source lane is unavailable"
            )
        continuation = self._continuation_context(source_lane, continuation_lane)
        sample_times = np.arange(
            0.0,
            float(maximum_time_s) + 0.5 * self.DENSE_DT_S,
            self.DENSE_DT_S,
            dtype=np.float64,
        )
        progress = self._longitudinal_progress(
            float(params["initial_speed_mps"]),
            float(candidate.acceleration_mps2),
            sample_times,
            acceleration_duration_s=float(candidate.acceleration_duration_s),
            recovery_acceleration_mps2=float(
                candidate.recovery_acceleration_mps2
            ),
        )
        trajectory = self._build_dense_candidate(
            source_lane=source_lane,
            continuation_lane=continuation_lane,
            continuation=continuation,
            start_s=float(params["start_s"]),
            start_d=float(params["start_d"]),
            progress=progress,
            end_d=float(params["end_d"]),
            lane_change_duration_s=float(candidate.lane_change_duration_s),
            lane_change_start_delay_s=float(
                candidate.lane_change_start_delay_s
            ),
            default_heading=float(params["default_heading"]),
            times=sample_times,
        )
        if trajectory is None:
            raise NormalPlannerKinematicError(
                f"{agent_id} selected trajectory cannot be parameterized"
            )
        initial_count = min(candidate.dense.shape[0], trajectory.shape[0])
        xy_matches = np.allclose(
            trajectory[:initial_count, :2],
            candidate.dense[:initial_count, :2],
            rtol=1e-5,
            atol=2e-4,
        )
        heading_count = max(initial_count - 1, 0)
        heading_matches = np.allclose(
            trajectory[:heading_count, 2],
            candidate.dense[:heading_count, 2],
            rtol=1e-5,
            atol=2e-4,
        )
        if not xy_matches or not heading_matches:
            raise NormalPlannerKinematicError(
                f"{agent_id} execution parameterization does not reproduce selected path"
            )
        return TrajectoryExecutionSpec(
            agent_id=str(agent_id),
            source_lane_index=tuple(params["source_lane_index"]),
            continuation_lane_index=continuation_index,
            target_lane_index=tuple(params["target_lane_index"]),
            start_s=float(params["start_s"]),
            start_d=float(params["start_d"]),
            end_d=float(params["end_d"]),
            initial_speed_mps=float(params["initial_speed_mps"]),
            acceleration_mps2=float(candidate.acceleration_mps2),
            acceleration_duration_s=float(candidate.acceleration_duration_s),
            recovery_acceleration_mps2=float(
                candidate.recovery_acceleration_mps2
            ),
            lane_change_duration_s=float(candidate.lane_change_duration_s),
            lane_change_start_delay_s=float(
                candidate.lane_change_start_delay_s
            ),
            default_heading=float(params["default_heading"]),
            selected_candidate_index=int(selected_candidate_index),
            rule_target_point=tuple(params["rule_target_point"]),
            sample_times_s=sample_times,
            trajectory_world=trajectory,
        )

    def get_last_debug(self) -> dict | None:
        return copy.deepcopy(self._last_debug)

    def _generate_candidate_pool(
        self,
        env,
        vehicle,
        action: int,
        target_point: np.ndarray,
        *,
        target_lane_index: tuple = (),
        commitment_elapsed_s: float | None = None,
    ) -> tuple[list[_TrajectoryCandidate], dict]:
        stats = {
            "raw_candidate_count": 0,
            "kinematic_rejection_count": 0,
            "local_kinematic_rejection_count": 0,
            "corridor_rejection_count": 0,
            "road_rejection_count": 0,
            "background_collision_rejection_count": 0,
            "lane_end_rejection_count": 0,
        }
        collision_hits: Counter[str] = Counter()
        kinematic_hits: Counter[str] = Counter()
        road_hits: Counter[str] = Counter()
        source_lane = getattr(vehicle, "lane", None)
        if source_lane is None:
            return [], self._empty_debug("missing_source_lane", stats, collision_hits)

        try:
            start_s, start_d = source_lane.local_coordinates(
                np.asarray(vehicle.position[:2], dtype=np.float64)
            )
            start_s = float(start_s)
            start_d = float(start_d)
        except Exception:
            return [], self._empty_debug("invalid_source_lane_projection", stats, collision_hits)

        continuation_lane = self._get_continuation_lane(env, vehicle, source_lane)
        source_length = float(getattr(source_lane, "length", 0.0) or 0.0)
        if start_s > source_length and continuation_lane is not None:
            try:
                source_lane = continuation_lane
                start_s, start_d = source_lane.local_coordinates(
                    np.asarray(vehicle.position[:2], dtype=np.float64)
                )
                start_s = float(start_s)
                start_d = float(start_d)
                source_length = float(getattr(source_lane, "length", 0.0) or 0.0)
                continuation_lane = self._get_continuation_lane(env, vehicle, source_lane)
            except Exception:
                return [], self._empty_debug(
                    "invalid_continuation_lane_projection", stats, collision_hits
                )

        target_lane = (
            self._lane_from_index(env, target_lane_index)
            if target_lane_index
            else self._resolve_target_lane(env, source_lane, int(action))
        )
        if target_lane is None:
            return [], self._empty_debug("target_lane_unavailable", stats, collision_hits)

        continuation = self._continuation_context(source_lane, continuation_lane)
        total_length = source_length + continuation["remaining_length"]
        target_world = self._ego_local_to_world(vehicle, target_point)
        try:
            desired_end_s = float(source_lane.local_coordinates(target_world)[0])
        except Exception:
            desired_end_s = start_s + max(float(target_point[0]), 0.0)
        desired_end_s = float(np.clip(desired_end_s, start_s, total_length))
        desired_end_d = self._desired_end_lateral(
            source_lane,
            target_lane,
            int(action),
            desired_end_s,
            target_world,
        )

        source_envelope = self._traffic_envelope(env, vehicle, source_lane)
        target_envelope = self._traffic_envelope(env, vehicle, target_lane)
        ego_speed = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
        reachable = self._reachable_progress_range(ego_speed, self.HORIZON_S)
        ego_length, _ = self._vehicle_dimensions(vehicle)
        usable_source_progress = max(
            source_length
            - start_s
            - 0.5 * ego_length
            - self.LANE_END_CLEARANCE_M,
            0.0,
        )
        lane_end_restricted = self._lane_end_restricted(
            action=int(action),
            ego_speed_mps=ego_speed,
            usable_source_progress_m=usable_source_progress,
        )
        durations = self._lane_change_durations(
            action=int(action),
            lane_end_restricted=lane_end_restricted,
        )
        commitment_deadline_remaining_s = None
        if int(action) != 0 and commitment_elapsed_s is not None:
            durations, commitment_deadline_remaining_s = (
                self._committed_lane_change_durations(
                    durations, commitment_elapsed_s
                )
            )
        lateral_targets = self._candidate_lateral_targets(
            int(action),
            start_d,
            desired_end_d,
            source_lane,
            target_lane,
        )
        background_predictions = self._predicted_obstacles(
            env,
            vehicle,
            self._dense_times,
            include_platoon=False,
        )
        candidates: list[_TrajectoryCandidate] = []
        corridor_debug: dict[str, list[float | None]] = {}
        shared_ttc_penalty = self._ttc_penalty(
            np.empty((0, 3), dtype=np.float64),
            source_lane=source_lane,
            env=env,
            vehicle=vehicle,
            duration=self.HORIZON_S,
        )

        for duration in durations:
            evaluation_time = min(float(duration), self.HORIZON_S)
            if int(action) != 0 and float(duration) > self.HORIZON_S:
                # The vehicle has not entered the target-lane centre by the
                # end of the stored four-second horizon. OBB checks still
                # enforce collision freedom along the partial manoeuvre.
                lower, upper = -float("inf"), float("inf")
            else:
                lower, upper = self._terminal_search_corridor(
                    self._background_only_envelope(target_envelope),
                    vehicle,
                    evaluation_time,
                )
            corridor_debug[f"{duration:.1f}"] = [
                None if not np.isfinite(lower) else float(lower),
                None if not np.isfinite(upper) else float(upper),
            ]
            if lower > upper:
                stats["corridor_rejection_count"] += len(lateral_targets)
                continue
            accelerations = self._candidate_accelerations(
                ego_speed_mps=ego_speed,
                desired_progress_m=max(desired_end_s - start_s, 0.0),
                envelope=self._nearest_front_envelope(source_envelope, target_envelope),
                corridor=(lower, upper),
                evaluation_time_s=evaluation_time,
            )
            start_delays = self._lane_change_start_delays(
                int(action),
                float(duration),
                target_envelope,
            )
            if commitment_elapsed_s is not None and int(action) != 0:
                start_delays = (0.0,)
            if lane_end_restricted and int(action) != 0:
                start_delays = (0.0,)
            profile_options = []
            for acceleration in accelerations:
                for acceleration_duration in self._acceleration_durations(float(acceleration)):
                    recovery_values = self._recovery_accelerations(
                        float(acceleration),
                        float(acceleration_duration),
                    )
                    for recovery_acceleration in recovery_values:
                        progress = self._longitudinal_progress(
                            ego_speed,
                            float(acceleration),
                            self._dense_times,
                            acceleration_duration_s=acceleration_duration,
                            recovery_acceleration_mps2=recovery_acceleration,
                        )
                        progress_at_completion = float(
                            np.interp(evaluation_time, self._dense_times, progress)
                        )
                        if (
                            progress_at_completion < lower - 1e-6
                            or progress_at_completion > upper + 1e-6
                        ):
                            stats["corridor_rejection_count"] += len(lateral_targets)
                            continue
                        desired_at_completion = float(
                            np.clip(
                                max(desired_end_s - start_s, 0.0),
                                lower,
                                upper,
                            )
                        )
                        profile_cost = (
                            abs(progress_at_completion - desired_at_completion)
                            + 0.05
                            * abs(float(acceleration))
                            * float(acceleration_duration)
                            + 0.02 * abs(float(recovery_acceleration))
                        )
                        profile_options.append(
                            (
                                profile_cost,
                                float(acceleration),
                                float(acceleration_duration),
                                float(recovery_acceleration),
                                progress,
                            )
                        )
            profile_options.sort(key=lambda value: value[0])
            for (
                _profile_cost,
                acceleration,
                acceleration_duration,
                recovery_acceleration,
                progress,
            ) in self._select_longitudinal_profiles(
                profile_options,
                maximum=12 if int(action) == 0 else 6,
            ):
                for start_delay in start_delays:
                    if lane_end_restricted and int(action) != 0:
                        completion_progress = self._lane_change_completion_progress(
                            progress,
                            duration_s=float(duration),
                            start_delay_s=float(start_delay),
                        )
                        if completion_progress > usable_source_progress + 1e-6:
                            stats["lane_end_rejection_count"] += len(
                                lateral_targets
                            )
                            continue
                    for end_d in lateral_targets:
                        stats["raw_candidate_count"] += 1
                        candidate_dense = self._build_dense_candidate(
                            source_lane=source_lane,
                            continuation_lane=continuation_lane,
                            continuation=continuation,
                            start_s=start_s,
                            start_d=start_d,
                            progress=progress,
                            end_d=float(end_d),
                            lane_change_duration_s=float(duration),
                            lane_change_start_delay_s=float(start_delay),
                            default_heading=float(
                                getattr(vehicle, "heading_theta", 0.0)
                            ),
                        )
                        if candidate_dense is None:
                            stats["road_rejection_count"] += 1
                            road_hits["path_parameterization_failed"] += 1
                            continue
                        footprint_valid, footprint_detail = (
                            audit_dense_footprint_on_lanes(
                                candidate_dense,
                                (source_lane, target_lane, continuation_lane),
                                self._vehicle_dimensions(vehicle),
                                dense_dt_s=self.DENSE_DT_S,
                            )
                        )
                        if not footprint_valid:
                            stats["road_rejection_count"] += 1
                            road_hits[
                                str(footprint_detail.get("reason", "unknown"))
                            ] += 1
                            continue
                        output = candidate_dense[self._output_indices].astype(
                            np.float32, copy=False
                        )
                        kinematic_audit = self._candidate_kinematics_audit(
                            output, vehicle
                        )
                        if not kinematic_audit.valid:
                            stats["kinematic_rejection_count"] += 1
                            kinematic_hits.update(kinematic_audit.violations)
                            continue
                        local_output = world_trajectory_to_ego_local(
                            output,
                            np.asarray(
                                [
                                    float(vehicle.position[0]),
                                    float(vehicle.position[1]),
                                    float(
                                        getattr(vehicle, "heading_theta", 0.0)
                                    ),
                                ],
                                dtype=np.float64,
                            ),
                        )
                        try:
                            local_audit = validate_trajectory_kinematics(
                                local_output,
                                ego_speed,
                                np.zeros((3,), dtype=np.float64),
                                HardModeMaskConfig(),
                            )
                        except ModeContractError as exc:
                            raise NormalPlannerKinematicError(
                                f"invalid local trajectory tensor: {exc}"
                            ) from exc
                        if not local_audit.valid:
                            stats["local_kinematic_rejection_count"] += 1
                            kinematic_hits.update(
                                f"local:{reason}"
                                for reason in local_audit.violations
                            )
                            continue
                        hits = self._collision_names_against_predictions(
                            candidate_dense,
                            self._vehicle_dimensions(vehicle),
                            background_predictions,
                        )
                        if hits:
                            stats["background_collision_rejection_count"] += 1
                            collision_hits.update(hits)
                            continue
                        score = self._score_candidate(
                            output,
                            target_world=target_world,
                            source_lane=source_lane,
                            target_lane=target_lane,
                            action=int(action),
                            desired_end_d=float(desired_end_d),
                            env=env,
                            vehicle=vehicle,
                            duration=self.HORIZON_S,
                            precomputed_ttc_penalty=shared_ttc_penalty,
                        )
                        score += self._background_clearance_penalty(
                            candidate_dense,
                            self._vehicle_dimensions(vehicle),
                            background_predictions,
                        )
                        score += (
                            0.05
                            * abs(float(acceleration))
                            * min(
                                float(acceleration_duration),
                                self.HORIZON_S,
                            )
                            / self.HORIZON_S
                        )
                        score += 0.02 * abs(float(recovery_acceleration))
                        score += 0.05 * abs(float(duration) - self.HORIZON_S)
                        score += 0.02 * float(start_delay)
                        if lane_end_restricted and int(action) != 0:
                            deadline_margin = max(
                                usable_source_progress - completion_progress,
                                0.0,
                            )
                            score += 0.1 / max(deadline_margin + 0.25, 0.25)
                        if not np.isfinite(score):
                            stats["kinematic_rejection_count"] += 1
                            continue
                        stop_time = (
                            ego_speed / -float(acceleration)
                            if (
                                acceleration < -1e-6
                                and ego_speed / -float(acceleration)
                                <= acceleration_duration
                            )
                            else None
                        )
                        candidates.append(
                            _TrajectoryCandidate(
                                dense=candidate_dense,
                                output=output,
                                score=float(score),
                                acceleration_mps2=float(acceleration),
                                acceleration_duration_s=float(
                                    acceleration_duration
                                ),
                                recovery_acceleration_mps2=float(
                                    recovery_acceleration
                                ),
                                lane_change_duration_s=float(duration),
                                lane_change_start_delay_s=float(start_delay),
                                stop_time_s=stop_time,
                                terminal_progress_m=float(progress[-1]),
                                execution_parameters={
                                    "source_lane_index": tuple(
                                        getattr(source_lane, "index", ()) or ()
                                    ),
                                    "continuation_lane_index": tuple(
                                        getattr(continuation_lane, "index", ()) or ()
                                    ),
                                    "target_lane_index": tuple(
                                        getattr(target_lane, "index", ()) or ()
                                    ),
                                    "start_s": float(start_s),
                                    "start_d": float(start_d),
                                    "end_d": float(end_d),
                                    "initial_speed_mps": float(ego_speed),
                                    "default_heading": float(
                                        getattr(vehicle, "heading_theta", 0.0)
                                    ),
                                    "rule_target_point": tuple(
                                        float(value) for value in target_point
                                    ),
                                },
                            )
                        )

        candidates.sort(key=lambda value: value.score)
        pool = self._diverse_candidate_pool(candidates)
        debug = {
            "fallback_used": not bool(pool),
            "fallback_reason": None if pool else "no_safe_candidate",
            "candidate_scores": [float(value.score) for value in pool],
            "best_index": None,
            "candidate_count": len(pool),
            "generated_valid_candidate_count": len(candidates),
            **stats,
            "collision_rejections_by_object": dict(sorted(collision_hits.items())),
            "road_rejections_by_reason": dict(sorted(road_hits.items())),
            "kinematic_rejections_by_reason": dict(
                sorted(kinematic_hits.items())
            ),
            "source_envelope": self._serialize_envelope(source_envelope),
            "target_envelope": self._serialize_envelope(target_envelope),
            "reachable_progress_m": [float(reachable[0]), float(reachable[1])],
            "lateral_targets_m": [float(value) for value in lateral_targets],
            "source_lane_remaining_m": float(
                max(source_length - start_s, 0.0)
            ),
            "source_lane_usable_progress_m": float(usable_source_progress),
            "lane_end_restricted": bool(lane_end_restricted),
            "commitment_elapsed_s": commitment_elapsed_s,
            "commitment_deadline_remaining_s": commitment_deadline_remaining_s,
            "lane_change_durations_s": [
                float(value) for value in durations
            ],
            "safe_corridor_by_duration_m": corridor_debug,
            "candidates": [self._candidate_debug(value) for value in pool],
        }
        return pool, debug

    def _diverse_candidate_pool(
        self,
        candidates: list[_TrajectoryCandidate],
    ) -> list[_TrajectoryCandidate]:
        """Retain local optima while spanning the safe progress interval."""

        if len(candidates) <= self.candidate_pool_size:
            return list(candidates)
        selected: list[_TrajectoryCandidate] = []
        selected_ids: set[int] = set()

        def add(candidate: _TrajectoryCandidate) -> None:
            if id(candidate) not in selected_ids:
                selected.append(candidate)
                selected_ids.add(id(candidate))

        # Preserve the strongest local choices.
        local_slots = min(4, self.candidate_pool_size)
        for candidate in candidates[:local_slots]:
            add(candidate)

        # Span emergency braking through normal progress. This is essential
        # for a jointly safe combination when the three local target points
        # imply incompatible speeds.
        by_progress = sorted(
            candidates,
            key=lambda value: (value.terminal_progress_m, value.score),
        )
        progress_slots = max(self.candidate_pool_size - local_slots, 1)
        for index in np.linspace(
            0,
            len(by_progress) - 1,
            num=progress_slots,
            dtype=np.int64,
        ):
            add(by_progress[int(index)])

        for candidate in candidates:
            add(candidate)
            if len(selected) >= self.candidate_pool_size:
                break
        return selected[: self.candidate_pool_size]

    @staticmethod
    def _select_longitudinal_profiles(
        profiles: list[tuple[float, float, float, float, np.ndarray]],
        *,
        maximum: int = 3,
    ) -> list[tuple[float, float, float, float, np.ndarray]]:
        """Choose profiles by score and terminal-progress coverage."""

        if len(profiles) <= maximum:
            return list(profiles)
        selected: list[tuple[float, float, float, float, np.ndarray]] = []
        selected_ids: set[int] = set()

        def add(profile: tuple[float, float, float, float, np.ndarray]) -> None:
            if id(profile) not in selected_ids:
                selected.append(profile)
                selected_ids.add(id(profile))

        braking_profiles = [
            profile for profile in profiles if float(profile[1]) <= -4.0
        ]
        yield_then_recover = [
            profile
            for profile in braking_profiles
            if 1.0 <= float(profile[2]) <= 2.0 and float(profile[3]) > 0.0
        ]
        if yield_then_recover:
            add(
                min(
                    yield_then_recover,
                    key=lambda value: (
                        abs(float(value[1]) - PlatoonNormalPlanner.MIN_ACCEL_MPS2),
                        abs(float(value[2]) - 2.0),
                        -float(value[3]),
                        float(value[0]),
                    ),
                )
            )
        stop_and_hold = [
            profile
            for profile in braking_profiles
            if float(profile[2]) >= PlatoonNormalPlanner.HORIZON_S
            and abs(float(profile[3])) <= 1e-9
        ]
        if stop_and_hold:
            add(
                min(
                    stop_and_hold,
                    key=lambda value: (
                        float(value[4][-1]),
                        float(value[0]),
                    ),
                )
            )

        score_slots = max(1, maximum // 3)
        for profile in profiles[:score_slots]:
            add(profile)
        by_progress = sorted(
            profiles,
            key=lambda value: (float(value[4][-1]), float(value[0])),
        )
        for index in np.linspace(
            0,
            len(by_progress) - 1,
            num=max(maximum - score_slots, 1),
            dtype=np.int64,
        ):
            add(by_progress[int(index)])
        for profile in profiles:
            add(profile)
            if len(selected) >= maximum:
                break
        return selected[:maximum]

    @staticmethod
    def _empty_debug(
        reason: str,
        stats: Mapping[str, int],
        collision_hits: Counter[str],
    ) -> dict:
        return {
            "fallback_used": True,
            "fallback_reason": str(reason),
            "candidate_scores": [],
            "best_index": None,
            "candidate_count": 0,
            "generated_valid_candidate_count": 0,
            **dict(stats),
            "collision_rejections_by_object": dict(collision_hits),
            "road_rejections_by_reason": {},
            "source_envelope": {"front": None, "rear": None},
            "target_envelope": {"front": None, "rear": None},
            "reachable_progress_m": None,
            "safe_corridor_by_duration_m": {},
            "candidates": [],
        }

    @staticmethod
    def _candidate_debug(candidate: _TrajectoryCandidate) -> dict:
        return {
            "score": float(candidate.score),
            "selected": False,
            "acceleration_mps2": float(candidate.acceleration_mps2),
            "acceleration_duration_s": float(candidate.acceleration_duration_s),
            "recovery_acceleration_mps2": float(
                candidate.recovery_acceleration_mps2
            ),
            "lane_change_duration_s": float(candidate.lane_change_duration_s),
            "lane_change_start_delay_s": float(
                candidate.lane_change_start_delay_s
            ),
            "stop_time_s": (
                None if candidate.stop_time_s is None else float(candidate.stop_time_s)
            ),
            "terminal_progress_m": float(candidate.terminal_progress_m),
            "trajectory_world": candidate.output.astype(
                np.float32, copy=False
            ).tolist(),
        }

    def _select_joint_candidates(
        self,
        env,
        ordered_ids: list[str],
        agents: Mapping[str, object],
        pools: Mapping[str, list[_TrajectoryCandidate]],
        *,
        formation_constraint_enabled: bool = True,
    ) -> tuple[tuple[int, ...] | None, dict]:
        combination_count = 0
        pairwise_conflict_count = 0
        conflict_tables: dict[tuple[int, int], np.ndarray] = {}
        conflict_pair_count = 0
        conflict_counts_by_pair: dict[str, int] = {}
        for first in range(len(ordered_ids)):
            for second in range(first + 1, len(ordered_ids)):
                first_pool = pools[ordered_ids[first]]
                second_pool = pools[ordered_ids[second]]
                table = np.zeros(
                    (len(first_pool), len(second_pool)),
                    dtype=np.bool_,
                )
                for first_index, first_candidate in enumerate(first_pool):
                    for second_index, second_candidate in enumerate(second_pool):
                        table[first_index, second_index] = (
                            self._trajectory_pair_collides(
                                first_candidate.dense,
                                agents[ordered_ids[first]],
                                second_candidate.dense,
                                agents[ordered_ids[second]],
                            )
                        )
                conflict_pair_count += int(np.count_nonzero(table))
                conflict_counts_by_pair[
                    f"{ordered_ids[first]}:{ordered_ids[second]}"
                ] = int(np.count_nonzero(table))
                conflict_tables[(first, second)] = table
        prefixes: list[tuple[int, ...]] = [tuple()]
        prefix_counts: list[int] = []
        for role_index, agent_id in enumerate(ordered_ids):
            next_prefixes: list[tuple[int, ...]] = []
            for prefix in prefixes:
                for candidate_index in range(len(pools[agent_id])):
                    conflict = False
                    for previous_index, previous_candidate_index in enumerate(prefix):
                        if conflict_tables[(previous_index, role_index)][
                            previous_candidate_index, candidate_index
                        ]:
                            pairwise_conflict_count += 1
                            conflict = True
                            break
                    if not conflict:
                        next_prefixes.append(prefix + (candidate_index,))
            prefixes = next_prefixes
            prefix_counts.append(len(prefixes))
            if not prefixes:
                break

        best_selection: tuple[int, ...] | None = None
        best_score = float("inf")
        for selection in prefixes:
            if len(selection) != len(ordered_ids):
                continue
            combination_count += 1
            chosen = [
                pools[agent_id][candidate_index]
                for agent_id, candidate_index in zip(ordered_ids, selection)
            ]
            score = sum(value.score for value in chosen)
            if formation_constraint_enabled:
                score += self._joint_formation_penalty(
                    env,
                    ordered_ids,
                    agents,
                    chosen,
                )
            if score < best_score:
                best_score = float(score)
                best_selection = tuple(int(value) for value in selection)
        return best_selection, {
            "fallback_used": best_selection is None,
            "fallback_reason": (
                "no_safe_joint_combination" if best_selection is None else None
            ),
            "combination_count": int(combination_count),
            "pairwise_conflict_count": int(pairwise_conflict_count),
            "pairwise_conflict_pair_count": int(conflict_pair_count),
            "pairwise_conflict_counts_by_pair": conflict_counts_by_pair,
            "prefix_counts": prefix_counts,
            "formation_constraint_enabled": bool(
                formation_constraint_enabled
            ),
            "formation_penalty_applied": bool(
                formation_constraint_enabled and self.joint_pair_weight > 0.0
            ),
            "selected_indices": (
                None if best_selection is None else list(best_selection)
            ),
            "selected_score": None if best_selection is None else float(best_score),
        }

    def _joint_formation_penalty(
        self,
        env,
        ordered_ids: list[str],
        agents: Mapping[str, object],
        chosen: list[_TrajectoryCandidate],
    ) -> float:
        if len(chosen) < 2 or self.joint_pair_weight <= 0.0:
            return 0.0
        total = 0.0
        for index in range(1, len(chosen)):
            rear_id = ordered_ids[index]
            front_id = ordered_ids[index - 1]
            desired = None
            spacing_fn = getattr(env, "_desired_center_spacing_m", None)
            if callable(spacing_fn):
                try:
                    desired = float(spacing_fn(rear_id, front_id))
                except Exception:
                    desired = None
            if desired is None or not np.isfinite(desired) or desired <= 0.0:
                rear_pos = np.asarray(agents[rear_id].position[:2], dtype=np.float64)
                front_pos = np.asarray(agents[front_id].position[:2], dtype=np.float64)
                desired = max(float(np.linalg.norm(front_pos - rear_pos)), 1.0)
            distances = np.linalg.norm(
                chosen[index - 1].output[:, :2] - chosen[index].output[:, :2],
                axis=1,
            )
            total += float(np.mean(np.abs(distances - desired)) / max(desired, 1.0))
        return float(self.joint_pair_weight * total)

    def _candidate_accelerations(
        self,
        *,
        ego_speed_mps: float,
        desired_progress_m: float,
        envelope: _TrafficEnvelope,
        corridor: tuple[float, float],
        evaluation_time_s: float,
    ) -> tuple[float, ...]:
        reference = 2.0 * (
            float(desired_progress_m) - float(ego_speed_mps) * self.HORIZON_S
        ) / (self.HORIZON_S**2)
        values = [
            reference,
            reference - 1.5,
            reference + 1.5,
            0.0,
            self.MIN_ACCEL_MPS2,
            -6.0,
            -4.0,
            -2.0,
            self._idm_acceleration(ego_speed_mps, envelope.front),
        ]
        lower, upper = corridor
        finite_bounds = [
            value for value in (lower, upper) if np.isfinite(value)
        ]
        if len(finite_bounds) == 2:
            finite_bounds.append(0.5 * (lower + upper))
        for progress in finite_bounds:
            values.append(
                2.0
                * (float(progress) - float(ego_speed_mps) * evaluation_time_s)
                / max(evaluation_time_s**2, 1e-6)
            )
        clipped = {
            round(
                float(np.clip(value, self.MIN_ACCEL_MPS2, self.MAX_ACCEL_MPS2)),
                3,
            )
            for value in values
            if np.isfinite(value)
        }
        return tuple(sorted(clipped))

    def _idm_acceleration(
        self,
        ego_speed_mps: float,
        front: _Neighbor | None,
    ) -> float:
        desired_speed = max(30.0 / 3.6, float(ego_speed_mps), 0.1)
        free_road = 1.0 - (float(ego_speed_mps) / desired_speed) ** 4
        if front is None:
            return float(np.clip(1.8 * free_road, self.MIN_ACCEL_MPS2, self.MAX_ACCEL_MPS2))
        closing_speed = float(ego_speed_mps) - float(front.speed_mps)
        desired_gap = 6.0 + max(
            0.0,
            float(ego_speed_mps) * 1.2
            + float(ego_speed_mps) * closing_speed
            / (2.0 * math.sqrt(1.8 * 2.5)),
        )
        interaction = (desired_gap / max(float(front.bumper_gap_m), 0.1)) ** 2
        return float(
            np.clip(
                1.8 * (free_road - interaction),
                self.MIN_ACCEL_MPS2,
                self.MAX_ACCEL_MPS2,
            )
        )

    def _reachable_progress_range(
        self,
        ego_speed_mps: float,
        horizon_s: float,
    ) -> tuple[float, float]:
        times = np.asarray([0.0, float(horizon_s)], dtype=np.float64)
        minimum = self._longitudinal_progress(
            ego_speed_mps, self.MIN_ACCEL_MPS2, times
        )[-1]
        maximum = self._longitudinal_progress(
            ego_speed_mps, self.MAX_ACCEL_MPS2, times
        )[-1]
        return float(minimum), float(maximum)

    def _acceleration_durations(self, acceleration_mps2: float) -> tuple[float, ...]:
        if acceleration_mps2 >= -0.25:
            return (self.HORIZON_S,)
        return (0.5, 1.0, 2.0, self.HORIZON_S)

    def _recovery_accelerations(
        self,
        acceleration_mps2: float,
        acceleration_duration_s: float,
    ) -> tuple[float, ...]:
        if acceleration_mps2 >= -0.25 or acceleration_duration_s >= self.HORIZON_S:
            return (0.0,)
        return (0.0, 1.5, 3.0)

    def _lane_change_start_delays(
        self,
        action: int,
        completion_time_s: float,
        envelope: _TrafficEnvelope,
    ) -> tuple[float, ...]:
        if int(action) == 0:
            return (0.0,)
        tight_gap = any(
            neighbor is not None
            and neighbor.bumper_gap_m
            < (
                self.platoon_safe_gap_m
                if neighbor.is_platoon
                else self.background_safe_gap_m
            )
            for neighbor in (envelope.front, envelope.rear)
        )
        if not tight_gap:
            base = (0.0, 0.5)
        else:
            latest_safe_start = min(
                max(float(completion_time_s) - 0.5, 1.0),
                3.5,
            )
            base = tuple(
                float(value)
                for value in np.arange(
                    0.5,
                    latest_safe_start + 0.25,
                    0.5,
                )
            )
        return tuple(
            value for value in base if value <= float(completion_time_s) - 0.5
        ) or (0.0,)

    def _longitudinal_progress(
        self,
        initial_speed_mps: float,
        acceleration_mps2: float,
        times: np.ndarray,
        *,
        acceleration_duration_s: float | None = None,
        recovery_acceleration_mps2: float = 0.0,
    ) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64)
        speed = float(np.clip(initial_speed_mps, 0.0, self.MAX_SPEED_MPS))
        acceleration = float(
            np.clip(acceleration_mps2, self.MIN_ACCEL_MPS2, self.MAX_ACCEL_MPS2)
        )
        if acceleration_duration_s is not None:
            active_duration = float(
                np.clip(acceleration_duration_s, 0.0, self.HORIZON_S)
            )
            active_times = np.minimum(times, active_duration)
            accelerated = self._longitudinal_progress(
                speed,
                acceleration,
                active_times,
            )
            terminal_progress = float(
                self._longitudinal_progress(
                    speed,
                    acceleration,
                    np.asarray([active_duration], dtype=np.float64),
                )[0]
            )
            terminal_speed = float(
                np.clip(
                    speed + acceleration * active_duration,
                    0.0,
                    self.MAX_SPEED_MPS,
                )
            )
            recovery_time = np.maximum(times - active_duration, 0.0)
            recovery_progress = self._longitudinal_progress(
                terminal_speed,
                float(recovery_acceleration_mps2),
                recovery_time,
            )
            return np.where(
                times <= active_duration,
                accelerated,
                terminal_progress + recovery_progress,
            )
        if acceleration < -1e-9:
            stop_time = speed / -acceleration
            active_time = np.minimum(times, stop_time)
            return speed * active_time + 0.5 * acceleration * active_time**2
        if acceleration > 1e-9:
            max_speed_time = max((self.MAX_SPEED_MPS - speed) / acceleration, 0.0)
            active_time = np.minimum(times, max_speed_time)
            accelerated = speed * active_time + 0.5 * acceleration * active_time**2
            return accelerated + self.MAX_SPEED_MPS * np.maximum(
                times - max_speed_time, 0.0
            )
        return speed * times

    def _build_dense_candidate(
        self,
        *,
        source_lane,
        continuation_lane,
        continuation: Mapping[str, float],
        start_s: float,
        start_d: float,
        progress: np.ndarray,
        end_d: float,
        lane_change_duration_s: float,
        lane_change_start_delay_s: float,
        default_heading: float,
        times: np.ndarray | None = None,
    ) -> np.ndarray | None:
        evaluation_times = (
            self._dense_times
            if times is None
            else np.asarray(times, dtype=np.float64)
        )
        progress = np.asarray(progress, dtype=np.float64)
        if progress.shape != evaluation_times.shape:
            return None
        source_length = float(getattr(source_lane, "length", 0.0) or 0.0)
        total_length = source_length + float(continuation["remaining_length"])
        s_values = np.clip(float(start_s) + np.asarray(progress), 0.0, total_length)
        completion_progress = float(
            np.interp(
                min(float(lane_change_duration_s), self.HORIZON_S),
                evaluation_times,
                progress,
            )
        )
        start_progress = float(
            np.interp(
                float(lane_change_start_delay_s),
                evaluation_times,
                progress,
            )
        )
        if float(lane_change_duration_s) > self.HORIZON_S:
            observed_duration = max(
                self.HORIZON_S - float(lane_change_start_delay_s),
                self.DENSE_DT_S,
            )
            full_duration = max(
                float(lane_change_duration_s) - float(lane_change_start_delay_s),
                observed_duration,
            )
            completion_progress = start_progress + (
                completion_progress - start_progress
            ) * full_duration / observed_duration
        lateral_progress = completion_progress - start_progress
        if lateral_progress <= 1e-3:
            return None
        # A lane change is a geometric path, not a lateral motion that should
        # continue while longitudinal speed is zero. At least eight metres of
        # progress are reserved for the transition; a tight gap therefore
        # produces a partial but dynamically meaningful change within 4 s.
        lateral_progress = max(lateral_progress, 8.0)
        ratio = np.clip(
            (np.asarray(progress, dtype=np.float64) - start_progress)
            / lateral_progress,
            0.0,
            1.0,
        )
        smootherstep = 6.0 * ratio**5 - 15.0 * ratio**4 + 10.0 * ratio**3
        lateral_values = float(start_d) + (
            float(end_d) - float(start_d)
        ) * smootherstep
        source_mask = (s_values <= source_length) | (continuation_lane is None)
        points_xy = np.empty((len(s_values), 2), dtype=np.float64)
        if np.any(source_mask):
            points_xy[source_mask] = self._lane_positions(
                source_lane,
                np.clip(s_values[source_mask], 0.0, source_length),
                lateral_values[source_mask],
            )
        if np.any(~source_mask):
            continuation_s = float(continuation["s_base"]) + (
                s_values[~source_mask] - source_length
            )
            points_xy[~source_mask] = self._lane_positions(
                continuation_lane,
                np.clip(
                    continuation_s,
                    0.0,
                    float(getattr(continuation_lane, "length", 0.0) or 0.0),
                ),
                lateral_values[~source_mask] + float(continuation["d_offset"]),
            )
        if points_xy.shape != (len(evaluation_times), 2) or not np.isfinite(points_xy).all():
            return None
        return self._append_heading(points_xy, default_heading=default_heading)

    @staticmethod
    def _lane_positions(lane, longitudinal: np.ndarray, lateral: np.ndarray) -> np.ndarray:
        """Vectorized lane projection for MetaDrive straight/circular lanes."""

        longitudinal = np.asarray(longitudinal, dtype=np.float64)
        lateral = np.asarray(lateral, dtype=np.float64)
        direction = np.asarray(getattr(lane, "direction", ()), dtype=np.float64)
        start = np.asarray(getattr(lane, "start", ()), dtype=np.float64)
        direction_lateral = np.asarray(
            getattr(lane, "direction_lateral", ()),
            dtype=np.float64,
        )
        if (
            start.shape == (2,)
            and direction.shape == (2,)
            and direction_lateral.shape == (2,)
        ):
            return (
                start[None, :]
                + longitudinal[:, None] * direction[None, :]
                + lateral[:, None] * direction_lateral[None, :]
            )

        center = np.asarray(getattr(lane, "center", ()), dtype=np.float64)
        radius = float(getattr(lane, "radius", 0.0) or 0.0)
        start_phase = getattr(lane, "start_phase", None)
        if (
            center.shape == (2,)
            and radius > 0.0
            and start_phase is not None
            and direction.shape == ()
        ):
            signed_direction = float(direction)
            phase = (
                signed_direction * longitudinal / radius + float(start_phase)
            )
            radial = radius + lateral * signed_direction
            return center[None, :] + radial[:, None] * np.column_stack(
                [np.cos(phase), np.sin(phase)]
            )

        return np.asarray(
            [
                np.asarray(lane.position(float(s_value), float(d_value))[:2])
                for s_value, d_value in zip(longitudinal, lateral)
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _candidate_kinematics_audit(trajectory: np.ndarray, vehicle):
        origin = np.asarray(
            [
                float(vehicle.position[0]),
                float(vehicle.position[1]),
                float(getattr(vehicle, "heading_theta", 0.0)),
            ],
            dtype=np.float64,
        )
        speed = max(
            float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6,
            0.0,
        )
        try:
            return validate_trajectory_kinematics(
                trajectory,
                speed,
                origin,
                HardModeMaskConfig(),
            )
        except ModeContractError as exc:
            raise NormalPlannerKinematicError(
                f"invalid Normal planner trajectory tensor: {exc}"
            ) from exc

    def _traffic_envelope(self, env, vehicle, lane) -> _TrafficEnvelope:
        try:
            ego_s = float(
                lane.local_coordinates(
                    np.asarray(vehicle.position[:2], dtype=np.float64)
                )[0]
            )
        except Exception:
            return _TrafficEnvelope(front=None, rear=None)
        ego_length, _ = self._vehicle_dimensions(vehicle)
        ego_speed = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
        platoon_ids = {
            id(value) for value in (getattr(env, "agents", {}) or {}).values()
        }
        front: _Neighbor | None = None
        rear: _Neighbor | None = None
        for other_id, other in self._surrounding_vehicles(env):
            if other is vehicle:
                continue
            try:
                other_s, other_d = lane.local_coordinates(
                    np.asarray(other.position[:2], dtype=np.float64)
                )
            except Exception:
                continue
            _, other_width = self._vehicle_dimensions(other)
            lane_half_width = 0.5 * float(getattr(lane, "width", 3.5) or 3.5)
            if abs(float(other_d)) > lane_half_width + 0.5 * other_width:
                continue
            delta_s = float(other_s) - ego_s
            if abs(delta_s) <= 1e-6:
                continue
            other_length, _ = self._vehicle_dimensions(other)
            bumper_gap = max(abs(delta_s) - 0.5 * (ego_length + other_length), 0.0)
            other_speed = max(float(getattr(other, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
            if delta_s > 0.0:
                closing = ego_speed - other_speed
                ttc = bumper_gap / closing if closing > 1e-6 else float("inf")
            else:
                closing = other_speed - ego_speed
                ttc = bumper_gap / closing if closing > 1e-6 else float("inf")
            neighbor = _Neighbor(
                name=str(getattr(other, "name", other_id)),
                vehicle=other,
                is_platoon=id(other) in platoon_ids,
                delta_s_m=delta_s,
                bumper_gap_m=float(bumper_gap),
                speed_mps=float(other_speed),
                ttc_s=float(ttc),
            )
            if delta_s > 0.0 and (
                front is None or delta_s < front.delta_s_m
            ):
                front = neighbor
            if delta_s < 0.0 and (
                rear is None or delta_s > rear.delta_s_m
            ):
                rear = neighbor
        return _TrafficEnvelope(front=front, rear=rear)

    @staticmethod
    def _nearest_front_envelope(
        first: _TrafficEnvelope,
        second: _TrafficEnvelope,
    ) -> _TrafficEnvelope:
        fronts = [value for value in (first.front, second.front) if value is not None]
        rears = [value for value in (first.rear, second.rear) if value is not None]
        return _TrafficEnvelope(
            front=min(fronts, key=lambda value: value.bumper_gap_m) if fronts else None,
            rear=min(rears, key=lambda value: value.bumper_gap_m) if rears else None,
        )

    def _safe_terminal_corridor(
        self,
        envelope: _TrafficEnvelope,
        ego_vehicle,
        time_s: float,
    ) -> tuple[float, float]:
        ego_length, _ = self._vehicle_dimensions(ego_vehicle)
        lower = -float("inf")
        upper = float("inf")
        if envelope.front is not None:
            front_length, _ = self._vehicle_dimensions(envelope.front.vehicle)
            gap = (
                self.platoon_safe_gap_m
                if envelope.front.is_platoon
                else self.background_safe_gap_m
            )
            upper = (
                envelope.front.delta_s_m
                + envelope.front.speed_mps * float(time_s)
                - 0.5 * (ego_length + front_length)
                - gap
            )
        if envelope.rear is not None:
            rear_length, _ = self._vehicle_dimensions(envelope.rear.vehicle)
            gap = (
                self.platoon_safe_gap_m
                if envelope.rear.is_platoon
                else self.background_safe_gap_m
            )
            lower = (
                envelope.rear.delta_s_m
                + envelope.rear.speed_mps * float(time_s)
                + 0.5 * (ego_length + rear_length)
                + gap
            )
        return float(lower), float(upper)

    def _terminal_search_corridor(
        self,
        envelope: _TrafficEnvelope,
        ego_vehicle,
        time_s: float,
    ) -> tuple[float, float]:
        """Return pruning bounds without discarding a valid yield-behind plan.

        A rear vehicle that is closing before the candidate completion time can
        change from "rear" to "front".  The static rear bound only represents
        the pass-in-front branch and would reject every brake-and-yield
        profile.  In that case the rear lower bound is left open; the dense
        OBB/SAT check still rejects trajectories that overlap while it passes.
        """

        lower, upper = self._safe_terminal_corridor(
            envelope,
            ego_vehicle,
            time_s,
        )
        rear = envelope.rear
        if (
            rear is not None
            and np.isfinite(rear.ttc_s)
            and float(rear.ttc_s) <= float(time_s)
        ):
            lower = -float("inf")
        return float(lower), float(upper)

    @staticmethod
    def _background_only_envelope(envelope: _TrafficEnvelope) -> _TrafficEnvelope:
        """Exclude platoon peers from the single-agent hard corridor.

        Platoon peers do not follow the constant-speed prediction assumed by
        the corridor. Their actual candidate trajectories are checked jointly
        with OBB/SAT after every local pool has been generated.
        """

        return _TrafficEnvelope(
            front=(
                envelope.front
                if envelope.front is not None and not envelope.front.is_platoon
                else None
            ),
            rear=(
                envelope.rear
                if envelope.rear is not None and not envelope.rear.is_platoon
                else None
            ),
        )

    @staticmethod
    def _serialize_envelope(envelope: _TrafficEnvelope) -> dict:
        def _item(value: _Neighbor | None) -> dict | None:
            if value is None:
                return None
            return {
                "name": value.name,
                "is_platoon": bool(value.is_platoon),
                "delta_s_m": float(value.delta_s_m),
                "bumper_gap_m": float(value.bumper_gap_m),
                "speed_mps": float(value.speed_mps),
                "ttc_s": None if not np.isfinite(value.ttc_s) else float(value.ttc_s),
            }

        return {"front": _item(envelope.front), "rear": _item(envelope.rear)}

    def _candidate_lateral_targets(
        self,
        action: int,
        start_d: float,
        desired_end_d: float,
        source_lane,
        target_lane,
    ) -> tuple[float, ...]:
        if int(action) == 0:
            margin = self.keep_lateral_margin_m
            lane_half_width = 0.5 * float(getattr(source_lane, "width", 3.5) or 3.5)
            values = [
                np.clip(start_d, -lane_half_width, lane_half_width),
                np.clip(desired_end_d, -lane_half_width, lane_half_width),
                np.clip(desired_end_d - margin, -lane_half_width, lane_half_width),
                np.clip(desired_end_d + margin, -lane_half_width, lane_half_width),
            ]
        else:
            margin = min(
                self.lane_change_target_margin_m,
                0.25 * float(getattr(target_lane, "width", 3.5) or 3.5),
            )
            values = [desired_end_d - margin, desired_end_d, desired_end_d + margin]
        return tuple(dict.fromkeys(round(float(value), 4) for value in values))

    def _lane_change_durations(
        self,
        *,
        action: int,
        lane_end_restricted: bool,
    ) -> tuple[float, ...]:
        if int(action) == 0:
            return (self.HORIZON_S,)
        if not lane_end_restricted:
            return self.LANE_CHANGE_DURATIONS_S
        return tuple(
            dict.fromkeys(
                self.URGENT_LANE_CHANGE_DURATIONS_S
                + self.LANE_CHANGE_DURATIONS_S
            )
        )

    def _committed_lane_change_durations(
        self,
        durations: tuple[float, ...],
        elapsed_s: float,
    ) -> tuple[tuple[float, ...], float]:
        remaining = max(
            self.OUTPUT_DT_S,
            max(self.LANE_CHANGE_DURATIONS_S) - max(float(elapsed_s), 0.0),
        )
        values = tuple(
            sorted({min(float(value), remaining) for value in durations})
        )
        return values, float(remaining)

    def _lane_end_restricted(
        self,
        *,
        action: int,
        ego_speed_mps: float,
        usable_source_progress_m: float,
    ) -> bool:
        if int(action) == 0:
            return False
        earliest_standard_s = min(self.LANE_CHANGE_DURATIONS_S)
        fastest_standard_progress = self._longitudinal_progress(
            ego_speed_mps,
            self.MAX_ACCEL_MPS2,
            np.asarray([earliest_standard_s], dtype=np.float64),
        )[0]
        return bool(
            usable_source_progress_m
            <= max(float(fastest_standard_progress), 8.0) + 1e-6
        )

    def _lane_change_completion_progress(
        self,
        progress: np.ndarray,
        *,
        duration_s: float,
        start_delay_s: float,
    ) -> float:
        completion = float(
            np.interp(
                min(float(duration_s), self.HORIZON_S),
                self._dense_times,
                progress,
            )
        )
        start = float(
            np.interp(
                float(start_delay_s),
                self._dense_times,
                progress,
            )
        )
        if float(duration_s) > self.HORIZON_S:
            observed_duration = max(
                self.HORIZON_S - float(start_delay_s),
                self.DENSE_DT_S,
            )
            full_duration = max(
                float(duration_s) - float(start_delay_s),
                observed_duration,
            )
            completion = start + (completion - start) * (
                full_duration / observed_duration
            )
        return float(completion)

    def _candidate_collision_names(
        self,
        candidate: np.ndarray,
        *,
        env,
        vehicle,
        times: np.ndarray,
        include_platoon: bool,
    ) -> list[str]:
        if not self.hard_collision_check_enabled:
            return []
        predictions = self._predicted_obstacles(
            env,
            vehicle,
            times,
            include_platoon=include_platoon,
        )
        return self._collision_names_against_predictions(
            candidate,
            self._vehicle_dimensions(vehicle),
            predictions,
        )

    def _predicted_obstacles(
        self,
        env,
        vehicle,
        times: np.ndarray,
        *,
        include_platoon: bool,
    ) -> list[tuple[str, np.ndarray, tuple[float, float]]]:
        platoon_ids = {
            id(value) for value in (getattr(env, "agents", {}) or {}).values()
        }
        predictions = []
        for other_id, other in self._surrounding_vehicles(env):
            if other is vehicle:
                continue
            if not include_platoon and id(other) in platoon_ids:
                continue
            predicted = self._predict_vehicle_trajectory(env, other, times)
            predictions.append(
                (
                    str(getattr(other, "name", other_id)),
                    predicted,
                    self._vehicle_dimensions(other),
                )
            )
        return predictions

    def _collision_names_against_predictions(
        self,
        candidate: np.ndarray,
        ego_dimensions: tuple[float, float],
        predictions: list[tuple[str, np.ndarray, tuple[float, float]]],
    ) -> list[str]:
        if not self.hard_collision_check_enabled:
            return []
        names = []
        margins = self._ramped_obb_margins(
            len(candidate),
            longitudinal_margin_m=self.collision_margin_m,
            lateral_margin_m=self.collision_margin_m,
        )
        for name, predicted, dimensions in predictions:
            if self._obb_overlap_series(
                candidate,
                ego_dimensions,
                predicted,
                dimensions,
                margins,
            ):
                names.append(name)
        return names

    def _background_clearance_penalty(
        self,
        candidate: np.ndarray,
        ego_dimensions: tuple[float, float],
        predictions: list[tuple[str, np.ndarray, tuple[float, float]]],
    ) -> float:
        """Softly prefer trajectories that preserve the 8 m background gap."""

        if self.safety_weight <= 0.0 or not predictions:
            return 0.0
        candidate = np.asarray(candidate, dtype=np.float64)
        ego_long = np.column_stack(
            [np.cos(candidate[:, 2]), np.sin(candidate[:, 2])]
        )
        ego_lat = np.column_stack([-ego_long[:, 1], ego_long[:, 0]])
        minimum_gap = float("inf")
        for _, predicted, dimensions in predictions:
            predicted = np.asarray(predicted, dtype=np.float64)
            if predicted.shape != candidate.shape:
                continue
            delta = predicted[:, :2] - candidate[:, :2]
            longitudinal = np.abs(np.einsum("ij,ij->i", delta, ego_long))
            lateral = np.abs(np.einsum("ij,ij->i", delta, ego_lat))
            lateral_limit = (
                0.5 * float(ego_dimensions[1])
                + 0.5 * float(dimensions[1])
                + self.collision_margin_m
            )
            same_corridor = lateral <= lateral_limit
            if not np.any(same_corridor):
                continue
            bumper_gap = longitudinal - (
                0.5 * float(ego_dimensions[0])
                + 0.5 * float(dimensions[0])
            )
            minimum_gap = min(
                minimum_gap,
                float(np.min(bumper_gap[same_corridor])),
            )
        if not np.isfinite(minimum_gap) or minimum_gap >= self.background_safe_gap_m:
            return 0.0
        deficit = self.background_safe_gap_m - max(minimum_gap, 0.0)
        return float(
            2.0
            * self.safety_weight
            * deficit
            / max(self.background_safe_gap_m, 1e-6)
        )

    def _candidate_collides_with_predicted_vehicles(
        self,
        candidate: np.ndarray,
        *,
        env=None,
        vehicle=None,
        duration: float | None = None,
    ) -> bool:
        if env is None or vehicle is None or candidate is None or len(candidate) == 0:
            return False
        times = np.linspace(
            0.0,
            max(float(duration or self.HORIZON_S), 0.0),
            len(candidate),
            dtype=np.float64,
        )
        return bool(
            self._candidate_collision_names(
                np.asarray(candidate, dtype=np.float64),
                env=env,
                vehicle=vehicle,
                times=times,
                include_platoon=True,
            )
        )

    def _predict_vehicle_trajectory(
        self,
        env,
        vehicle,
        times: np.ndarray,
    ) -> np.ndarray:
        lane = getattr(vehicle, "lane", None)
        position = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float64)
        heading = float(getattr(vehicle, "heading_theta", 0.0))
        speed = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
        if lane is not None:
            try:
                start_s, start_d = lane.local_coordinates(position)
                continuation_lane = self._get_continuation_lane(env, vehicle, lane)
                continuation = self._continuation_context(lane, continuation_lane)
                source_length = float(getattr(lane, "length", 0.0) or 0.0)
                total_length = source_length + float(continuation["remaining_length"])
                rows = []
                for time_s in times:
                    longitudinal = float(
                        np.clip(float(start_s) + speed * float(time_s), 0.0, total_length)
                    )
                    if longitudinal <= source_length or continuation_lane is None:
                        point = lane.position(longitudinal, float(start_d))
                        yaw = (
                            float(lane.heading_theta_at(longitudinal))
                            if hasattr(lane, "heading_theta_at")
                            else heading
                        )
                    else:
                        continuation_s = float(continuation["s_base"]) + (
                            longitudinal - source_length
                        )
                        point = continuation_lane.position(
                            float(
                                np.clip(
                                    continuation_s,
                                    0.0,
                                    float(
                                        getattr(continuation_lane, "length", 0.0)
                                        or 0.0
                                    ),
                                )
                            ),
                            float(start_d) + float(continuation["d_offset"]),
                        )
                        yaw = (
                            float(continuation_lane.heading_theta_at(continuation_s))
                            if hasattr(continuation_lane, "heading_theta_at")
                            else heading
                        )
                    rows.append([float(point[0]), float(point[1]), yaw])
                predicted = np.asarray(rows, dtype=np.float64)
                if np.isfinite(predicted).all():
                    return predicted
            except Exception:
                pass
        velocity = self._vehicle_velocity_xy(vehicle)
        xy = position[None, :] + np.asarray(times)[:, None] * velocity[None, :]
        headings = np.full((len(times), 1), heading, dtype=np.float64)
        return np.concatenate([xy, headings], axis=1)

    def _trajectory_pair_collides(
        self,
        first: np.ndarray,
        first_vehicle,
        second: np.ndarray,
        second_vehicle,
    ) -> bool:
        return self._obb_overlap_series(
            first,
            self._vehicle_dimensions(first_vehicle),
            second,
            self._vehicle_dimensions(second_vehicle),
            self._ramped_obb_margins(
                len(first),
                longitudinal_margin_m=self.collision_margin_m,
                lateral_margin_m=self.collision_margin_m,
            ),
        )

    def _ramped_obb_margins(
        self,
        count: int,
        *,
        longitudinal_margin_m: float,
        lateral_margin_m: float,
    ) -> np.ndarray:
        margins = np.empty((int(count), 2), dtype=np.float64)
        margins[:, 0] = float(longitudinal_margin_m)
        margins[:, 1] = float(lateral_margin_m)
        grace_samples = min(
            int(round(0.5 / self.DENSE_DT_S)),
            max(int(count) - 1, 0),
        )
        if grace_samples > 0:
            scale = np.linspace(0.0, 1.0, grace_samples + 1)
            margins[: grace_samples + 1] *= scale[:, None]
        return margins

    @staticmethod
    def _obb_overlap_series(
        first: np.ndarray,
        first_dimensions: tuple[float, float],
        second: np.ndarray,
        second_dimensions: tuple[float, float],
        margin_m,
    ) -> bool:
        return obb_overlap_series(
            first,
            first_dimensions,
            second,
            second_dimensions,
            margin_m,
        )

    def _score_candidate(
        self,
        candidate: np.ndarray,
        *,
        target_world: np.ndarray,
        source_lane,
        target_lane,
        action: int,
        desired_end_d: float,
        env=None,
        vehicle=None,
        duration: float | None = None,
        precomputed_ttc_penalty: float | None = None,
    ) -> float:
        if candidate.shape[0] < 2:
            return float("inf")
        endpoint = candidate[-1, :2]
        end_distance = float(
            np.linalg.norm(endpoint - np.asarray(target_world[:2], dtype=np.float64))
        )
        try:
            end_d = float(source_lane.local_coordinates(endpoint)[1])
        except Exception:
            return float("inf")
        end_d_error = abs(end_d - float(desired_end_d))
        segments = np.linalg.norm(np.diff(candidate[:, :2], axis=0), axis=1)
        if np.any(segments < -1e-9):
            return float("inf")
        heading_delta = np.abs(self._wrap_angle(np.diff(candidate[:, 2])))
        moving = segments > 1e-4
        curvature = (
            float(np.mean(heading_delta[moving] / segments[moving]))
            if np.any(moving)
            else 0.0
        )
        return (
            end_distance
            + 1.5 * end_d_error
            + 1.5 * curvature
            + self._safety_distance_penalty(candidate, env=env, vehicle=vehicle)
            + (
                float(precomputed_ttc_penalty)
                if precomputed_ttc_penalty is not None
                else self._ttc_penalty(
                    candidate,
                    source_lane=source_lane,
                    env=env,
                    vehicle=vehicle,
                    duration=duration,
                )
            )
        )

    def _safety_distance_penalty(self, candidate, *, env=None, vehicle=None) -> float:
        if self.safety_weight <= 0.0 or env is None:
            return 0.0
        minimum = self._min_distance_to_other_agents(candidate, env=env, vehicle=vehicle)
        if not np.isfinite(minimum) or minimum >= self.safety_distance_m:
            return 0.0
        return float(
            self.safety_weight
            * (self.safety_distance_m - minimum)
            / max(self.safety_distance_m, 1e-6)
        )

    def _ttc_penalty(
        self,
        candidate,
        *,
        source_lane,
        env=None,
        vehicle=None,
        duration=None,
    ) -> float:
        if self.ttc_weight <= 0.0 or env is None or vehicle is None:
            return 0.0
        envelope = self._traffic_envelope(env, vehicle, source_lane)
        if envelope.front is None or not np.isfinite(envelope.front.ttc_s):
            return 0.0
        if envelope.front.ttc_s >= self.ttc_threshold_s:
            return 0.0
        return float(
            self.ttc_weight
            * (self.ttc_threshold_s - envelope.front.ttc_s)
            / max(self.ttc_threshold_s, 1e-6)
        )

    def _min_distance_to_other_agents(self, candidate, *, env=None, vehicle=None) -> float:
        agents = getattr(env, "agents", {}) or {}
        minimum = float("inf")
        for other in agents.values():
            if other is vehicle:
                continue
            try:
                other_xy = np.asarray(other.position[:2], dtype=np.float64)
            except Exception:
                continue
            minimum = min(
                minimum,
                float(
                    np.min(
                        np.linalg.norm(
                            np.asarray(candidate[:, :2], dtype=np.float64)
                            - other_xy[None, :],
                            axis=1,
                        )
                    )
                ),
            )
        return minimum

    @staticmethod
    def _continuation_context(source_lane, continuation_lane) -> dict[str, float]:
        if continuation_lane is None:
            return {"s_base": 0.0, "d_offset": 0.0, "remaining_length": 0.0}
        source_length = float(getattr(source_lane, "length", 0.0) or 0.0)
        continuation_length = float(
            getattr(continuation_lane, "length", 0.0) or 0.0
        )
        try:
            stitch = np.asarray(source_lane.position(source_length, 0.0)[:2])
            raw_s, raw_d = continuation_lane.local_coordinates(stitch)
            s_base = float(np.clip(raw_s, 0.0, continuation_length))
            d_offset = float(raw_d)
        except Exception:
            s_base = 0.0
            d_offset = 0.0
        return {
            "s_base": s_base,
            "d_offset": d_offset,
            "remaining_length": max(continuation_length - s_base, 0.0),
        }

    @staticmethod
    def _get_continuation_lane(env, vehicle, source_lane):
        lane_index = getattr(source_lane, "index", None)
        if lane_index is None or len(lane_index) < 3:
            return None
        source_length = float(getattr(source_lane, "length", 0.0) or 0.0)
        try:
            source_end = np.asarray(source_lane.position(source_length, 0.0)[:2])
        except Exception:
            return None
        road_network = getattr(
            getattr(getattr(env, "engine", None), "current_map", None),
            "road_network",
            None,
        )
        candidates = []
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None:
            candidates.extend(getattr(navigation, "next_ref_lanes", None) or [])
        if road_network is not None:
            for lanes in (
                (getattr(road_network, "graph", None) or {})
                .get(lane_index[1], {})
                .values()
            ):
                candidates.extend(lanes)
        best = None
        best_distance = float("inf")
        for lane in candidates:
            try:
                distance = float(
                    np.linalg.norm(np.asarray(lane.position(0.0, 0.0)[:2]) - source_end)
                )
            except Exception:
                continue
            if distance < best_distance:
                best = lane
                best_distance = distance
        if best_distance > 10.0 and road_network is not None:
            for end_dict in (getattr(road_network, "graph", None) or {}).values():
                for lanes in end_dict.values():
                    for lane in lanes:
                        if lane is source_lane:
                            continue
                        try:
                            distance = float(
                                np.linalg.norm(
                                    np.asarray(lane.position(0.0, 0.0)[:2])
                                    - source_end
                                )
                            )
                        except Exception:
                            continue
                        if distance < best_distance:
                            best = lane
                            best_distance = distance
        return best if best_distance <= 10.0 else None

    @staticmethod
    def _resolve_target_lane(env, source_lane, action: int):
        if int(action) == 0:
            return source_lane
        lane_index = tuple(getattr(source_lane, "index", ()) or ())
        if len(lane_index) < 3:
            return None
        road_network = getattr(
            getattr(getattr(env, "engine", None), "current_map", None),
            "road_network",
            None,
        )
        if road_network is None or not hasattr(road_network, "get_lane"):
            return None
        if lane_index == ("3C0_1_", "4G0_0_", 2) and int(action) == 1:
            try:
                return road_network.get_lane(("3C0_1_", "4G1_0_", 0))
            except Exception:
                return None
        try:
            return road_network.get_lane(
                (lane_index[0], lane_index[1], int(lane_index[2]) + int(action))
            )
        except Exception:
            return None

    @staticmethod
    def _lane_from_index(env, lane_index: tuple):
        road_network = getattr(
            getattr(getattr(env, "engine", None), "current_map", None),
            "road_network",
            None,
        )
        if road_network is None or not hasattr(road_network, "get_lane"):
            return None
        try:
            return road_network.get_lane(tuple(lane_index))
        except Exception:
            return None

    @staticmethod
    def _ego_local_to_world(vehicle, target_point: np.ndarray) -> np.ndarray:
        position = np.asarray(vehicle.position[:2], dtype=np.float64)
        heading = float(getattr(vehicle, "heading_theta", 0.0))
        forward, lateral = float(target_point[0]), float(target_point[1])
        return position + np.asarray(
            [
                math.cos(heading) * forward - math.sin(heading) * lateral,
                math.sin(heading) * forward + math.cos(heading) * lateral,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _desired_end_lateral(
        source_lane,
        target_lane,
        action: int,
        desired_end_s: float,
        target_world: np.ndarray,
    ) -> float:
        if int(action) == 0:
            target_d = float(source_lane.local_coordinates(target_world)[1])
            lane_half_width = 0.2 * float(
                getattr(source_lane, "width", 3.5) or 3.5
            )
            return float(np.clip(target_d, -lane_half_width, lane_half_width))
        reference = np.asarray(
            target_lane.position(
                float(
                    np.clip(
                        desired_end_s,
                        0.0,
                        float(getattr(target_lane, "length", desired_end_s)),
                    )
                ),
                0.0,
            )[:2]
        )
        return float(source_lane.local_coordinates(reference)[1])

    @staticmethod
    def _solve_quintic_coefficients(
        d0,
        d0_dot,
        d0_ddot,
        df,
        df_dot,
        df_ddot,
        duration,
    ) -> np.ndarray:
        a0 = d0
        a1 = d0_dot
        a2 = d0_ddot / 2.0
        matrix = np.asarray(
            [
                [duration**3, duration**4, duration**5],
                [3 * duration**2, 4 * duration**3, 5 * duration**4],
                [6 * duration, 12 * duration**2, 20 * duration**3],
            ],
            dtype=np.float64,
        )
        rhs = np.asarray(
            [
                df - (a0 + a1 * duration + a2 * duration**2),
                df_dot - (a1 + 2 * a2 * duration),
                df_ddot - 2 * a2,
            ],
            dtype=np.float64,
        )
        a3, a4, a5 = np.linalg.solve(matrix, rhs)
        return np.asarray([a0, a1, a2, a3, a4, a5], dtype=np.float64)

    @staticmethod
    def _evaluate_quintic(coefficients, time_s) -> tuple[float, float, float]:
        d = sum(value * time_s**index for index, value in enumerate(coefficients))
        d_dot = sum(
            index * value * time_s ** (index - 1)
            for index, value in enumerate(coefficients)
            if index >= 1
        )
        d_ddot = sum(
            index * (index - 1) * value * time_s ** (index - 2)
            for index, value in enumerate(coefficients)
            if index >= 2
        )
        return float(d), float(d_dot), float(d_ddot)

    @staticmethod
    def _append_heading(
        points_xy: np.ndarray,
        *,
        default_heading: float,
    ) -> np.ndarray:
        deltas = np.diff(points_xy, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        headings = np.full((len(points_xy),), float(default_heading), dtype=np.float64)
        last_heading = float(default_heading)
        for index, (delta, length) in enumerate(zip(deltas, lengths)):
            if length > 1e-6:
                last_heading = math.atan2(float(delta[1]), float(delta[0]))
            headings[index] = last_heading
        if len(headings) > 1:
            headings[-1] = last_heading
        return np.concatenate([points_xy, headings[:, None]], axis=1)

    @staticmethod
    def _wrap_angle(value):
        return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _surrounding_vehicles(env) -> list[tuple[object, object]]:
        vehicles: list[tuple[object, object]] = []
        seen: set[int] = set()
        agents = getattr(env, "agents", {}) or {}
        iterable = agents.items() if isinstance(agents, dict) else enumerate(agents)
        for vehicle_id, vehicle in iterable:
            if vehicle is None or id(vehicle) in seen:
                continue
            seen.add(id(vehicle))
            vehicles.append((vehicle_id, vehicle))
        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        traffic = getattr(traffic_manager, "traffic_vehicles", None)
        if traffic is None:
            traffic = getattr(traffic_manager, "_traffic_vehicles", []) or []
        try:
            traffic = list(traffic)
        except TypeError:
            traffic = []
        for index, vehicle in enumerate(traffic):
            if vehicle is None or id(vehicle) in seen:
                continue
            seen.add(id(vehicle))
            vehicles.append((getattr(vehicle, "name", f"traffic_{index}"), vehicle))
        return vehicles

    @staticmethod
    def _vehicle_dimensions(vehicle) -> tuple[float, float]:
        length = getattr(vehicle, "LENGTH", getattr(vehicle, "length", 5.74))
        width = getattr(vehicle, "WIDTH", getattr(vehicle, "width", 2.3))
        try:
            length = float(length)
            width = float(width)
        except (TypeError, ValueError):
            return 5.74, 2.3
        if not np.isfinite(length) or length <= 0.0:
            length = 5.74
        if not np.isfinite(width) or width <= 0.0:
            width = 2.3
        return length, width

    @staticmethod
    def _vehicle_velocity_xy(vehicle) -> np.ndarray:
        try:
            velocity = np.asarray(vehicle.velocity[:2], dtype=np.float64)
        except Exception:
            speed = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
            heading = float(getattr(vehicle, "heading_theta", 0.0))
            velocity = np.asarray(
                [speed * math.cos(heading), speed * math.sin(heading)],
                dtype=np.float64,
            )
        return velocity if velocity.shape == (2,) and np.isfinite(velocity).all() else np.zeros(2)

    def _fallback_keep_trajectory(self, vehicle) -> np.ndarray:
        lane = getattr(vehicle, "lane", None)
        speed = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
        times = np.arange(1, 9, dtype=np.float64) * self.OUTPUT_DT_S
        position = np.asarray(vehicle.position[:2], dtype=np.float64)
        heading = float(getattr(vehicle, "heading_theta", 0.0))
        if lane is None:
            forward = np.asarray([math.cos(heading), math.sin(heading)])
            xy = position[None, :] + (speed * times)[:, None] * forward[None, :]
            return np.concatenate(
                [xy, np.full((8, 1), heading, dtype=np.float64)], axis=1
            ).astype(np.float32)
        try:
            start_s, start_d = lane.local_coordinates(position)
            rows = []
            for time_s in times:
                longitudinal = float(
                    np.clip(
                        float(start_s) + speed * float(time_s),
                        0.0,
                        float(getattr(lane, "length", 0.0) or 0.0),
                    )
                )
                world = lane.position(longitudinal, float(start_d))
                yaw = (
                    float(lane.heading_theta_at(longitudinal))
                    if hasattr(lane, "heading_theta_at")
                    else heading
                )
                rows.append([float(world[0]), float(world[1]), yaw])
            return np.asarray(rows, dtype=np.float32)
        except Exception:
            forward = np.asarray([math.cos(heading), math.sin(heading)])
            xy = position[None, :] + (speed * times)[:, None] * forward[None, :]
            return np.concatenate(
                [xy, np.full((8, 1), heading, dtype=np.float64)], axis=1
            ).astype(np.float32)


class JointTrajectoryExecutor:
    """Roll an accepted three-agent native plan without restarting its maneuver."""

    TRACKING_LONGITUDINAL_LIMIT_M = 1.0
    TRACKING_LATERAL_LIMIT_M = 0.5
    TRACKING_HEADING_LIMIT_RAD = 0.1

    def __init__(self, planner: PlatoonNormalPlanner) -> None:
        self.planner = planner
        self._plan: JointTrajectoryExecutionPlan | None = None
        self._last_debug: dict | None = None

    @property
    def active(self) -> bool:
        return self._plan is not None

    @property
    def plan(self) -> JointTrajectoryExecutionPlan | None:
        return self._plan

    def reset(self) -> None:
        self._plan = None
        self._last_debug = None

    def start(self, env, plan: JointTrajectoryExecutionPlan) -> None:
        if self._plan is not None:
            raise CommittedTrajectoryError(
                "a committed joint trajectory is already active",
                reason_code="committed_trajectory_tracking_deviation",
                debug={"active_execution_id": self._plan.execution_id},
            )
        expected = tuple(str(value) for value in plan.agent_specs)
        active = tuple(
            value for value in expected if value in (getattr(env, "agents", {}) or {})
        )
        if active != expected or not plan.committed_agents:
            raise CommittedTrajectoryError(
                "execution plan does not contain three active agents and a commitment",
                reason_code="committed_trajectory_lane_chain_invalid",
                debug={"expected_agents": expected, "active_agents": active},
            )
        self._plan = plan
        self._last_debug = {
            "trajectory_source": "new_native_plan",
            "execution_id": int(plan.execution_id),
            "source_proposal_id": int(plan.proposal_id),
            "source_proposal_rank": int(plan.proposal_rank),
            "execution_start_step": int(plan.start_step),
            "committed_agents": list(plan.committed_agents),
            "completion_deadline_s": float(plan.completion_deadline_s),
        }

    def get_last_debug(self) -> dict | None:
        return copy.deepcopy(self._last_debug)

    def roll(self, env) -> RolledJointTrajectory:
        plan = self._plan
        if plan is None:
            raise RuntimeError("no committed joint trajectory is active")
        config = getattr(env, "config", {}) or {}
        decision_dt_s = float(config.get("physics_world_step_size", 0.02)) * float(
            config.get("decision_repeat", 5)
        )
        current_step = int(getattr(env, "_scenario_step_count", 0) or 0)
        elapsed_s = max(0.0, float(current_step - plan.start_step) * decision_dt_s)
        if elapsed_s > float(plan.completion_deadline_s) + decision_dt_s + 1e-6:
            debug = self._base_debug(plan, elapsed_s)
            debug["completion_reason"] = "deadline_missed"
            self._last_debug = debug
            raise CommittedTrajectoryError(
                "committed lane change did not enter its target lane before deadline",
                reason_code="committed_trajectory_deadline_missed",
                debug=debug,
            )

        dense_offsets = np.arange(0, 41, dtype=np.float64) * self.planner.DENSE_DT_S
        sparse_offsets = np.arange(1, 9, dtype=np.float64) * self.planner.OUTPUT_DT_S
        agents = getattr(env, "agents", {}) or {}
        dense_by_agent: dict[str, np.ndarray] = {}
        world: dict[str, np.ndarray] = {}
        local: dict[str, np.ndarray] = {}
        longitudinal_references: dict[str, LongitudinalTrackingReference] = {}
        per_agent: dict[str, dict] = {}

        for agent_id, spec in plan.agent_specs.items():
            vehicle = agents.get(agent_id)
            if vehicle is None:
                self._raise(
                    plan,
                    elapsed_s,
                    "committed trajectory agent is no longer active",
                    "committed_trajectory_tracking_deviation",
                    {"agent_id": agent_id},
                )
            for lane_index in (
                spec.source_lane_index,
                spec.target_lane_index,
                spec.continuation_lane_index,
            ):
                if lane_index and self.planner._lane_from_index(env, lane_index) is None:
                    self._raise(
                        plan,
                        elapsed_s,
                        "committed trajectory lane chain is unavailable",
                        "committed_trajectory_lane_chain_invalid",
                        {"agent_id": agent_id, "lane_index": lane_index},
                    )
            planned_now = self._sample_spec(spec, np.asarray([elapsed_s]))[0]
            tracking = self._tracking_error(vehicle, planned_now)
            if (
                abs(tracking["longitudinal_m"])
                > self.TRACKING_LONGITUDINAL_LIMIT_M
                or abs(tracking["lateral_m"]) > self.TRACKING_LATERAL_LIMIT_M
                or abs(tracking["heading_rad"]) > self.TRACKING_HEADING_LIMIT_RAD
            ):
                self._raise(
                    plan,
                    elapsed_s,
                    "vehicle tracking error left the committed trajectory envelope",
                    "committed_trajectory_tracking_deviation",
                    {"agent_id": agent_id, "tracking_error": tracking},
                )
            current_pose = np.asarray(
                [
                    float(vehicle.position[0]),
                    float(vehicle.position[1]),
                    float(getattr(vehicle, "heading_theta", 0.0)),
                ],
                dtype=np.float64,
            )
            speed = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
            try:
                actual_arc, path_error = project_point_to_path_arc(
                    current_pose[:2], spec.trajectory_world[:, :2], spec.path_arc_m
                )
                path_distance = np.diff(spec.path_arc_m)
                path_heading_delta = np.abs(
                    np.arctan2(
                        np.sin(np.diff(spec.trajectory_world[:, 2])),
                        np.cos(np.diff(spec.trajectory_world[:, 2])),
                    )
                )
                path_curvature = path_heading_delta / np.maximum(
                    path_distance, 1.0e-3
                )
                segment_speed_limit = np.where(
                    path_curvature > 1.0e-6,
                    np.sqrt(6.0 / np.maximum(path_curvature, 1.0e-6)),
                    self.planner.MAX_SPEED_MPS,
                )
                path_speed_limit = np.concatenate(
                    (segment_speed_limit[:1], segment_speed_limit)
                )
                path_speed_limit = np.clip(
                    0.97 * path_speed_limit, 0.1, self.planner.MAX_SPEED_MPS
                )
                longitudinal_reference = build_feedback_executable_profile(
                    path_times_s=spec.sample_times_s,
                    path_arc_m=spec.path_arc_m,
                    elapsed_s=elapsed_s,
                    actual_arc_m=actual_arc,
                    actual_speed_mps=speed,
                    path_speed_limit_mps=path_speed_limit,
                    source="committed_roll",
                )
                dense_arc = np.interp(
                    dense_offsets,
                    longitudinal_reference.sample_times_s,
                    longitudinal_reference.arc_position_m,
                )
                dense_future = sample_path_at_arc(
                    spec.trajectory_world,
                    spec.path_arc_m,
                    dense_arc[1:],
                )
                dense = np.concatenate((current_pose[None, :], dense_future), axis=0)
                output = sample_path_at_arc(
                    spec.trajectory_world,
                    spec.path_arc_m,
                    longitudinal_reference.arc_position_m[1:],
                ).astype(np.float32)
            except LongitudinalReferenceError as exc:
                self._raise(
                    plan,
                    elapsed_s,
                    "committed longitudinal reference cannot be rolled",
                    "committed_trajectory_kinematic_infeasible",
                    {"agent_id": agent_id, "longitudinal_reference_error": str(exc)},
                )
            local_output = world_trajectory_to_ego_local(output, current_pose)
            world_audit = validate_trajectory_kinematics(
                output, speed, current_pose, HardModeMaskConfig()
            )
            local_audit = validate_trajectory_kinematics(
                local_output,
                speed,
                np.zeros((3,), dtype=np.float64),
                HardModeMaskConfig(),
            )
            if not world_audit.valid or not local_audit.valid:
                self._raise(
                    plan,
                    elapsed_s,
                    "rolled trajectory failed the unified kinematic contract",
                    "committed_trajectory_kinematic_infeasible",
                    {
                        "agent_id": agent_id,
                        "world_violations": list(world_audit.violations),
                        "local_violations": list(local_audit.violations),
                        "tracking_error": tracking,
                        "world_max_speed_mps": float(
                            world_audit.speed_mps.max(initial=0.0)
                        ),
                        "world_min_acceleration_mps2": float(
                            world_audit.acceleration_mps2.min(initial=0.0)
                        ),
                        "world_max_acceleration_mps2": float(
                            world_audit.acceleration_mps2.max(initial=0.0)
                        ),
                        "world_max_curvature_per_m": float(
                            world_audit.curvature_per_m.max(initial=0.0)
                        ),
                        "world_max_lateral_acceleration_mps2": float(
                            world_audit.lateral_acceleration_mps2.max(initial=0.0)
                        ),
                    },
                )
            footprint_on_road, footprint_detail = self._footprint_road_audit(
                env, vehicle, dense, spec
            )
            if not footprint_on_road:
                self._raise(
                    plan,
                    elapsed_s,
                    "rolled trajectory footprint leaves the road",
                    "committed_trajectory_out_of_road",
                    {"agent_id": agent_id, "road_audit": footprint_detail},
                )
            background = self.planner._predicted_obstacles(
                env,
                vehicle,
                self.planner._dense_times,
                include_platoon=False,
            )
            collision_names = self.planner._collision_names_against_predictions(
                dense,
                self.planner._vehicle_dimensions(vehicle),
                background,
            )
            minimum_background_gap = self._minimum_gap(
                dense,
                self.planner._vehicle_dimensions(vehicle),
                background,
            )
            if collision_names or minimum_background_gap < self.planner.background_safe_gap_m - 1e-6:
                self._raise(
                    plan,
                    elapsed_s,
                    "rolled trajectory violates background safety",
                    "committed_trajectory_background_unsafe",
                    {
                        "agent_id": agent_id,
                        "collision_objects": collision_names,
                        "minimum_background_gap_m": minimum_background_gap,
                    },
                )
            dense_by_agent[agent_id] = dense
            world[agent_id] = np.ascontiguousarray(output)
            local[agent_id] = np.ascontiguousarray(local_output)
            longitudinal_references[agent_id] = longitudinal_reference
            per_agent[agent_id] = {
                "tracking_error": tracking,
                "spatial_path_projection_error_m": float(path_error),
                "original_arc_error_m": float(
                    longitudinal_reference.original_arc_error_m
                ),
                "reference_speed_mps": float(
                    longitudinal_reference.target_speed_mps
                ),
                "reference_acceleration_mps2": float(
                    longitudinal_reference.feedforward_acceleration_mps2
                ),
                "minimum_background_gap_m": minimum_background_gap,
                "selected_lane_change_duration_s": float(
                    spec.lane_change_duration_s
                ),
                "reference_cursor_s": float(elapsed_s),
            }

        minimum_platoon_gap = float("inf")
        ordered_ids = tuple(plan.agent_specs)
        for first_index in range(len(ordered_ids)):
            for second_index in range(first_index + 1, len(ordered_ids)):
                first_id = ordered_ids[first_index]
                second_id = ordered_ids[second_index]
                first_dense = dense_by_agent[first_id]
                second_dense = dense_by_agent[second_id]
                if self.planner._trajectory_pair_collides(
                    first_dense,
                    agents[first_id],
                    second_dense,
                    agents[second_id],
                ):
                    self._raise(
                        plan,
                        elapsed_s,
                        "rolled joint trajectory has a pairwise OBB conflict",
                        "committed_trajectory_pairwise_unsafe",
                        {"pair": [first_id, second_id], "obb_collision": True},
                    )
                pair_gap = self._minimum_pair_gap(
                    first_dense,
                    self.planner._vehicle_dimensions(agents[first_id]),
                    second_dense,
                    self.planner._vehicle_dimensions(agents[second_id]),
                )
                minimum_platoon_gap = min(minimum_platoon_gap, pair_gap)
                if pair_gap < self.planner.platoon_safe_gap_m - 1e-6:
                    self._raise(
                        plan,
                        elapsed_s,
                        "rolled joint trajectory violates platoon safety gap",
                        "committed_trajectory_pairwise_unsafe",
                        {"pair": [first_id, second_id], "minimum_gap_m": pair_gap},
                    )

        debug = self._base_debug(plan, elapsed_s)
        debug.update(
            rolling_hard_audit="passed",
            minimum_platoon_gap_m=minimum_platoon_gap,
            agents=per_agent,
        )
        self._last_debug = debug
        return RolledJointTrajectory(
            execution_id=int(plan.execution_id),
            elapsed_s=float(elapsed_s),
            trajectories_world=world,
            trajectories_local=local,
            longitudinal_references=longitudinal_references,
            rule_actions=dict(plan.rule_actions),
            debug=debug,
        )

    def _raise(
        self,
        plan: JointTrajectoryExecutionPlan,
        elapsed_s: float,
        message: str,
        reason_code: str,
        detail: Mapping[str, object],
    ) -> None:
        debug = self._base_debug(plan, elapsed_s)
        debug.update(rolling_hard_audit="failed", failure_detail=dict(detail))
        self._last_debug = debug
        raise CommittedTrajectoryError(
            message, reason_code=reason_code, debug=debug
        )

    @staticmethod
    def _base_debug(plan: JointTrajectoryExecutionPlan, elapsed_s: float) -> dict:
        return {
            "trajectory_source": "committed_roll",
            "execution_id": int(plan.execution_id),
            "source_proposal_id": int(plan.proposal_id),
            "source_proposal_rank": int(plan.proposal_rank),
            "execution_start_step": int(plan.start_step),
            "elapsed_s": float(elapsed_s),
            "remaining_s": max(float(plan.completion_deadline_s) - elapsed_s, 0.0),
            "committed_agents": list(plan.committed_agents),
            "completion_deadline_s": float(plan.completion_deadline_s),
        }

    @staticmethod
    def _sample_spec(spec: TrajectoryExecutionSpec, query_times: np.ndarray) -> np.ndarray:
        query = np.asarray(query_times, dtype=np.float64)
        if query.size and (
            float(np.min(query)) < -1e-9
            or float(np.max(query)) > float(spec.sample_times_s[-1]) + 1e-9
        ):
            raise CommittedTrajectoryError(
                "committed trajectory buffer is exhausted",
                reason_code="committed_trajectory_deadline_missed",
                debug={"maximum_time_s": float(spec.sample_times_s[-1])},
            )
        x = np.interp(query, spec.sample_times_s, spec.trajectory_world[:, 0])
        y = np.interp(query, spec.sample_times_s, spec.trajectory_world[:, 1])
        headings = np.unwrap(spec.trajectory_world[:, 2])
        heading = np.interp(query, spec.sample_times_s, headings)
        heading = np.arctan2(np.sin(heading), np.cos(heading))
        return np.column_stack((x, y, heading))

    @staticmethod
    def _tracking_error(vehicle, planned_pose: np.ndarray) -> dict[str, float]:
        heading = float(planned_pose[2])
        delta = np.asarray(vehicle.position[:2], dtype=np.float64) - planned_pose[:2]
        longitudinal = float(delta[0] * math.cos(heading) + delta[1] * math.sin(heading))
        lateral = float(-delta[0] * math.sin(heading) + delta[1] * math.cos(heading))
        heading_error = float(
            math.atan2(
                math.sin(float(getattr(vehicle, "heading_theta", 0.0)) - heading),
                math.cos(float(getattr(vehicle, "heading_theta", 0.0)) - heading),
            )
        )
        return {
            "longitudinal_m": longitudinal,
            "lateral_m": lateral,
            "heading_rad": heading_error,
        }

    def _footprint_on_road(self, env, vehicle, trajectory: np.ndarray, spec: TrajectoryExecutionSpec) -> bool:
        valid, _ = self._footprint_road_audit(env, vehicle, trajectory, spec)
        return valid

    def _footprint_road_audit(
        self,
        env,
        vehicle,
        trajectory: np.ndarray,
        spec: TrajectoryExecutionSpec,
    ) -> tuple[bool, dict]:
        lanes = [
            self.planner._lane_from_index(env, lane_index)
            for lane_index in (
                spec.source_lane_index,
                spec.target_lane_index,
                spec.continuation_lane_index,
            )
            if lane_index
        ]
        return audit_dense_footprint_on_lanes(
            trajectory,
            lanes,
            self.planner._vehicle_dimensions(vehicle),
            dense_dt_s=self.planner.DENSE_DT_S,
        )

    @classmethod
    def _minimum_gap(
        cls,
        trajectory: np.ndarray,
        dimensions: tuple[float, float],
        predictions: list[tuple[str, np.ndarray, tuple[float, float]]],
    ) -> float:
        value = float("inf")
        for _, predicted, other_dimensions in predictions:
            value = min(
                value,
                cls._minimum_pair_gap(
                    trajectory, dimensions, predicted, other_dimensions
                ),
            )
        return value

    @staticmethod
    def _minimum_pair_gap(
        first: np.ndarray,
        first_dimensions: tuple[float, float],
        second: np.ndarray,
        second_dimensions: tuple[float, float],
    ) -> float:
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        if first.shape != second.shape:
            return -float("inf")
        forward = np.column_stack((np.cos(first[:, 2]), np.sin(first[:, 2])))
        lateral_axis = np.column_stack((-forward[:, 1], forward[:, 0]))
        delta = second[:, :2] - first[:, :2]
        longitudinal = np.abs(np.einsum("ij,ij->i", delta, forward))
        lateral = np.abs(np.einsum("ij,ij->i", delta, lateral_axis))
        lateral_limit = 0.5 * (float(first_dimensions[1]) + float(second_dimensions[1]))
        same_corridor = lateral <= lateral_limit + 1e-6
        if not np.any(same_corridor):
            return float("inf")
        bumper = longitudinal - 0.5 * (
            float(first_dimensions[0]) + float(second_dimensions[0])
        )
        return float(np.min(bumper[same_corridor]))
