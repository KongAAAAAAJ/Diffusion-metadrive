"""Deterministic four-second simulator branches for joint reward calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from expert_dataset.collect_joint_bev import (
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from models.bev_planner.joint_reward import (
    AGENT_IDS,
    JointRewardConfig,
    JointRewardError,
    JointRewardResult,
    compose_joint_reward,
)
from models.platoon_planner.platoon_normal_planner import PlatoonNormalPlanner
from models.controller.longitudinal_reference import (
    LongitudinalTrackingReference,
    signed_longitudinal_speed_mps,
)
from scenarios.bev_round13_contract import deterministic_initial_speed_km_h


@dataclass(frozen=True)
class JointEpisodeSpec:
    scenario_id: str
    local_route: str
    seed: int
    reference_pose_global: np.ndarray
    env_config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.local_route:
            raise JointRewardError("episode scenario and route must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise JointRewardError("episode seed must be an integer")
        pose = np.asarray(self.reference_pose_global)
        if (
            pose.shape != (3, 3)
            or not np.issubdtype(pose.dtype, np.floating)
            or not np.isfinite(pose).all()
        ):
            raise JointRewardError(
                "reference_pose_global must be finite floating-point [3,3]"
            )
        object.__setattr__(
            self,
            "reference_pose_global",
            np.ascontiguousarray(pose, dtype=np.float64),
        )
        object.__setattr__(self, "env_config", dict(self.env_config))


@dataclass(frozen=True)
class SimulatorBranchResult:
    reward: JointRewardResult
    initial_speed_mps: np.ndarray
    replay_position_error_m: np.ndarray
    replay_heading_error_rad: np.ndarray
    executed_steps: np.ndarray
    minimum_platoon_gap_m: np.ndarray
    minimum_background_gap_m: np.ndarray
    failure_reasons: tuple[tuple[str, ...], ...]
    tracking_longitudinal_error_m: np.ndarray
    tracking_lateral_error_m: np.ndarray
    tracking_heading_error_rad: np.ndarray
    reference_curvature_max_per_m: np.ndarray
    minimum_road_clearance_m: np.ndarray
    tracking_traces: tuple[tuple[Mapping[str, object], ...], ...]

    def __post_init__(self) -> None:
        group_size = self.reward.rewards.shape[0]
        for name in (
            "replay_position_error_m",
            "replay_heading_error_rad",
            "minimum_platoon_gap_m",
            "minimum_background_gap_m",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (group_size,) or not np.isfinite(value).all():
                raise JointRewardError(f"{name} must be finite [G]")
        for name in (
            "tracking_longitudinal_error_m",
            "tracking_lateral_error_m",
            "tracking_heading_error_rad",
            "reference_curvature_max_per_m",
            "minimum_road_clearance_m",
            "initial_speed_mps",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (group_size, 3) or not np.isfinite(value).all():
                raise JointRewardError(f"{name} must be finite [G,3]")
        steps = np.asarray(self.executed_steps)
        if (
            steps.shape != (group_size,)
            or not np.issubdtype(steps.dtype, np.integer)
            or np.any(steps <= 0)
        ):
            raise JointRewardError("executed_steps must be positive integer [G]")
        if (
            not isinstance(self.failure_reasons, tuple)
            or len(self.failure_reasons) != group_size
            or any(
                not isinstance(group, tuple)
                or any(not isinstance(value, str) for value in group)
                for group in self.failure_reasons
            )
        ):
            raise JointRewardError("failure_reasons must contain one tuple per group")
        if (
            not isinstance(self.tracking_traces, tuple)
            or len(self.tracking_traces) != group_size
            or any(
                not isinstance(group, tuple)
                or len(group) != 3
                or any(not isinstance(role, Mapping) for role in group)
                for group in self.tracking_traces
            )
        ):
            raise JointRewardError(
                "tracking_traces must contain three role mappings per group"
            )


def capture_joint_pose_global(env: object) -> np.ndarray:
    poses = []
    agents = getattr(env, "agents", {}) or {}
    for agent_id in AGENT_IDS:
        vehicle = agents.get(agent_id)
        if vehicle is None:
            raise JointRewardError(f"simulator is missing {agent_id}")
        position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float64)
        heading = float(getattr(vehicle, "heading_theta", float("nan")))
        if (
            position.size < 2
            or not np.isfinite(position[:2]).all()
            or not math.isfinite(heading)
        ):
            raise JointRewardError(f"invalid simulator pose for {agent_id}")
        poses.append((position[0], position[1], heading))
    return np.asarray(poses, dtype=np.float64)


def _wrap(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.arctan2(np.sin(array), np.cos(array))


def _local_reference_to_world(
    trajectory: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    source = np.concatenate(
        (
            pose.reshape(1, 3),
            np.zeros((8, 3), dtype=np.float64),
        ),
        axis=0,
    )
    cos_h = math.cos(float(pose[2]))
    sin_h = math.sin(float(pose[2]))
    source[1:, 0] = (
        pose[0] + cos_h * trajectory[:, 0] - sin_h * trajectory[:, 1]
    )
    source[1:, 1] = (
        pose[1] + sin_h * trajectory[:, 0] + cos_h * trajectory[:, 1]
    )
    source[1:, 2] = _wrap(pose[2] + trajectory[:, 2])
    return source


def _world_reference_to_current_local(
    world_reference: np.ndarray,
    current_pose: np.ndarray,
    elapsed_s: float,
) -> np.ndarray:
    source_times = np.arange(9, dtype=np.float64) * 0.5
    query_times = np.minimum(
        elapsed_s + np.arange(1, 9, dtype=np.float64) * 0.5,
        4.0,
    )
    target = np.empty((8, 3), dtype=np.float64)
    target[:, 0] = np.interp(query_times, source_times, world_reference[:, 0])
    target[:, 1] = np.interp(query_times, source_times, world_reference[:, 1])
    target[:, 2] = _wrap(
        np.interp(
            query_times,
            source_times,
            np.unwrap(world_reference[:, 2]),
        )
    )
    delta = target[:, :2] - current_pose[:2]
    cos_h = math.cos(float(current_pose[2]))
    sin_h = math.sin(float(current_pose[2]))
    local = np.empty((8, 3), dtype=np.float32)
    local[:, 0] = cos_h * delta[:, 0] + sin_h * delta[:, 1]
    local[:, 1] = -sin_h * delta[:, 0] + cos_h * delta[:, 1]
    local[:, 2] = _wrap(target[:, 2] - current_pose[2])
    return local


def _world_reference_pose_at(
    world_reference: np.ndarray, elapsed_s: float
) -> np.ndarray:
    source_times = np.arange(9, dtype=np.float64) * 0.5
    query = float(np.clip(elapsed_s, 0.0, 4.0))
    return np.asarray(
        [
            np.interp(query, source_times, world_reference[:, 0]),
            np.interp(query, source_times, world_reference[:, 1]),
            _wrap(
                np.interp(
                    query,
                    source_times,
                    np.unwrap(world_reference[:, 2]),
                )
            ).item(),
        ],
        dtype=np.float64,
    )


def _maximum_reference_curvature(world_reference: np.ndarray) -> float:
    distances = np.linalg.norm(np.diff(world_reference[:, :2], axis=0), axis=1)
    headings = np.unwrap(world_reference[:, 2])
    curvature = np.abs(np.diff(headings)) / np.maximum(distances, 1.0e-3)
    return float(curvature.max(initial=0.0))


def _reference_arc_kinematics(
    world_reference: np.ndarray, elapsed_s: float
) -> tuple[float, float, float]:
    points = np.asarray(world_reference[:, :2], dtype=np.float64)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    source_times = np.arange(9, dtype=np.float64) * 0.5
    query = float(np.clip(elapsed_s, 0.0, 4.0))
    arc_position = float(np.interp(query, source_times, cumulative))
    segment_speed = np.diff(cumulative) / 0.5
    segment_centers = source_times[:-1] + 0.25
    speed = float(
        np.interp(
            query,
            segment_centers,
            segment_speed,
            left=segment_speed[0],
            right=segment_speed[-1],
        )
    )
    acceleration_values = np.gradient(segment_speed, 0.5)
    acceleration = float(
        np.interp(
            query,
            segment_centers,
            acceleration_values,
            left=acceleration_values[0],
            right=acceleration_values[-1],
        )
    )
    return arc_position, speed, acceleration


def _fixed_world_longitudinal_reference(
    world_reference: np.ndarray,
    current_pose: np.ndarray,
    elapsed_s: float,
) -> LongitudinalTrackingReference:
    """Preserve fixed-world timing without turning position lag into speed."""

    offsets = np.arange(9, dtype=np.float64) * 0.5
    absolute = np.clip(float(elapsed_s) + offsets, 0.0, 4.0)
    kinematics = np.asarray(
        [_reference_arc_kinematics(world_reference, value) for value in absolute],
        dtype=np.float64,
    )
    longitudinal, _, _, _ = _tracking_error_against_reference(
        np.asarray(current_pose, dtype=np.float64),
        world_reference,
        float(elapsed_s),
    )
    speed = kinematics[:, 1]
    acceleration = np.clip(kinematics[:, 2], -8.0, 5.0)
    stop_requested = bool(
        np.allclose(kinematics[-2:, 0], kinematics[-1, 0], atol=1.0e-8)
        and speed[-1] <= 1.0e-6
    )
    return LongitudinalTrackingReference(
        sample_times_s=offsets,
        arc_position_m=kinematics[:, 0],
        speed_mps=np.clip(speed, 0.0, 100.0 / 3.6),
        acceleration_mps2=acceleration,
        original_arc_error_m=float(-longitudinal),
        stop_requested=stop_requested,
        source="simulator_branch",
    )


def _tracking_error_against_reference(
    actual_pose: np.ndarray,
    world_reference: np.ndarray,
    elapsed_s: float,
) -> tuple[float, float, float, np.ndarray]:
    """Separate path-tracking error from longitudinal timing lag."""
    points = np.asarray(world_reference[:, :2], dtype=np.float64)
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance = float("inf")
    best_index = 0
    best_fraction = 0.0
    best_point = points[0]
    for index, (start, segment, length) in enumerate(
        zip(points[:-1], segments, lengths)
    ):
        if length <= 1.0e-9:
            fraction = 0.0
        else:
            fraction = float(
                np.clip(
                    np.dot(actual_pose[:2] - start, segment) / (length * length),
                    0.0,
                    1.0,
                )
            )
        projection = start + fraction * segment
        distance = float(np.linalg.norm(actual_pose[:2] - projection))
        if distance < best_distance:
            best_distance = distance
            best_index = index
            best_fraction = fraction
            best_point = projection

    headings = np.unwrap(np.asarray(world_reference[:, 2], dtype=np.float64))
    nearest_heading = float(
        headings[best_index]
        + best_fraction * (headings[best_index + 1] - headings[best_index])
    )
    nearest_arc = (
        cumulative[best_index] + best_fraction * lengths[best_index]
    )
    source_times = np.arange(9, dtype=np.float64) * 0.5
    expected_arc = float(
        np.interp(np.clip(elapsed_s, 0.0, 4.0), source_times, cumulative)
    )
    delta = np.asarray(actual_pose[:2] - best_point, dtype=np.float64)
    cos_h = math.cos(nearest_heading)
    sin_h = math.sin(nearest_heading)
    lateral = -sin_h * delta[0] + cos_h * delta[1]
    heading = float(_wrap(actual_pose[2] - nearest_heading))
    nearest_pose = np.asarray(
        [best_point[0], best_point[1], _wrap(nearest_heading).item()],
        dtype=np.float64,
    )
    return float(nearest_arc - expected_arc), float(lateral), heading, nearest_pose


def _road_clearance_m(vehicle: object) -> float:
    lane = getattr(vehicle, "lane", None)
    if lane is None or not hasattr(lane, "local_coordinates"):
        return 1.0e6
    try:
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        width_at = getattr(lane, "width_at", None)
        width = (
            float(width_at(longitudinal))
            if callable(width_at)
            else float(getattr(lane, "width", 1.0e6))
        )
        vehicle_width = float(getattr(vehicle, "WIDTH", 2.3))
        value = 0.5 * width - abs(float(lateral)) - 0.5 * vehicle_width
    except (AttributeError, TypeError, ValueError):
        return 1.0e6
    return value if math.isfinite(value) else 1.0e6


def _trajectory_target_speed_mps(trajectory: np.ndarray) -> float:
    points = np.concatenate(
        (np.zeros((1, 2), dtype=np.float64), trajectory[:2, :2]), axis=0
    )
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _has_failure(info: Mapping[str, object]) -> tuple[bool, bool]:
    collision = False
    out = False
    for agent_id in AGENT_IDS:
        value = info.get(agent_id)
        if not isinstance(value, Mapping):
            continue
        collision |= any(
            bool(value.get(name, False))
            for name in (
                "crash",
                "crash_vehicle",
                "crash_object",
                "crash_building",
                "crash_human",
            )
        )
        out |= bool(value.get("out_of_road", False)) or bool(
            value.get("out_of_route", False)
        )
    return collision, out


class JointSimulatorBranchEvaluator:
    """Recreate and replay each candidate branch from one physical state."""

    def __init__(
        self,
        config: JointRewardConfig | None = None,
        *,
        env_factory: Callable[[Mapping[str, object]], object] | None = None,
    ) -> None:
        self.config = config or JointRewardConfig()
        self._env_factory = env_factory or SensorlessJointBEVPlatoonEnv
        self._vehicle_helper = PlatoonNormalPlanner(
            background_safe_gap_m=self.config.background_safe_gap_m,
            platoon_safe_gap_m=self.config.platoon_safe_gap_m,
        )

    @staticmethod
    def _validate_prefix(
        prefix_actions: Sequence[Mapping[str, np.ndarray]],
    ) -> tuple[dict[str, np.ndarray], ...]:
        checked = []
        for step in prefix_actions:
            if not isinstance(step, Mapping) or set(step) != set(AGENT_IDS):
                raise JointRewardError(
                    "each prefix action must contain exactly agent0/1/2"
                )
            values = {}
            for agent_id in AGENT_IDS:
                trajectory = np.asarray(step[agent_id])
                if (
                    trajectory.shape != (8, 3)
                    or not np.issubdtype(trajectory.dtype, np.floating)
                    or not np.isfinite(trajectory).all()
                ):
                    raise JointRewardError(
                        "prefix trajectories must be finite floating-point [8,3]"
                    )
                values[agent_id] = np.ascontiguousarray(
                    trajectory, dtype=np.float32
                )
            checked.append(values)
        return tuple(checked)

    def _make_env(self, spec: JointEpisodeSpec) -> object:
        config = {
            "num_agents": 3,
            "traffic_density": 0.0,
            "initial_speed_km_h": deterministic_initial_speed_km_h(
                spec.scenario_id, spec.seed
            ),
            "start_seed": spec.seed,
            "num_scenarios": 1,
            **dict(spec.env_config),
        }
        env = self._env_factory(config)
        setter = getattr(env, "set_runtime_scenario_route", None)
        if not callable(setter):
            raise JointRewardError("branch environment has no route setter")
        setter(spec.scenario_id, spec.local_route)
        spawn_manager = getattr(
            getattr(env, "engine", None), "spawn_manager", None
        )
        set_spawn_seed = getattr(spawn_manager, "set_episode_spawn_seed", None)
        if callable(set_spawn_seed):
            set_spawn_seed(spec.seed)
        try:
            env.reset(seed=spec.seed)
        except TypeError:
            # Test-only minimal env factories may expose the pre-Gym reset API.
            env.reset()
        return env

    def _instant_gaps(self, env: object) -> tuple[float, float]:
        agents = getattr(env, "agents", {}) or {}
        platoon_gap = float("inf")
        for leader, follower in zip(AGENT_IDS[:-1], AGENT_IDS[1:]):
            first = np.asarray(agents[leader].position, dtype=np.float64)[:2]
            second = np.asarray(agents[follower].position, dtype=np.float64)[:2]
            platoon_gap = min(
                platoon_gap,
                float(np.linalg.norm(first - second))
                - self.config.vehicle_length_m,
            )

        background_gap = float("inf")
        vehicles = [
            vehicle
            for _, vehicle in self._vehicle_helper._surrounding_vehicles(env)
        ]
        platoon_objects = {id(vehicle) for vehicle in agents.values()}
        for agent_id in AGENT_IDS:
            vehicle = agents[agent_id]
            position = np.asarray(vehicle.position, dtype=np.float64)[:2]
            for other in vehicles:
                if id(other) in platoon_objects:
                    continue
                other_position = np.asarray(
                    getattr(other, "position", ()), dtype=np.float64
                ).reshape(-1)
                if other_position.size < 2 or not np.isfinite(
                    other_position[:2]
                ).all():
                    continue
                other_length = float(
                    getattr(other, "LENGTH", self.config.vehicle_length_m)
                )
                background_gap = min(
                    background_gap,
                    float(np.linalg.norm(position - other_position[:2]))
                    - 0.5 * (self.config.vehicle_length_m + other_length),
                )
        return platoon_gap, background_gap

    def evaluate(
        self,
        episode_spec: JointEpisodeSpec,
        prefix_actions: Sequence[Mapping[str, np.ndarray]],
        trajectories: np.ndarray,
    ) -> SimulatorBranchResult:
        candidates = np.asarray(trajectories)
        if (
            candidates.ndim != 4
            or candidates.shape[1:] != (3, 8, 3)
            or not np.issubdtype(candidates.dtype, np.floating)
            or not np.isfinite(candidates).all()
            or candidates.shape[0] <= 0
        ):
            raise JointRewardError(
                "branch trajectories must be finite floating-point [G,3,8,3]"
            )
        candidates = np.ascontiguousarray(candidates, dtype=np.float32)
        prefix = self._validate_prefix(prefix_actions)
        group_size = candidates.shape[0]
        progress = np.zeros(group_size, dtype=np.float64)
        formation = np.zeros(group_size, dtype=np.float64)
        clearance = np.zeros(group_size, dtype=np.float64)
        comfort = np.zeros(group_size, dtype=np.float64)
        collision = np.zeros(group_size, dtype=np.bool_)
        out = np.zeros(group_size, dtype=np.bool_)
        replay_position = np.zeros(group_size, dtype=np.float64)
        replay_heading = np.zeros(group_size, dtype=np.float64)
        initial_speed = np.zeros((group_size, 3), dtype=np.float64)
        executed = np.zeros(group_size, dtype=np.int64)
        minimum_platoon = np.full(group_size, np.inf, dtype=np.float64)
        minimum_background = np.full(group_size, np.inf, dtype=np.float64)
        failure_reasons: list[set[str]] = [set() for _ in range(group_size)]
        tracking_longitudinal = np.zeros((group_size, 3), dtype=np.float64)
        tracking_lateral = np.zeros((group_size, 3), dtype=np.float64)
        tracking_heading = np.zeros((group_size, 3), dtype=np.float64)
        reference_curvature = np.zeros((group_size, 3), dtype=np.float64)
        road_clearance = np.full((group_size, 3), np.inf, dtype=np.float64)
        tracking_traces: list[list[dict[str, object]]] = [
            [
                {
                    "reference_world": [],
                    "nearest_reference_world": [],
                    "actual_world": [],
                    "longitudinal_errors_m": [],
                    "lateral_errors_m": [],
                    "heading_errors_rad": [],
                    "road_clearance_m": [],
                    "steering": [],
                    "throttle": [],
                    "target_speed_mps": [],
                    "reference_arc_position_m": [],
                    "actual_projected_arc_position_m": [],
                    "reference_feedforward_speed_mps": [],
                    "reference_feedforward_acceleration_mps2": [],
                    "position_error_speed_increment_mps": [],
                    "formation_gap_error_m": [],
                    "formation_control_increment": [],
                    "actual_speed_mps": [],
                    "actual_acceleration_mps2": [],
                    "desired_acceleration_mps2": [],
                    "compensated_acceleration_mps2": [],
                    "raw_desired_acceleration_mps2": [],
                    "acceleration_bias_mps2": [],
                    "drive_acceleration_scale_mps2": [],
                    "brake_acceleration_scale_mps2": [],
                    "controller_speed_error_mps": [],
                    "controller_preview_speed_mps": [],
                    "controller_preview_acceleration_mps2": [],
                    "position_feedback_mps2": [],
                    "gap_feedback_mps2": [],
                    "speed_integral": [],
                    "speed_overzero_guard": [],
                    "control_regime": [],
                    "control_saturated": [],
                    "lateral_heading_contaminated": [],
                    "maximum_continuous_saturation_s": 0.0,
                }
                for _ in range(3)
            ]
            for _ in range(group_size)
        ]

        for group in range(group_size):
            env = self._make_env(episode_spec)
            try:
                for action in prefix:
                    env.step(action)
                replay_pose = capture_joint_pose_global(env)
                replay_position[group] = float(
                    np.linalg.norm(
                        replay_pose[:, :2] - episode_spec.reference_pose_global[:, :2],
                        axis=1,
                    ).max()
                )
                replay_heading[group] = float(
                    np.abs(
                        _wrap(
                            replay_pose[:, 2]
                            - episode_spec.reference_pose_global[:, 2]
                        )
                    ).max()
                )
                if replay_position[group] > 0.01 or replay_heading[group] > 0.01:
                    raise JointRewardError(
                        "branch prefix replay did not reproduce the calibration state"
                    )

                initial = replay_pose.copy()
                for role, agent_id in enumerate(AGENT_IDS):
                    initial_speed[group, role] = signed_longitudinal_speed_mps(
                        env.agents[agent_id]
                    )
                references = [
                    _local_reference_to_world(candidates[group, role], initial[role])
                    for role in range(3)
                ]
                for role in range(3):
                    reference_curvature[group, role] = (
                        _maximum_reference_curvature(references[role])
                    )
                dt_s = simulator_decision_dt_s(env)
                maximum_steps = max(1, int(math.ceil(4.0 / dt_s)))
                positions = [[initial[role, :2].copy()] for role in range(3)]
                speeds = [[] for _ in range(3)]
                headings = [[initial[role, 2]] for role in range(3)]
                formation_errors = []
                clearance_deficits = []

                for step_index in range(maximum_steps):
                    current = capture_joint_pose_global(env)
                    elapsed = step_index * dt_s
                    action = {
                        agent_id: _world_reference_to_current_local(
                            references[role],
                            current[role],
                            elapsed,
                        )
                        for role, agent_id in enumerate(AGENT_IDS)
                    }
                    explicit_references = {
                        agent_id: _fixed_world_longitudinal_reference(
                            references[role], current[role], elapsed
                        )
                        for role, agent_id in enumerate(AGENT_IDS)
                    }
                    controller_diagnostics = []
                    for role, agent_id in enumerate(AGENT_IDS):
                        feedforward_speed = float(
                            explicit_references[agent_id].target_speed_mps
                        )
                        feedforward_acceleration = float(
                            explicit_references[
                                agent_id
                            ].feedforward_acceleration_mps2
                        )
                        target_speed = float(
                            explicit_references[agent_id].target_speed_mps
                        )
                        current_speed = signed_longitudinal_speed_mps(
                            env.agents[agent_id]
                        )
                        speed_only_control = float(
                            np.clip(
                                -0.35 * (current_speed - target_speed),
                                -1.0,
                                1.0,
                            )
                        )
                        if role == 0:
                            formation_gap_error = 0.0
                        else:
                            follower_heading = float(current[role, 2])
                            relative = (
                                current[role - 1, :2] - current[role, :2]
                            )
                            actual_gap = float(
                                math.cos(follower_heading) * relative[0]
                                + math.sin(follower_heading) * relative[1]
                            )
                            desired_gap = float(
                                getattr(env, "_desired_center_spacing_m")(
                                    agent_id, AGENT_IDS[role - 1]
                                )
                            )
                            formation_gap_error = desired_gap - actual_gap
                        controller_diagnostics.append(
                            {
                                "feedforward_speed": feedforward_speed,
                                "feedforward_acceleration": (
                                    feedforward_acceleration
                                ),
                                "target_speed": target_speed,
                                "current_speed": current_speed,
                                "speed_only_control": speed_only_control,
                                "formation_gap_error": formation_gap_error,
                            }
                        )
                    low_level_action = {
                        agent_id: env.trajectory_reference_to_control(
                            agent_id,
                            action[agent_id],
                            explicit_references[agent_id],
                        )
                        for agent_id in AGENT_IDS
                    }
                    controller_debug = dict(
                        getattr(env, "_last_longitudinal_control_debug", {})
                        or {}
                    )
                    _, _, terminated, truncated, info = env.step(low_level_action)
                    executed_controls = (
                        getattr(env, "_pending_low_level_actions", {}) or {}
                    )
                    controls: dict[str, np.ndarray] = {}
                    for agent_id in AGENT_IDS:
                        control = np.asarray(
                            executed_controls.get(
                                agent_id, np.asarray([0.0, 0.0])
                            ),
                            dtype=np.float64,
                        )
                        if control.shape != (2,) or not np.isfinite(control).all():
                            raise JointRewardError(
                                "branch environment recorded invalid executed control"
                            )
                        controls[agent_id] = control
                    executed[group] += 1
                    crashed, left_road = _has_failure(info)
                    collision[group] |= crashed
                    out[group] |= left_road
                    for agent_id in AGENT_IDS:
                        agent_info = info.get(agent_id)
                        if not isinstance(agent_info, Mapping):
                            continue
                        for key in (
                            "crash",
                            "crash_vehicle",
                            "crash_object",
                            "crash_building",
                            "crash_human",
                            "out_of_road",
                            "out_of_route",
                        ):
                            if bool(agent_info.get(key, False)):
                                failure_reasons[group].add(f"{agent_id}:{key}")
                    ended = (
                        crashed
                        or left_road
                        or bool(terminated.get("__all__", False))
                        or bool(truncated.get("__all__", False))
                    )
                    if any(
                        agent_id not in (getattr(env, "agents", {}) or {})
                        for agent_id in AGENT_IDS
                    ):
                        if not (crashed or left_road):
                            failure_reasons[group].add(
                                "simulator:agent_removed_without_failure_flag"
                            )
                        break
                    current = capture_joint_pose_global(env)
                    for role, agent_id in enumerate(AGENT_IDS):
                        positions[role].append(current[role, :2].copy())
                        headings[role].append(float(current[role, 2]))
                        speeds[role].append(
                            signed_longitudinal_speed_mps(env.agents[agent_id])
                        )
                        reference_pose = _world_reference_pose_at(
                            references[role], (step_index + 1) * dt_s
                        )
                        (
                            longitudinal,
                            lateral,
                            heading_error,
                            nearest_reference_pose,
                        ) = _tracking_error_against_reference(
                            current[role],
                            references[role],
                            (step_index + 1) * dt_s,
                        )
                        trace = tracking_traces[group][role]
                        reference_arc, _, _ = _reference_arc_kinematics(
                            references[role], (step_index + 1) * dt_s
                        )
                        diagnostic = controller_diagnostics[role]
                        trace["reference_world"].append(reference_pose.tolist())
                        trace["nearest_reference_world"].append(
                            nearest_reference_pose.tolist()
                        )
                        trace["actual_world"].append(current[role].tolist())
                        trace["longitudinal_errors_m"].append(longitudinal)
                        trace["lateral_errors_m"].append(lateral)
                        trace["heading_errors_rad"].append(heading_error)
                        clearance_value = _road_clearance_m(
                            env.agents[agent_id]
                        )
                        trace["road_clearance_m"].append(clearance_value)
                        trace["steering"].append(
                            float(controls[agent_id][0])
                        )
                        trace["throttle"].append(
                            float(controls[agent_id][1])
                        )
                        trace["target_speed_mps"].append(
                            diagnostic["target_speed"]
                        )
                        trace["reference_arc_position_m"].append(reference_arc)
                        trace["actual_projected_arc_position_m"].append(
                            reference_arc + longitudinal
                        )
                        trace["reference_feedforward_speed_mps"].append(
                            diagnostic["feedforward_speed"]
                        )
                        trace[
                            "reference_feedforward_acceleration_mps2"
                        ].append(diagnostic["feedforward_acceleration"])
                        trace[
                            "position_error_speed_increment_mps"
                        ].append(
                            diagnostic["target_speed"]
                            - diagnostic["feedforward_speed"]
                        )
                        trace["formation_gap_error_m"].append(
                            diagnostic["formation_gap_error"]
                        )
                        trace["formation_control_increment"].append(
                            float(controls[agent_id][1])
                            - diagnostic["speed_only_control"]
                        )
                        trace["actual_speed_mps"].append(speeds[role][-1])
                        trace["actual_acceleration_mps2"].append(
                            (
                                speeds[role][-1]
                                - diagnostic["current_speed"]
                            )
                            / dt_s
                        )
                        debug = controller_debug.get(agent_id, {})
                        trace["desired_acceleration_mps2"].append(
                            float(debug.get("desired_acceleration_mps2", 0.0))
                        )
                        trace["compensated_acceleration_mps2"].append(
                            float(debug.get("compensated_acceleration_mps2", 0.0))
                        )
                        trace["raw_desired_acceleration_mps2"].append(
                            float(debug.get("raw_desired_acceleration_mps2", 0.0))
                        )
                        trace["acceleration_bias_mps2"].append(
                            float(debug.get("acceleration_bias_mps2", 0.0))
                        )
                        trace["drive_acceleration_scale_mps2"].append(
                            float(
                                debug.get(
                                    "drive_acceleration_scale_mps2", 0.0
                                )
                            )
                        )
                        trace["brake_acceleration_scale_mps2"].append(
                            float(
                                debug.get(
                                    "brake_acceleration_scale_mps2", 0.0
                                )
                            )
                        )
                        trace["controller_speed_error_mps"].append(
                            float(debug.get("speed_error_mps", 0.0))
                        )
                        trace["controller_preview_speed_mps"].append(
                            float(debug.get("reference_speed_mps", 0.0))
                        )
                        trace[
                            "controller_preview_acceleration_mps2"
                        ].append(
                            float(debug.get("reference_acceleration_mps2", 0.0))
                        )
                        trace["position_feedback_mps2"].append(
                            float(debug.get("position_feedback_mps2", 0.0))
                        )
                        trace["gap_feedback_mps2"].append(
                            float(debug.get("gap_feedback_mps2", 0.0))
                        )
                        trace["speed_integral"].append(
                            float(debug.get("speed_integral", 0.0))
                        )
                        trace["speed_overzero_guard"].append(
                            bool(debug.get("speed_overzero_guard", False))
                        )
                        trace["control_regime"].append(
                            str(debug.get("control_regime", "unknown"))
                        )
                        trace["control_saturated"].append(
                            abs(float(controls[agent_id][1])) >= 0.999
                        )
                        trace["lateral_heading_contaminated"].append(
                            abs(lateral) > 0.5
                            or abs(heading_error) > 0.1
                        )
                        tracking_longitudinal[group, role] = max(
                            tracking_longitudinal[group, role],
                            abs(longitudinal),
                        )
                        tracking_lateral[group, role] = max(
                            tracking_lateral[group, role], abs(lateral)
                        )
                        tracking_heading[group, role] = max(
                            tracking_heading[group, role], abs(heading_error)
                        )
                        road_clearance[group, role] = min(
                            road_clearance[group, role], clearance_value
                        )
                    pair_error = []
                    for leader, follower in ((0, 1), (1, 2)):
                        center = float(
                            np.linalg.norm(
                                current[leader, :2] - current[follower, :2]
                            )
                        )
                        target = float(
                            getattr(env, "_desired_center_spacing_m")(
                                AGENT_IDS[follower], AGENT_IDS[leader]
                            )
                        )
                        pair_error.append(abs(center - target))
                    formation_errors.append(float(np.mean(pair_error)))
                    platoon_gap, background_gap = self._instant_gaps(env)
                    minimum_platoon[group] = min(
                        minimum_platoon[group], platoon_gap
                    )
                    minimum_background[group] = min(
                        minimum_background[group], background_gap
                    )
                    deficits = [
                        np.clip(
                            (self.config.platoon_safe_gap_m - platoon_gap)
                            / self.config.platoon_safe_gap_m,
                            0.0,
                            1.0,
                        )
                    ]
                    if math.isfinite(background_gap):
                        deficits.append(
                            np.clip(
                                (
                                    self.config.background_safe_gap_m
                                    - background_gap
                                )
                                / self.config.background_safe_gap_m,
                                0.0,
                                1.0,
                            )
                        )
                    clearance_deficits.append(float(np.mean(deficits)))
                    if ended:
                        break

                role_progress = []
                role_comfort = []
                for role in range(3):
                    displacement = positions[role][-1] - positions[role][0]
                    heading0 = initial[role, 2]
                    role_progress.append(
                        np.clip(
                            (
                                math.cos(heading0) * displacement[0]
                                + math.sin(heading0) * displacement[1]
                            )
                            / self.config.progress_norm_m,
                            0.0,
                            1.0,
                        )
                    )
                    speed_values = np.asarray(speeds[role], dtype=np.float64)
                    heading_values = np.unwrap(
                        np.asarray(headings[role], dtype=np.float64)
                    )
                    acceleration = (
                        np.diff(speed_values) / dt_s
                        if speed_values.size > 1
                        else np.zeros(1, dtype=np.float64)
                    )
                    yaw_rate = np.diff(heading_values) / dt_s
                    role_comfort.append(
                        -float(
                            np.clip(
                                0.5 * np.mean(np.abs(acceleration)) / 8.0
                                + 0.5 * np.mean(np.abs(yaw_rate)),
                                0.0,
                                1.0,
                            )
                        )
                    )
                progress[group] = float(np.mean(role_progress))
                formation[group] = -min(
                    (
                        float(np.mean(formation_errors))
                        if formation_errors
                        else self.config.formation_norm_m
                    )
                    / self.config.formation_norm_m,
                    1.0,
                )
                clearance[group] = -min(
                    (
                        float(np.mean(clearance_deficits))
                        if clearance_deficits
                        else 1.0
                    ),
                    1.0,
                )
                comfort[group] = float(np.mean(role_comfort))
                for role in range(3):
                    saturated = tracking_traces[group][role][
                        "control_saturated"
                    ]
                    longest = 0
                    current_run = 0
                    for value in saturated:
                        current_run = current_run + 1 if value else 0
                        longest = max(longest, current_run)
                    tracking_traces[group][role][
                        "maximum_continuous_saturation_s"
                    ] = longest * dt_s
            finally:
                env.close()

        # An absent background vehicle is represented as a large finite gap in
        # the report, never as an NaN/inf sentinel.
        minimum_background[~np.isfinite(minimum_background)] = 1.0e6
        minimum_platoon[~np.isfinite(minimum_platoon)] = 1.0e6
        road_clearance[~np.isfinite(road_clearance)] = 1.0e6
        clearance_violation = (
            (minimum_background < self.config.background_safe_gap_m)
            | (minimum_platoon < self.config.platoon_safe_gap_m)
        )
        reward = compose_joint_reward(
            progress=progress,
            formation=formation,
            clearance=clearance,
            comfort=comfort,
            collision=collision,
            out_of_drivable=out,
            clearance_violation=clearance_violation.astype(np.bool_),
            config=self.config,
        )
        return SimulatorBranchResult(
            reward=reward,
            initial_speed_mps=initial_speed.astype(np.float32),
            replay_position_error_m=replay_position.astype(np.float32),
            replay_heading_error_rad=replay_heading.astype(np.float32),
            executed_steps=executed,
            minimum_platoon_gap_m=minimum_platoon.astype(np.float32),
            minimum_background_gap_m=minimum_background.astype(np.float32),
            failure_reasons=tuple(
                tuple(sorted(values)) for values in failure_reasons
            ),
            tracking_longitudinal_error_m=tracking_longitudinal.astype(
                np.float32
            ),
            tracking_lateral_error_m=tracking_lateral.astype(np.float32),
            tracking_heading_error_rad=tracking_heading.astype(np.float32),
            reference_curvature_max_per_m=reference_curvature.astype(
                np.float32
            ),
            minimum_road_clearance_m=road_clearance.astype(np.float32),
            tracking_traces=tuple(
                tuple(dict(role) for role in group)
                for group in tracking_traces
            ),
        )


__all__ = [
    "JointEpisodeSpec",
    "JointSimulatorBranchEvaluator",
    "SimulatorBranchResult",
    "capture_joint_pose_global",
]
