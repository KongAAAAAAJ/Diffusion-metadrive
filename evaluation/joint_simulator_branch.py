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
    replay_position_error_m: np.ndarray
    replay_heading_error_rad: np.ndarray
    executed_steps: np.ndarray
    minimum_platoon_gap_m: np.ndarray
    minimum_background_gap_m: np.ndarray
    failure_reasons: tuple[tuple[str, ...], ...]

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
            "start_seed": spec.seed,
            "num_scenarios": 1,
            **dict(spec.env_config),
        }
        env = self._env_factory(config)
        setter = getattr(env, "set_runtime_scenario_route", None)
        if not callable(setter):
            raise JointRewardError("branch environment has no route setter")
        setter(spec.scenario_id, spec.local_route)
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
        executed = np.zeros(group_size, dtype=np.int64)
        minimum_platoon = np.full(group_size, np.inf, dtype=np.float64)
        minimum_background = np.full(group_size, np.inf, dtype=np.float64)
        failure_reasons: list[set[str]] = [set() for _ in range(group_size)]

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
                references = [
                    _local_reference_to_world(candidates[group, role], initial[role])
                    for role in range(3)
                ]
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
                    _, _, terminated, truncated, info = env.step(action)
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
                            out[group] = True
                            failure_reasons[group].add(
                                "simulator:agent_removed_without_failure_flag"
                            )
                        break
                    current = capture_joint_pose_global(env)
                    for role, agent_id in enumerate(AGENT_IDS):
                        positions[role].append(current[role, :2].copy())
                        headings[role].append(float(current[role, 2]))
                        speeds[role].append(
                            float(
                                getattr(
                                    env.agents[agent_id], "speed_km_h", 0.0
                                )
                            )
                            / 3.6
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
            finally:
                env.close()

        # An absent background vehicle is represented as a large finite gap in
        # the report, never as an NaN/inf sentinel.
        minimum_background[~np.isfinite(minimum_background)] = 1.0e6
        minimum_platoon[~np.isfinite(minimum_platoon)] = 1.0e6
        reward = compose_joint_reward(
            progress=progress,
            formation=formation,
            clearance=clearance,
            comfort=comfort,
            collision=collision,
            out_of_drivable=out,
            config=self.config,
        )
        return SimulatorBranchResult(
            reward=reward,
            replay_position_error_m=replay_position.astype(np.float32),
            replay_heading_error_rad=replay_heading.astype(np.float32),
            executed_steps=executed,
            minimum_platoon_gap_m=minimum_platoon.astype(np.float32),
            minimum_background_gap_m=minimum_background.astype(np.float32),
            failure_reasons=tuple(
                tuple(sorted(values)) for values in failure_reasons
            ),
        )


__all__ = [
    "JointEpisodeSpec",
    "JointSimulatorBranchEvaluator",
    "SimulatorBranchResult",
    "capture_joint_pose_global",
]
