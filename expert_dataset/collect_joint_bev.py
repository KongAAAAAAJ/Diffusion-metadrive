"""Joint-first BEV expert collection primitives.

One :class:`JointBEVSample` represents one synchronized simulator state for the
three fixed platoon roles.  This module deliberately stops before persistence:
episode splitting, shard writers, and resume semantics belong to round four.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np

from envs.observations.semantic_bev import MetaDriveSceneAdapter, SemanticBEVConfig, SimulatorSnapshot
from envs.platoon_env import PlatoonEnv
from models.bev_planner.dynamic_anchors import SimulatorDynamicAnchorGenerator
from models.bev_planner.mode_contract import (
    NUM_MODES,
    TRAJECTORY_DIM,
    TRAJECTORY_STEPS,
    ModeContractError,
    build_hard_mode_valid_mask,
    label_gt_mode,
)
from models.controller.LQRFollowerController import LQRFollowerController
from models.controller.PIDController import PIDTrajectoryController, _world_trajectory_to_ego_local
from models.decisioner.rule_decisioner import make_rule_maker, select_controller_by_formation
from models.platoon_planner.platoon_normal_planner import PlatoonNormalPlanner

try:
    from metadrive.obs.observation_base import DummyObservation
except ImportError:  # pragma: no cover - surfaced when constructing the environment
    DummyObservation = None  # type: ignore[assignment]


NUM_PLATOON_AGENTS = 3
EGO_STATE_DIM = 8
FORMATION_RELATION_DIM = 12
NUM_RELATION_NEIGHBORS = 2


class JointCollectionError(RuntimeError):
    """The synchronized joint state is unsuitable for the frozen dataset."""


class JointStepRejected(JointCollectionError):
    """A data-quality conflict requiring this entire synchronized step to be skipped."""


class AgentRole(IntEnum):
    LEADER = 0
    MIDDLE = 1
    REAR = 2


JOINT_SAMPLE_DTYPES = {
    "bev": np.dtype(np.uint8),
    "ego_state": np.dtype(np.float32),
    "ego_pose_global": np.dtype(np.float32),
    "formation_relation_state": np.dtype(np.float32),
    "relation_valid_mask": np.dtype(np.bool_),
    "agent_role": np.dtype(np.int64),
    "coarse_trajectories": np.dtype(np.float32),
    "mode_valid_mask": np.dtype(np.bool_),
    "gt_mode": np.dtype(np.int64),
    "expert_trajectory": np.dtype(np.float32),
}


JOINT_SAMPLE_SHAPES = {
    "bev": (NUM_PLATOON_AGENTS, *SemanticBEVConfig().shape),
    "ego_state": (NUM_PLATOON_AGENTS, EGO_STATE_DIM),
    "ego_pose_global": (NUM_PLATOON_AGENTS, 3),
    "formation_relation_state": (NUM_PLATOON_AGENTS, FORMATION_RELATION_DIM),
    "relation_valid_mask": (NUM_PLATOON_AGENTS, NUM_RELATION_NEIGHBORS),
    "agent_role": (NUM_PLATOON_AGENTS,),
    "coarse_trajectories": (
        NUM_PLATOON_AGENTS,
        NUM_MODES,
        TRAJECTORY_STEPS,
        TRAJECTORY_DIM,
    ),
    "mode_valid_mask": (NUM_PLATOON_AGENTS, NUM_MODES),
    "gt_mode": (NUM_PLATOON_AGENTS,),
    "expert_trajectory": (NUM_PLATOON_AGENTS, TRAJECTORY_STEPS, TRAJECTORY_DIM),
}


@dataclass(frozen=True)
class JointBEVSample:
    bev: np.ndarray
    ego_state: np.ndarray
    ego_pose_global: np.ndarray
    formation_relation_state: np.ndarray
    relation_valid_mask: np.ndarray
    agent_role: np.ndarray
    coarse_trajectories: np.ndarray
    mode_valid_mask: np.ndarray
    gt_mode: np.ndarray
    expert_trajectory: np.ndarray

    def __post_init__(self) -> None:
        for item in fields(self):
            name = item.name
            array = np.asarray(getattr(self, name))
            expected_shape = JOINT_SAMPLE_SHAPES[name]
            expected_dtype = JOINT_SAMPLE_DTYPES[name]
            if array.shape != expected_shape:
                raise JointCollectionError(
                    f"{name} must have shape {expected_shape}, got {array.shape}"
                )
            if array.dtype != expected_dtype:
                raise JointCollectionError(
                    f"{name} must have dtype {expected_dtype}, got {array.dtype}"
                )
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise JointCollectionError(f"{name} contains non-finite values")
            owned = np.ascontiguousarray(array).copy()
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

        if not np.array_equal(self.agent_role, np.asarray(list(AgentRole), dtype=np.int64)):
            raise JointCollectionError("agent_role must be [LEADER, MIDDLE, REAR] in joint-first order")
        if np.any(self.gt_mode < 0) or np.any(self.gt_mode >= NUM_MODES):
            raise JointCollectionError("gt_mode contains an index outside the fixed ten modes")
        if not np.all(self.mode_valid_mask[np.arange(NUM_PLATOON_AGENTS), self.gt_mode]):
            raise JointCollectionError("every gt_mode must be enabled by mode_valid_mask")

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the exact round-three tensor schema without metadata fields."""

        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class ExpertJointStep:
    rule_actions: Mapping[str, int]
    trajectories_world: Mapping[str, np.ndarray]
    controls: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class JointEpisodeRollout:
    samples: tuple[JointBEVSample, ...]
    simulator_steps: int
    rejected_joint_steps: int
    failure_reason: str | None
    terminated: bool
    truncated: bool


class SensorlessJointBEVPlatoonEnv(PlatoonEnv):
    """PlatoonEnv configured without visual/range observations or rendering."""

    @staticmethod
    def default_config():
        # BaseEnv._post_process_config merges sensors from self.default_config()
        # after applying user config.  Replacing this nested dictionary at the
        # source is therefore required; passing ``sensors={}`` alone is not.
        from metadrive.utils.config import Config

        config = PlatoonEnv.default_config()
        # It must remain Config: MetaDrive expects ``Config.update`` to return
        # the mapping, whereas built-in ``dict.update`` returns None.
        config.update(
            {
                "sensors": Config({}),
                "ground_truth_traffic_policy": True,
            },
            stop_recursive_update=["sensors"],
        )
        return config

    def __init__(self, config: Mapping[str, object] | None = None):
        if DummyObservation is None:
            raise RuntimeError("MetaDrive DummyObservation is unavailable")
        payload = dict(config or {})
        if int(payload.get("num_agents", NUM_PLATOON_AGENTS)) != NUM_PLATOON_AGENTS:
            raise ValueError("joint BEV collection is frozen to exactly three agents")
        payload.update(
            {
                "num_agents": NUM_PLATOON_AGENTS,
                "observation_mode": "bev_gt",
                "use_render": False,
                "image_observation": False,
                "image_on_cuda": False,
                "agent_observation": DummyObservation,
                "sensors": {},
                "interface_panel": [],
                "ground_truth_traffic_policy": True,
            }
        )
        super().__init__(payload)


class RulePlannerExpert:
    """Three-car RuleMaker + normal-planner expert with explicit outputs."""

    def __init__(self, env: PlatoonEnv, agent_ids: Sequence[str]) -> None:
        self.agent_ids = tuple(str(value) for value in agent_ids)
        if len(self.agent_ids) != NUM_PLATOON_AGENTS:
            raise JointCollectionError("RulePlannerExpert requires exactly three ordered agents")
        config = dict(getattr(env, "config", {}) or {})
        self.rule_maker = make_rule_maker(config)
        self.rule_maker.reset(env, list(self.agent_ids))
        self.planner = PlatoonNormalPlanner()
        self.pid_controller = PIDTrajectoryController(config)
        self.pid_controller.reset()
        self.lqr_controller = LQRFollowerController(config)
        self.lqr_controller.reset()

    @staticmethod
    def _normalize_decision(value: object) -> tuple[int, np.ndarray]:
        if not isinstance(value, Mapping):
            raise JointCollectionError("RuleMaker decision must contain action and target_point")
        try:
            action = int(value["action"])
            target = np.asarray(value["target_point"], dtype=np.float32).reshape(2)
        except (KeyError, TypeError, ValueError) as exc:
            raise JointCollectionError("invalid RuleMaker decision") from exc
        if action not in (-1, 0, 1) or not np.isfinite(target).all():
            raise JointCollectionError("RuleMaker decision is outside the frozen label contract")
        return action, target

    def plan(self, env: PlatoonEnv) -> ExpertJointStep:
        active = [agent_id for agent_id in self.agent_ids if agent_id in getattr(env, "agents", {})]
        if tuple(active) != self.agent_ids:
            raise JointCollectionError("all three ordered platoon agents must be active")
        raw = self.rule_maker.compute(env, active, getattr(env, "_last_planner_batch", None) or {})
        decisions: dict[str, dict[str, object]] = {}
        actions: dict[str, int] = {}
        for agent_id in self.agent_ids:
            if agent_id not in raw:
                raise JointCollectionError(f"RuleMaker omitted {agent_id}")
            action, target = self._normalize_decision(raw[agent_id])
            actions[agent_id] = action
            decisions[agent_id] = {"action": action, "target_point": target}

        rule_debug = getattr(self.rule_maker, "get_last_debug", lambda: None)() or {}
        dynamic_roles = rule_debug.get("dynamic_roles", {}) if isinstance(rule_debug, Mapping) else {}
        if dynamic_roles:
            env.apply_dynamic_roles(dynamic_roles)

        trajectories = self.planner.plan(env, decisions)
        planner_debug = self.planner.get_last_debug() or {}
        for agent_id in self.agent_ids:
            trajectory = np.asarray(trajectories.get(agent_id), dtype=np.float32)
            if trajectory.shape != (TRAJECTORY_STEPS, TRAJECTORY_DIM) or not np.isfinite(trajectory).all():
                raise JointCollectionError(f"normal planner returned an invalid trajectory for {agent_id}")
            agent_debug = planner_debug.get(agent_id)
            if not isinstance(agent_debug, Mapping) or bool(agent_debug.get("fallback_used", True)):
                raise JointCollectionError(
                    f"normal planner did not produce a native expert trajectory for {agent_id}"
                )

        controller = select_controller_by_formation(
            self.rule_maker, self.pid_controller, self.lqr_controller
        )
        controls = controller.compute_actions(env, trajectories)
        for agent_id in self.agent_ids:
            control = np.asarray(controls.get(agent_id), dtype=np.float32)
            if control.shape != (2,) or not np.isfinite(control).all():
                raise JointCollectionError(f"controller returned an invalid action for {agent_id}")
        return ExpertJointStep(
            rule_actions=actions,
            trajectories_world={
                key: np.ascontiguousarray(np.asarray(value, dtype=np.float32))
                for key, value in trajectories.items()
            },
            controls={
                key: np.ascontiguousarray(np.asarray(value, dtype=np.float32))
                for key, value in controls.items()
            },
        )


class JointBEVSampleBuilder:
    """Build synchronized joint samples from a live simulator state."""

    def __init__(
        self,
        agent_ids: Sequence[str] = ("agent0", "agent1", "agent2"),
        *,
        scene_adapter: MetaDriveSceneAdapter | None = None,
        anchor_generator: SimulatorDynamicAnchorGenerator | None = None,
    ) -> None:
        self.agent_ids = tuple(str(value) for value in agent_ids)
        if self.agent_ids != ("agent0", "agent1", "agent2"):
            raise JointCollectionError("joint-first role order is frozen to agent0, agent1, agent2")
        self.scene_adapter = scene_adapter or MetaDriveSceneAdapter()
        self.anchor_generator = anchor_generator or SimulatorDynamicAnchorGenerator()
        self.snapshots: list[SimulatorSnapshot] = []
        self._previous_motion: dict[str, tuple[float, float, float]] = {}
        self._motion_derivatives: dict[str, tuple[float, float]] = {}

    def reset(self) -> None:
        self.snapshots.clear()
        self._previous_motion.clear()
        self._motion_derivatives.clear()

    def capture_state(self, env: PlatoonEnv, timestamp_s: float) -> SimulatorSnapshot:
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise JointCollectionError("timestamp_s must be finite")
        if self.snapshots and timestamp <= self.snapshots[-1].timestamp_s:
            raise JointCollectionError("snapshot timestamps must be strictly increasing")
        snapshot = self.scene_adapter.capture_snapshot(env, timestamp)
        missing = [agent_id for agent_id in self.agent_ids if agent_id not in snapshot.platoon]
        if missing:
            raise JointCollectionError(f"snapshot is missing platoon agents: {missing}")

        for agent_id in self.agent_ids:
            vehicle = env.agents[agent_id]
            speed = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6
            heading = float(getattr(vehicle, "heading_theta", 0.0))
            acceleration = 0.0
            yaw_rate = 0.0
            previous = self._previous_motion.get(agent_id)
            if previous is not None:
                previous_time, previous_speed, previous_heading = previous
                dt_s = timestamp - previous_time
                if dt_s <= 0.0:
                    raise JointCollectionError("motion timestamps must be strictly increasing")
                acceleration = (speed - previous_speed) / dt_s
                yaw_rate = float(
                    (heading - previous_heading + np.pi) % (2.0 * np.pi) - np.pi
                ) / dt_s
            self._motion_derivatives[agent_id] = (float(acceleration), float(yaw_rate))
            self._previous_motion[agent_id] = (timestamp, speed, heading)

        self.snapshots.append(snapshot)
        oldest = timestamp - max(self.scene_adapter.config.history_offsets_s) - 0.1
        self.snapshots = [value for value in self.snapshots if value.timestamp_s >= oldest]
        return snapshot

    def history_ready(self) -> bool:
        if not self.snapshots:
            return False
        current_time = self.snapshots[-1].timestamp_s
        try:
            for offset in self.scene_adapter.config.history_offsets_s:
                self.scene_adapter._snapshot_for_offset(self.snapshots, current_time, offset)
        except ValueError:
            return False
        return True

    @staticmethod
    def _global_pose(vehicle: object) -> np.ndarray:
        position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float32).reshape(-1)
        heading = float(getattr(vehicle, "heading_theta"))
        if position.size < 2 or not np.isfinite(position[:2]).all() or not np.isfinite(heading):
            raise JointCollectionError("vehicle global pose is invalid")
        return np.asarray([position[0], position[1], heading], dtype=np.float32)

    def _ego_state(self, env: PlatoonEnv, agent_id: str) -> np.ndarray:
        vehicle = env.agents[agent_id]
        speed = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6
        acceleration, yaw_rate = self._motion_derivatives.get(agent_id, (0.0, 0.0))
        steering = float(getattr(vehicle, "steering", 0.0) or 0.0)
        throttle_brake = float(getattr(vehicle, "throttle_brake", 0.0) or 0.0)
        lane = getattr(vehicle, "lane", None)
        if lane is None:
            raise JointCollectionError(f"agent {agent_id} has no current lane")
        position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float64).reshape(-1)
        try:
            longitudinal, lateral = lane.local_coordinates(position[:2])
            heading_at = getattr(lane, "heading_theta_at", None)
            lane_heading = float(heading_at(float(longitudinal))) if callable(heading_at) else float(vehicle.heading_theta)
        except (AttributeError, TypeError, ValueError) as exc:
            raise JointCollectionError(f"agent {agent_id} lane state is unavailable") from exc
        heading_error = float(
            (float(vehicle.heading_theta) - lane_heading + np.pi) % (2.0 * np.pi) - np.pi
        )
        target_speed = float(getattr(env, "config", {}).get("target_speed_km_h", 0.0)) / 3.6
        state = np.asarray(
            [
                speed,
                acceleration,
                steering,
                throttle_brake,
                yaw_rate,
                float(lateral),
                heading_error,
                target_speed - speed,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(state).all():
            raise JointCollectionError(f"agent {agent_id} ego_state contains non-finite values")
        return state

    def _relation_valid_mask(self, env: PlatoonEnv, ego_id: str) -> np.ndarray:
        active = set(getattr(env, "agents", {}) or {})
        values = [other_id in active for other_id in self.agent_ids if other_id != ego_id]
        return np.asarray(values, dtype=np.bool_)

    def build_sample(self, env: PlatoonEnv, expert_step: ExpertJointStep) -> JointBEVSample:
        if not self.history_ready():
            raise JointCollectionError("t/t-0.5/t-1.0 simulator history is not ready")
        missing = [agent_id for agent_id in self.agent_ids if agent_id not in getattr(env, "agents", {})]
        if missing:
            raise JointCollectionError(f"joint sample is missing active agents: {missing}")

        bev_values = []
        ego_states = []
        poses = []
        relations = []
        relation_masks = []
        coarse_values = []
        mode_masks = []
        gt_modes = []
        expert_values = []

        try:
            for agent_id in self.agent_ids:
                if agent_id not in expert_step.rule_actions or agent_id not in expert_step.trajectories_world:
                    raise JointCollectionError(f"expert step omitted {agent_id}")
                vehicle = env.agents[agent_id]
                bev = self.scene_adapter.rasterize(env, agent_id, self.snapshots)
                anchors = self.anchor_generator.generate(env, agent_id)
                speed_mps = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6
                mask_result = build_hard_mode_valid_mask(
                    bev,
                    anchors.coarse_trajectories,
                    speed_mps,
                    anchors.topology,
                )
                expert_local = _world_trajectory_to_ego_local(
                    vehicle, expert_step.trajectories_world[agent_id]
                )
                if expert_local.shape != (TRAJECTORY_STEPS, TRAJECTORY_DIM):
                    raise JointCollectionError(f"expert trajectory for {agent_id} is not [8,3]")
                gt_mode = label_gt_mode(
                    expert_step.rule_actions[agent_id],
                    expert_local,
                    anchors.coarse_trajectories,
                    mask_result.valid_mask,
                )

                bev_values.append(bev)
                ego_states.append(self._ego_state(env, agent_id))
                poses.append(self._global_pose(vehicle))
                relations.append(np.asarray(env.get_formation_relation_state(agent_id), dtype=np.float32))
                relation_masks.append(self._relation_valid_mask(env, agent_id))
                coarse_values.append(anchors.coarse_trajectories)
                mode_masks.append(mask_result.valid_mask)
                gt_modes.append(gt_mode)
                expert_values.append(expert_local)
        except ModeContractError as exc:
            raise JointStepRejected(
                "one agent violated the mode/label contract; discard the entire joint step"
            ) from exc

        return JointBEVSample(
            bev=np.stack(bev_values).astype(np.uint8, copy=False),
            ego_state=np.stack(ego_states).astype(np.float32, copy=False),
            ego_pose_global=np.stack(poses).astype(np.float32, copy=False),
            formation_relation_state=np.stack(relations).astype(np.float32, copy=False),
            relation_valid_mask=np.stack(relation_masks).astype(np.bool_, copy=False),
            agent_role=np.asarray(list(AgentRole), dtype=np.int64),
            coarse_trajectories=np.stack(coarse_values).astype(np.float32, copy=False),
            mode_valid_mask=np.stack(mode_masks).astype(np.bool_, copy=False),
            gt_mode=np.asarray(gt_modes, dtype=np.int64),
            expert_trajectory=np.stack(expert_values).astype(np.float32, copy=False),
        )


def simulator_decision_dt_s(env: PlatoonEnv) -> float:
    config = getattr(env, "config", {})
    physics_dt = float(config.get("physics_world_step_size", 0.02))
    decision_repeat = int(config.get("decision_repeat", 5))
    dt_s = physics_dt * decision_repeat
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise JointCollectionError("simulator decision interval must be positive and finite")
    return dt_s


def collect_joint_episode(
    env: PlatoonEnv,
    *,
    max_steps: int,
    builder: JointBEVSampleBuilder | None = None,
) -> JointEpisodeRollout:
    """Collect one episode in memory; persistence is deliberately out of scope."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    env.reset()
    agent_ids = ("agent0", "agent1", "agent2")
    expert = RulePlannerExpert(env, agent_ids)
    sample_builder = builder or JointBEVSampleBuilder(agent_ids)
    sample_builder.reset()
    dt_s = simulator_decision_dt_s(env)
    samples: list[JointBEVSample] = []
    rejected_joint_steps = 0
    simulator_steps = 0
    failure_reason = None
    episode_terminated = False
    episode_truncated = False

    for joint_step in range(int(max_steps)):
        sample_builder.capture_state(env, timestamp_s=joint_step * dt_s)
        expert_step = expert.plan(env)
        if sample_builder.history_ready():
            try:
                samples.append(sample_builder.build_sample(env, expert_step))
            except JointStepRejected:
                # The expert controls remain valid, but a joint label/mask
                # conflict makes all three aligned training rows unusable.
                rejected_joint_steps += 1
        _, _, terminated, truncated, info = env.low_level_step(dict(expert_step.controls))
        simulator_steps += 1
        for agent_id in agent_ids:
            agent_info = info.get(agent_id, {}) if isinstance(info, Mapping) else {}
            if not isinstance(agent_info, Mapping):
                continue
            if any(
                bool(agent_info.get(key, False))
                for key in (
                    "crash",
                    "crash_vehicle",
                    "crash_object",
                    "crash_building",
                    "crash_human",
                )
            ):
                failure_reason = f"crash:{agent_id}"
                break
            if any(
                bool(agent_info.get(key, False))
                for key in ("out_of_road", "out_of_route")
            ):
                failure_reason = f"out_of_road:{agent_id}"
                break
        episode_terminated = bool(terminated.get("__all__", False))
        episode_truncated = bool(truncated.get("__all__", False))
        if failure_reason is not None or episode_terminated or episode_truncated:
            break
    return JointEpisodeRollout(
        samples=tuple(samples),
        simulator_steps=simulator_steps,
        rejected_joint_steps=rejected_joint_steps,
        failure_reason=failure_reason,
        terminated=episode_terminated,
        truncated=episode_truncated,
    )


__all__ = [
    "AgentRole",
    "EGO_STATE_DIM",
    "ExpertJointStep",
    "JointBEVSample",
    "JointBEVSampleBuilder",
    "JointCollectionError",
    "JointEpisodeRollout",
    "JointStepRejected",
    "JOINT_SAMPLE_DTYPES",
    "JOINT_SAMPLE_SHAPES",
    "NUM_PLATOON_AGENTS",
    "RulePlannerExpert",
    "SensorlessJointBEVPlatoonEnv",
    "collect_joint_episode",
    "simulator_decision_dt_s",
]
