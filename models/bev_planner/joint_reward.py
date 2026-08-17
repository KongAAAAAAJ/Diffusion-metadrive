"""Joint trajectory rewards for the BEV-only platoon policy.

This module contains no expert labels or policy-gradient code.  It scores one
group of three synchronized trajectories from simulator ground truth and the
semantic drivable BEV used by the planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from envs.observations.semantic_bev import BEVChannel, SemanticBEVConfig
AGENT_IDS = ("agent0", "agent1", "agent2")
NUM_ROLES = 3
TRAJECTORY_SHAPE = (8, 3)


class JointRewardError(RuntimeError):
    """Raised when the strict joint reward contract is violated."""


@dataclass(frozen=True)
class JointRewardConfig:
    trajectory_dt_s: float = 0.5
    interpolation_dt_s: float = 0.1
    vehicle_length_m: float = 5.74
    vehicle_width_m: float = 2.3
    platoon_safe_gap_m: float = 7.0
    background_safe_gap_m: float = 5.0
    tracking_longitudinal_margin_m: float = 0.0
    tracking_lateral_margin_m: float = 0.0
    tracking_heading_margin_rad: float = 0.0
    progress_norm_m: float = 30.0
    formation_norm_m: float = 10.0
    local_progress_weight: float = 0.8
    local_formation_weight: float = 1.0
    local_clearance_weight: float = 0.8
    local_comfort_weight: float = 0.05
    team_formation_weight: float = 1.0
    team_safety_weight: float = 1.5
    team_progress_weight: float = 0.2
    local_mix: float = 0.45
    team_mix: float = 0.55
    unsafe_base_reward: float = -20.0

    def __post_init__(self) -> None:
        positive = (
            "trajectory_dt_s",
            "interpolation_dt_s",
            "vehicle_length_m",
            "vehicle_width_m",
            "platoon_safe_gap_m",
            "background_safe_gap_m",
            "progress_norm_m",
            "formation_norm_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise JointRewardError(f"{name} must be positive and finite")
        margin_limits = {
            "tracking_longitudinal_margin_m": 1.5,
            "tracking_lateral_margin_m": 1.0,
            "tracking_heading_margin_rad": 0.15,
        }
        for name, maximum in margin_limits.items():
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0 or value > maximum:
                raise JointRewardError(
                    f"{name} must be finite and within [0,{maximum}]"
                )
        if self.interpolation_dt_s > self.trajectory_dt_s:
            raise JointRewardError(
                "interpolation_dt_s cannot exceed trajectory_dt_s"
            )
        for name in (
            "local_progress_weight",
            "local_formation_weight",
            "local_clearance_weight",
            "local_comfort_weight",
            "team_formation_weight",
            "team_safety_weight",
            "team_progress_weight",
            "local_mix",
            "team_mix",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise JointRewardError(f"{name} must be non-negative and finite")
        if not math.isclose(self.local_mix + self.team_mix, 1.0):
            raise JointRewardError("local_mix and team_mix must sum to one")
        if not math.isfinite(self.unsafe_base_reward) or (
            self.unsafe_base_reward >= -5.0
        ):
            raise JointRewardError("unsafe_base_reward must be below -5")


@dataclass(frozen=True)
class JointRewardResult:
    rewards: np.ndarray
    unsafe: np.ndarray
    collision: np.ndarray
    out_of_drivable: np.ndarray
    clearance_violation: np.ndarray
    components: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        rewards = np.asarray(self.rewards)
        if rewards.ndim != 1 or rewards.dtype != np.float32:
            raise JointRewardError("rewards must be float32 [G]")
        if not np.isfinite(rewards).all():
            raise JointRewardError("rewards must be finite")
        group_size = rewards.shape[0]
        for name in (
            "unsafe",
            "collision",
            "out_of_drivable",
            "clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.bool_ or value.shape != (group_size,):
                raise JointRewardError(f"{name} must be bool [G]")
        for name, value in self.components.items():
            array = np.asarray(value)
            if array.shape != (group_size,) or not np.isfinite(array).all():
                raise JointRewardError(
                    f"reward component {name} must be finite [G]"
                )


@dataclass(frozen=True)
class RewardCalibrationResult:
    mean_spearman: float
    pairwise_agreement: float
    informative_groups: int
    pairwise_comparisons: int
    false_safe_count: int
    passed: bool


def _as_numpy_model_field(model_inputs: object, name: str) -> np.ndarray:
    if isinstance(model_inputs, Mapping):
        value = model_inputs.get(name)
    else:
        value = getattr(model_inputs, name, None)
    if value is None:
        raise JointRewardError(f"model_inputs is missing {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_trajectories(trajectories: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectories)
    if (
        value.ndim != 4
        or value.shape[1:] != (NUM_ROLES, *TRAJECTORY_SHAPE)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
    ):
        raise JointRewardError(
            "trajectories must be finite floating-point [G,3,8,3]"
        )
    if value.shape[0] <= 0:
        raise JointRewardError("trajectory group cannot be empty")
    return np.ascontiguousarray(value, dtype=np.float32)


def _dense_local_trajectories(
    trajectories: np.ndarray, config: JointRewardConfig
) -> tuple[np.ndarray, np.ndarray]:
    source_times = np.arange(1, 9, dtype=np.float64) * config.trajectory_dt_s
    target_times = np.arange(
        config.interpolation_dt_s,
        source_times[-1] + 0.5 * config.interpolation_dt_s,
        config.interpolation_dt_s,
        dtype=np.float64,
    )
    source_with_origin = np.concatenate(([0.0], source_times))
    dense = np.empty(
        (*trajectories.shape[:2], len(target_times), 3), dtype=np.float64
    )
    for group in range(trajectories.shape[0]):
        for role in range(NUM_ROLES):
            trajectory = trajectories[group, role].astype(np.float64)
            xy_with_origin = np.concatenate(
                (np.zeros((1, 2), dtype=np.float64), trajectory[:, :2]), axis=0
            )
            heading = np.unwrap(
                np.concatenate(([0.0], trajectory[:, 2]), axis=0)
            )
            dense[group, role, :, 0] = np.interp(
                target_times, source_with_origin, xy_with_origin[:, 0]
            )
            dense[group, role, :, 1] = np.interp(
                target_times, source_with_origin, xy_with_origin[:, 1]
            )
            dense[group, role, :, 2] = np.arctan2(
                np.sin(np.interp(target_times, source_with_origin, heading)),
                np.cos(np.interp(target_times, source_with_origin, heading)),
            )
    return dense, target_times


def _agent_pose(env: object, agent_id: str) -> np.ndarray:
    vehicle = (getattr(env, "agents", {}) or {}).get(agent_id)
    if vehicle is None:
        raise JointRewardError(f"simulator is missing active {agent_id}")
    position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float64)
    heading = float(getattr(vehicle, "heading_theta", float("nan")))
    if position.size < 2 or not np.isfinite(position[:2]).all() or not math.isfinite(
        heading
    ):
        raise JointRewardError(f"simulator pose for {agent_id} is invalid")
    return np.asarray([position[0], position[1], heading], dtype=np.float64)


def _local_to_world(local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    cos_h = math.cos(float(pose[2]))
    sin_h = math.sin(float(pose[2]))
    world = np.empty_like(local, dtype=np.float64)
    world[..., 0] = pose[0] + cos_h * local[..., 0] - sin_h * local[..., 1]
    world[..., 1] = pose[1] + sin_h * local[..., 0] + cos_h * local[..., 1]
    world[..., 2] = np.arctan2(
        np.sin(local[..., 2] + pose[2]), np.cos(local[..., 2] + pose[2])
    )
    return world


def _footprint_points(local: np.ndarray, config: JointRewardConfig) -> np.ndarray:
    half_length, half_width = _tracking_aware_half_extents(config)
    offsets = np.asarray(
        [
            [0.0, 0.0],
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
        ],
        dtype=np.float64,
    )
    heading = local[:, 2]
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    points = np.empty((len(local), len(offsets), 2), dtype=np.float64)
    for index, (longitudinal, lateral) in enumerate(offsets):
        points[:, index, 0] = (
            local[:, 0] + cos_h * longitudinal - sin_h * lateral
        )
        points[:, index, 1] = (
            local[:, 1] + sin_h * longitudinal + cos_h * lateral
        )
    return points


def _tracking_aware_half_extents(
    config: JointRewardConfig,
) -> tuple[float, float]:
    """Conservative vehicle extents under measured closed-loop tracking error."""
    heading = config.tracking_heading_margin_rad
    half_length = (
        0.5 * config.vehicle_length_m
        + config.tracking_longitudinal_margin_m
        + 0.5 * config.vehicle_width_m * math.sin(heading)
    )
    half_width = (
        0.5 * config.vehicle_width_m
        + config.tracking_lateral_margin_m
        + 0.5 * config.vehicle_length_m * math.sin(heading)
    )
    return half_length, half_width


def _footprint_outside_drivable(
    local: np.ndarray, drivable: np.ndarray, config: JointRewardConfig
) -> bool:
    bev_config = SemanticBEVConfig()
    points = _footprint_points(local, config).reshape(-1, 2)
    col = (
        (bev_config.y_max_m - points[:, 1])
        / (bev_config.y_max_m - bev_config.y_min_m)
        * (bev_config.width - 1)
    )
    row = (
        (bev_config.x_max_m - points[:, 0])
        / (bev_config.x_max_m - bev_config.x_min_m)
        * (bev_config.height - 1)
    )
    inside = (
        (row >= 0.0)
        & (row <= bev_config.height - 1)
        & (col >= 0.0)
        & (col <= bev_config.width - 1)
    )
    if not bool(inside.all()):
        return True
    row_index = np.rint(row).astype(np.int64)
    col_index = np.rint(col).astype(np.int64)
    return bool(np.any(drivable[row_index, col_index] == 0))


def _rank_average(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        stop = start + 1
        while stop < len(value) and value[order[stop]] == value[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    first_rank = _rank_average(first)
    second_rank = _rank_average(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator <= 1e-12:
        return None
    return float(np.dot(first_centered, second_centered) / denominator)


def calibrate_joint_rewards(
    proxy_rewards: np.ndarray,
    simulator_rewards: np.ndarray,
    proxy_unsafe: np.ndarray,
    simulator_unsafe: np.ndarray,
    *,
    min_informative_groups: int = 12,
    min_mean_spearman: float = 0.50,
    min_pairwise_agreement: float = 0.70,
) -> RewardCalibrationResult:
    proxy = np.asarray(proxy_rewards, dtype=np.float64)
    simulator = np.asarray(simulator_rewards, dtype=np.float64)
    proxy_bad = np.asarray(proxy_unsafe)
    simulator_bad = np.asarray(simulator_unsafe)
    if (
        proxy.ndim != 2
        or proxy.shape[1] != 4
        or simulator.shape != proxy.shape
        or proxy_bad.shape != proxy.shape
        or simulator_bad.shape != proxy.shape
        or proxy_bad.dtype != np.bool_
        or simulator_bad.dtype != np.bool_
        or not np.isfinite(proxy).all()
        or not np.isfinite(simulator).all()
    ):
        raise JointRewardError(
            "calibration arrays must be finite [N,4] with bool unsafe masks"
        )
    correlations = []
    agreement = 0
    comparisons = 0
    for proxy_group, simulator_group in zip(proxy, simulator):
        correlation = _spearman(proxy_group, simulator_group)
        if correlation is not None:
            correlations.append(correlation)
        for first in range(4):
            for second in range(first + 1, 4):
                proxy_delta = proxy_group[first] - proxy_group[second]
                simulator_delta = (
                    simulator_group[first] - simulator_group[second]
                )
                if abs(proxy_delta) <= 1e-12 or abs(simulator_delta) <= 1e-12:
                    continue
                comparisons += 1
                agreement += int(np.sign(proxy_delta) == np.sign(simulator_delta))
    mean_spearman = (
        float(np.mean(correlations)) if correlations else float("-inf")
    )
    pairwise = agreement / comparisons if comparisons else 0.0
    false_safe = int(np.count_nonzero(simulator_bad & ~proxy_bad))
    passed = (
        len(correlations) >= int(min_informative_groups)
        and mean_spearman >= float(min_mean_spearman)
        and pairwise >= float(min_pairwise_agreement)
        and false_safe == 0
    )
    return RewardCalibrationResult(
        mean_spearman=mean_spearman,
        pairwise_agreement=float(pairwise),
        informative_groups=len(correlations),
        pairwise_comparisons=int(comparisons),
        false_safe_count=false_safe,
        passed=bool(passed),
    )


def compose_joint_reward(
    *,
    progress: np.ndarray,
    formation: np.ndarray,
    clearance: np.ndarray,
    comfort: np.ndarray,
    collision: np.ndarray,
    out_of_drivable: np.ndarray,
    clearance_violation: np.ndarray,
    config: JointRewardConfig,
    diagnostic_components: Mapping[str, np.ndarray] | None = None,
) -> JointRewardResult:
    """Apply the shared local/team formula and the non-negotiable safety gate."""

    arrays = {
        "progress": np.asarray(progress, dtype=np.float64),
        "formation": np.asarray(formation, dtype=np.float64),
        "clearance": np.asarray(clearance, dtype=np.float64),
        "comfort": np.asarray(comfort, dtype=np.float64),
    }
    group_size = arrays["progress"].shape[0] if arrays["progress"].ndim == 1 else -1
    if group_size <= 0 or any(
        value.shape != (group_size,) or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise JointRewardError("reward components must be finite vectors [G]")
    collision_values = np.asarray(collision)
    out_values = np.asarray(out_of_drivable)
    clearance_values = np.asarray(clearance_violation)
    if (
        collision_values.dtype != np.bool_
        or out_values.dtype != np.bool_
        or clearance_values.dtype != np.bool_
        or collision_values.shape != (group_size,)
        or out_values.shape != (group_size,)
        or clearance_values.shape != (group_size,)
    ):
        raise JointRewardError("reward safety masks must be bool [G]")

    local_reward = (
        config.local_progress_weight * arrays["progress"]
        + config.local_formation_weight * arrays["formation"]
        + config.local_clearance_weight * arrays["clearance"]
        + config.local_comfort_weight * arrays["comfort"]
    )
    team_reward = (
        config.team_formation_weight * arrays["formation"]
        + config.team_safety_weight * arrays["clearance"]
        + config.team_progress_weight * arrays["progress"]
    )
    safe_reward = np.clip(
        config.local_mix * local_reward + config.team_mix * team_reward,
        -5.0,
        5.0,
    )
    unsafe = collision_values | out_values | clearance_values
    severity = (
        collision_values.astype(np.float64)
        + out_values.astype(np.float64)
        + clearance_values.astype(np.float64)
    )
    rewards = np.where(
        unsafe,
        config.unsafe_base_reward - severity,
        safe_reward,
    ).astype(np.float32)
    diagnostics = {
        str(name): np.asarray(value, dtype=np.float32)
        for name, value in (diagnostic_components or {}).items()
    }
    return JointRewardResult(
        rewards=rewards,
        unsafe=unsafe.astype(np.bool_),
        collision=collision_values,
        out_of_drivable=out_values,
        clearance_violation=clearance_values,
        components={
            **{name: value.astype(np.float32) for name, value in arrays.items()},
            "local": local_reward.astype(np.float32),
            "team": team_reward.astype(np.float32),
            **diagnostics,
        },
    )


class JointTrajectoryProxyReward:
    """Score synchronized three-role trajectories from simulator GT."""

    def __init__(self, config: JointRewardConfig | None = None) -> None:
        # Imported lazily because the normal planner consumes the BEV mode
        # contract while this reward is re-exported by models.bev_planner.
        from models.platoon_planner.platoon_normal_planner import (
            PlatoonNormalPlanner,
        )

        self.config = config or JointRewardConfig()
        self._prediction_planner = PlatoonNormalPlanner(
            background_safe_gap_m=self.config.background_safe_gap_m,
            platoon_safe_gap_m=self.config.platoon_safe_gap_m,
        )

    def score(
        self,
        env: object,
        model_inputs: object,
        trajectories: np.ndarray,
    ) -> JointRewardResult:
        from models.platoon_planner.platoon_normal_planner import (
            minimum_dense_background_gap,
            minimum_dense_pair_gap,
        )

        trajectory_values = _validate_trajectories(trajectories)
        bev = _as_numpy_model_field(model_inputs, "bev")
        relation = _as_numpy_model_field(
            model_inputs, "formation_relation_state"
        )
        if (
            bev.shape != (3, 8, 256, 256)
            or bev.dtype != np.uint8
            or relation.shape != (3, 12)
            or not np.issubdtype(relation.dtype, np.floating)
            or not np.isfinite(relation).all()
        ):
            raise JointRewardError("model input BEV/relation contract mismatch")

        dense_local, times = _dense_local_trajectories(
            trajectory_values, self.config
        )
        poses = [_agent_pose(env, agent_id) for agent_id in AGENT_IDS]
        dense_world = np.empty_like(dense_local)
        for role in range(NUM_ROLES):
            dense_world[:, role] = _local_to_world(dense_local[:, role], poses[role])

        group_size = trajectory_values.shape[0]
        progress = np.zeros(group_size, dtype=np.float64)
        formation = np.zeros(group_size, dtype=np.float64)
        clearance = np.zeros(group_size, dtype=np.float64)
        comfort = np.zeros(group_size, dtype=np.float64)
        collision = np.zeros(group_size, dtype=np.bool_)
        out_of_drivable = np.zeros(group_size, dtype=np.bool_)
        clearance_violation = np.zeros(group_size, dtype=np.bool_)
        minimum_background_by_group = np.full(
            group_size, np.inf, dtype=np.float64
        )
        minimum_platoon_by_group = np.full(
            group_size, np.inf, dtype=np.float64
        )

        background = self._prediction_planner._predicted_obstacles(
            env,
            (getattr(env, "agents", {}) or {})[AGENT_IDS[0]],
            times,
            include_platoon=False,
            include_policy_branches=True,
        )
        half_length, half_width = _tracking_aware_half_extents(self.config)
        dimensions = (2.0 * half_length, 2.0 * half_width)

        for group in range(group_size):
            role_progress = []
            role_clearance_deficits = []
            role_comfort = []
            for role in range(NUM_ROLES):
                local = dense_local[group, role]
                world = dense_world[group, role]
                role_progress.append(
                    float(
                        np.clip(
                            trajectory_values[group, role, -1, 0]
                            / self.config.progress_norm_m,
                            0.0,
                            1.0,
                        )
                    )
                )
                out_of_drivable[group] |= _footprint_outside_drivable(
                    local,
                    bev[role, int(BEVChannel.DRIVABLE)],
                    self.config,
                )
                speed = np.linalg.norm(
                    np.diff(
                        np.concatenate(
                            (np.zeros((1, 2)), local[:, :2]), axis=0
                        ),
                        axis=0,
                    ),
                    axis=1,
                ) / self.config.interpolation_dt_s
                acceleration = np.diff(speed, prepend=speed[0]) / (
                    self.config.interpolation_dt_s
                )
                unwrapped_heading = np.unwrap(local[:, 2])
                yaw_rate = np.diff(
                    unwrapped_heading, prepend=unwrapped_heading[0]
                ) / self.config.interpolation_dt_s
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

                for _, predicted, other_dimensions in background:
                    if self._prediction_planner._obb_overlap_series(
                        world,
                        dimensions,
                        predicted,
                        other_dimensions,
                        0.0,
                    ):
                        collision[group] = True
                minimum_background_gap = minimum_dense_background_gap(
                    world,
                    dimensions,
                    background,
                )
                background_deficit = (
                    0.0
                    if not math.isfinite(minimum_background_gap)
                    else max(
                        0.0,
                        self.config.background_safe_gap_m
                        - minimum_background_gap,
                    )
                    / self.config.background_safe_gap_m
                )
                if (
                    math.isfinite(minimum_background_gap)
                    and minimum_background_gap
                    < self.config.background_safe_gap_m
                ):
                    clearance_violation[group] = True
                minimum_background_by_group[group] = min(
                    minimum_background_by_group[group], minimum_background_gap
                )
                role_clearance_deficits.append(min(background_deficit, 1.0))

            pair_errors = []
            platoon_deficits = []
            for leader, follower in ((0, 1), (1, 2), (0, 2)):
                first = dense_world[group, leader]
                second = dense_world[group, follower]
                if self._prediction_planner._obb_overlap_series(
                    first, dimensions, second, dimensions, 0.0
                ):
                    collision[group] = True
                minimum_pair_gap = minimum_dense_pair_gap(
                    first,
                    dimensions,
                    second,
                    dimensions,
                )
                center_distance = np.linalg.norm(
                    first[:, :2] - second[:, :2], axis=1
                )
                minimum_platoon_by_group[group] = min(
                    minimum_platoon_by_group[group], minimum_pair_gap
                )
                if minimum_pair_gap < self.config.platoon_safe_gap_m:
                    clearance_violation[group] = True
                if follower == leader + 1:
                    deficit = np.maximum(
                        self.config.platoon_safe_gap_m - minimum_pair_gap,
                        0.0,
                    ) / self.config.platoon_safe_gap_m
                    platoon_deficits.append(float(np.clip(deficit, 0, 1)))

                    target_gap = abs(float(relation[follower, leader * 6 + 4]))
                    if target_gap <= 0.0:
                        target_gap = float(
                            getattr(env, "_desired_center_spacing_m")(
                                AGENT_IDS[follower], AGENT_IDS[leader]
                            )
                        )
                    pair_errors.append(
                        float(np.mean(np.abs(center_distance - target_gap)))
                    )
            progress[group] = float(np.mean(role_progress))
            formation[group] = -min(
                float(np.mean(pair_errors)) / self.config.formation_norm_m,
                1.0,
            )
            clearance[group] = -min(
                float(
                    np.mean(
                        [*role_clearance_deficits, *platoon_deficits]
                    )
                ),
                1.0,
            )
            comfort[group] = float(np.mean(role_comfort))

        minimum_background_by_group[
            ~np.isfinite(minimum_background_by_group)
        ] = 1.0e6
        minimum_platoon_by_group[
            ~np.isfinite(minimum_platoon_by_group)
        ] = 1.0e6
        return compose_joint_reward(
            progress=progress,
            formation=formation,
            clearance=clearance,
            comfort=comfort,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance_violation,
            config=self.config,
            diagnostic_components={
                "minimum_background_gap_m": minimum_background_by_group,
                "minimum_platoon_gap_m": minimum_platoon_by_group,
            },
        )


__all__ = [
    "JointRewardConfig",
    "JointRewardError",
    "JointRewardResult",
    "JointTrajectoryProxyReward",
    "RewardCalibrationResult",
    "calibrate_joint_rewards",
    "compose_joint_reward",
]
