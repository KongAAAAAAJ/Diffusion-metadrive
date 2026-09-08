"""Joint trajectory rewards for the BEV-only platoon policy.

This module contains no expert labels or policy-gradient code.  It scores one
group of three synchronized trajectories from simulator ground truth and the
semantic drivable BEV used by the planner.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from envs.observations.semantic_bev import BEVChannel, SemanticBEVConfig
from models.platoon_planner.collision_geometry import (
    obb_overlap_series,
    shared_corridor_gap_series,
)

AGENT_IDS = ("agent0", "agent1", "agent2")
NUM_ROLES = 3
NUM_MODES = 10
TRAJECTORY_SHAPE = (8, 3)

JOINT_REWARD_CONTRACT = {
    "version": "stage2_joint_reward_v2",
    "formula": (
        "+progress_weight*progress_score"
        "-formation_weight*formation_penalty"
        "-gap_weight*gap_penalty"
        "-ttc_weight*ttc_penalty"
        "-road_weight*road_penalty"
        "-comfort_weight*comfort_penalty"
        "-collision_penalty*collision"
        "-out_of_drivable_penalty*out_of_drivable"
    ),
    "quality_component_range": [0.0, 1.0],
    "total_reward_clip": None,
    "unsafe_reward_branch": False,
    "continuous_risk_aggregation": "temporal_max_weight*max+temporal_mean_weight*mean",
    "continuous_risk_timeline": "0.0s through 4.0s inclusive at 0.1s",
    "gap_geometry": (
        "shared-corridor oriented-box bumper gap with relative-heading support"
    ),
    "background_branch_semantics": (
        "max per time over mutually-exclusive branches of one actor, then max "
        "per time over actors and all other interactions"
    ),
    "collision_semantics": (
        "physical policy footprint against physical platoon footprints and "
        "nominal background predictions only"
    ),
    "road_semantics": (
        "full raster-resolution tracking footprint for signed-margin risk; "
        "full raster-resolution physical footprint for out-of-drivable event"
    ),
    "diagnostic_unsafe_semantics": (
        "collision or out_of_drivable or clearance_violation; never changes reward"
    ),
}
JOINT_REWARD_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        JOINT_REWARD_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

VEHICLE_MODE_REWARD_CONTRACT = {
    "version": "stage2_vehicle_mode_reward_v1",
    "comparison_unit": "vehicle_mode",
    "candidate_shape": "[3,10,N,8,3]",
    "same_mode_pretrain_shape": "[3,10,8,3]",
    "teammate_context": "frozen Stage 1 argmax raw tau_d",
    "counterfactual": (
        "replace only the target vehicle trajectory; keep both teammates fixed"
    ),
    "reward_scope": (
        "target progress, comfort, road and out-of-drivable plus target-to-"
        "background and target-to-teammate gap, TTC, collision and clearance"
    ),
    "formation_component": False,
    "other_vehicle_self_events": False,
    "formula": (
        "+progress_weight*progress_score"
        "-gap_weight*gap_penalty"
        "-ttc_weight*ttc_penalty"
        "-road_weight*road_penalty"
        "-comfort_weight*comfort_penalty"
        "-collision_penalty*collision"
        "-out_of_drivable_penalty*out_of_drivable"
    ),
    "invalid_mode_values": "zero-filled and excluded by valid_mode_mask",
}
VEHICLE_MODE_REWARD_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        VEHICLE_MODE_REWARD_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

GRPO_OPEN_REWARD_APPLICATION_CONTRACT = {
    "version": "stage2_grpo_open_application_v5",
    "comparison_unit": "vehicle_mode",
    "policy_sample_domain": "all-mode raw tau_d",
    "policy_probability_domain": "per-vehicle per-mode DDIM trajectory transitions",
    "mode_policy_terms": False,
    "reward_input_domain": "tau_d",
    "teammate_reward_context": "frozen Stage 1 argmax raw tau_d",
    "same_mode_reference": "frozen Stage 1 raw tau_d for the target mode",
    "advantage_normalization": (
        "per-(vehicle,mode) mean-centered population-RMS standardization"
    ),
    "active_mode_condition": (
        "hard-valid, optimizer-executable and mean(candidate_reward) >= "
        "same-mode pretrain reward"
    ),
    "activation_margin": None,
    "activation_reward_span_threshold": None,
    "inactive_mode_loss_terms": "excluded from trajectory PG only",
    "anchor_scope": (
        "trajectory BC and trajectory KL cover every hard-valid "
        "optimizer-executable vehicle-mode"
    ),
    "sampled_candidate_execution": False,
    "sampled_candidate_optimizer_or_rule_maker": False,
    "live_state_retry": (
        "resample diffusion noise only when every vehicle-mode is inactive"
    ),
    "environment_execution_policy": (
        "always cached deterministic frozen Stage 1 argmax baseline"
    ),
    "execution_input_domain": "tau_cmd",
    "execution_transform": (
        "raw frozen baseline -> KinematicTrajectoryOptimizer -> "
        "RuleMaker finalize/safe-stop -> env.step"
    ),
    "baseline_execution_per_live_state": 1,
    "tracking_expansion_enabled": False,
    "calibration_required": False,
    "best_checkpoint_metric": (
        "validation/safety_constrained_simulator_reward_gain_trailing3"
    ),
    "joint_reward_role": "historical/final team evaluation diagnostic only",
}
GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


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
    gap_softness_m: float = 0.5
    ttc_warning_s: float = 4.0
    ttc_softness_s: float = 0.5
    closing_speed_epsilon_mps: float = 0.1
    road_margin_warning_m: float = 1.0
    road_margin_softness_m: float = 0.25
    tracking_longitudinal_margin_m: float = 0.0
    tracking_lateral_margin_m: float = 0.0
    tracking_heading_margin_rad: float = 0.0
    progress_norm_m: float = 30.0
    formation_norm_m: float = 10.0
    progress_weight: float = 0.47
    formation_weight: float = 1.0
    gap_weight: float = 1.185
    ttc_weight: float = 0.5
    road_weight: float = 0.5
    comfort_weight: float = 0.0225
    collision_penalty: float = 5.0
    out_of_drivable_penalty: float = 4.0
    temporal_max_weight: float = 0.7
    temporal_mean_weight: float = 0.3
    no_risk_gap_m: float = 1.0e6
    no_risk_ttc_s: float = 1.0e6

    def __post_init__(self) -> None:
        positive = (
            "trajectory_dt_s",
            "interpolation_dt_s",
            "vehicle_length_m",
            "vehicle_width_m",
            "platoon_safe_gap_m",
            "background_safe_gap_m",
            "gap_softness_m",
            "ttc_warning_s",
            "ttc_softness_s",
            "closing_speed_epsilon_mps",
            "road_margin_warning_m",
            "road_margin_softness_m",
            "progress_norm_m",
            "formation_norm_m",
            "no_risk_gap_m",
            "no_risk_ttc_s",
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
            "progress_weight",
            "formation_weight",
            "gap_weight",
            "ttc_weight",
            "road_weight",
            "comfort_weight",
            "collision_penalty",
            "out_of_drivable_penalty",
            "temporal_max_weight",
            "temporal_mean_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise JointRewardError(f"{name} must be non-negative and finite")
        if not math.isclose(
            self.temporal_max_weight + self.temporal_mean_weight,
            1.0,
        ):
            raise JointRewardError(
                "temporal_max_weight and temporal_mean_weight must sum to one"
            )


@dataclass(frozen=True)
class VehicleModeRewardConfig:
    """Configuration for target-vehicle counterfactual trajectory rewards."""

    trajectories_per_mode: int = 48
    trajectory_dt_s: float = 0.5
    interpolation_dt_s: float = 0.1
    vehicle_length_m: float = 5.74
    vehicle_width_m: float = 2.3
    platoon_safe_gap_m: float = 7.0
    background_safe_gap_m: float = 5.0
    gap_softness_m: float = 0.5
    ttc_warning_s: float = 4.0
    ttc_softness_s: float = 0.5
    closing_speed_epsilon_mps: float = 0.1
    road_margin_warning_m: float = 1.0
    road_margin_softness_m: float = 0.25
    tracking_longitudinal_margin_m: float = 0.0
    tracking_lateral_margin_m: float = 0.0
    tracking_heading_margin_rad: float = 0.0
    progress_norm_m: float = 30.0
    progress_weight: float = 0.47
    gap_weight: float = 1.185
    ttc_weight: float = 0.5
    road_weight: float = 0.5
    comfort_weight: float = 0.0225
    collision_penalty: float = 5.0
    out_of_drivable_penalty: float = 4.0
    temporal_max_weight: float = 0.7
    temporal_mean_weight: float = 0.3
    no_risk_gap_m: float = 1.0e6
    no_risk_ttc_s: float = 1.0e6

    def __post_init__(self) -> None:
        if (
            isinstance(self.trajectories_per_mode, bool)
            or not isinstance(self.trajectories_per_mode, int)
            or self.trajectories_per_mode <= 0
        ):
            raise JointRewardError("trajectories_per_mode must be a positive integer")
        # Reuse the established geometry/weight validation without exposing a
        # formation term in this contract.
        JointRewardConfig(
            trajectory_dt_s=self.trajectory_dt_s,
            interpolation_dt_s=self.interpolation_dt_s,
            vehicle_length_m=self.vehicle_length_m,
            vehicle_width_m=self.vehicle_width_m,
            platoon_safe_gap_m=self.platoon_safe_gap_m,
            background_safe_gap_m=self.background_safe_gap_m,
            gap_softness_m=self.gap_softness_m,
            ttc_warning_s=self.ttc_warning_s,
            ttc_softness_s=self.ttc_softness_s,
            closing_speed_epsilon_mps=self.closing_speed_epsilon_mps,
            road_margin_warning_m=self.road_margin_warning_m,
            road_margin_softness_m=self.road_margin_softness_m,
            tracking_longitudinal_margin_m=self.tracking_longitudinal_margin_m,
            tracking_lateral_margin_m=self.tracking_lateral_margin_m,
            tracking_heading_margin_rad=self.tracking_heading_margin_rad,
            progress_norm_m=self.progress_norm_m,
            formation_norm_m=1.0,
            progress_weight=self.progress_weight,
            formation_weight=0.0,
            gap_weight=self.gap_weight,
            ttc_weight=self.ttc_weight,
            road_weight=self.road_weight,
            comfort_weight=self.comfort_weight,
            collision_penalty=self.collision_penalty,
            out_of_drivable_penalty=self.out_of_drivable_penalty,
            temporal_max_weight=self.temporal_max_weight,
            temporal_mean_weight=self.temporal_mean_weight,
            no_risk_gap_m=self.no_risk_gap_m,
            no_risk_ttc_s=self.no_risk_ttc_s,
        )


def joint_reward_config_sha256(config: JointRewardConfig) -> str:
    """Return the canonical digest binding every numeric reward setting."""

    if not isinstance(config, JointRewardConfig):
        raise JointRewardError("config must be JointRewardConfig")
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def vehicle_mode_reward_config_sha256(config: VehicleModeRewardConfig) -> str:
    """Return the canonical digest for the counterfactual reward settings."""

    if not isinstance(config, VehicleModeRewardConfig):
        raise JointRewardError("config must be VehicleModeRewardConfig")
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
class VehicleModePretrainRewardResult:
    """Frozen same-mode target rewards cached once for one live state."""

    rewards: np.ndarray
    valid_mode_mask: np.ndarray
    unsafe: np.ndarray
    collision: np.ndarray
    out_of_drivable: np.ndarray
    clearance_violation: np.ndarray
    components: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        expected = (NUM_ROLES, NUM_MODES)
        rewards = np.asarray(self.rewards)
        if (
            rewards.shape != expected
            or rewards.dtype != np.float32
            or not np.isfinite(rewards).all()
        ):
            raise JointRewardError(
                "pretrain rewards must be finite float32 [3,10]"
            )
        valid = np.asarray(self.valid_mode_mask)
        if valid.shape != expected or valid.dtype != np.bool_:
            raise JointRewardError("valid_mode_mask must be bool [3,10]")
        for name in (
            "unsafe",
            "collision",
            "out_of_drivable",
            "clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != expected or value.dtype != np.bool_:
                raise JointRewardError(f"pretrain {name} must be bool [3,10]")
        for name, value in self.components.items():
            array = np.asarray(value)
            if array.shape != expected or not np.isfinite(array).all():
                raise JointRewardError(
                    f"pretrain reward component {name} must be finite [3,10]"
                )


@dataclass(frozen=True)
class RewardGeometryContext:
    """Immutable geometry shared by every reward pass for one live state.

    The context is intentionally process-local: it is valid only while the
    simulator state, BEV observation, and frozen teammate trajectories remain
    the state used to build it.  It avoids rebuilding background predictions
    and signed-distance fields for same-state pretrain/current/frozen scoring.
    """

    bev: np.ndarray
    poses: tuple[np.ndarray, ...]
    backgrounds: tuple[
        Mapping[str, list[tuple[str, np.ndarray, tuple[float, float]]]], ...
    ]
    road_fields: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class VehicleModeRewardResult:
    """Current and frozen same-mode rewards under identical teammate context."""

    rewards: np.ndarray
    pretrain_rewards: np.ndarray
    valid_mode_mask: np.ndarray
    unsafe: np.ndarray
    collision: np.ndarray
    out_of_drivable: np.ndarray
    clearance_violation: np.ndarray
    pretrain_unsafe: np.ndarray
    pretrain_collision: np.ndarray
    pretrain_out_of_drivable: np.ndarray
    pretrain_clearance_violation: np.ndarray
    components: Mapping[str, np.ndarray]
    pretrain_components: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        rewards = np.asarray(self.rewards)
        expected_prefix = (NUM_ROLES, NUM_MODES)
        if (
            rewards.ndim != 3
            or rewards.shape[:2] != expected_prefix
            or rewards.shape[2] <= 0
            or rewards.dtype != np.float32
            or not np.isfinite(rewards).all()
        ):
            raise JointRewardError("rewards must be finite float32 [3,10,N]")
        pretrain = np.asarray(self.pretrain_rewards)
        if (
            pretrain.shape != expected_prefix
            or pretrain.dtype != np.float32
            or not np.isfinite(pretrain).all()
        ):
            raise JointRewardError(
                "pretrain_rewards must be finite float32 [3,10]"
            )
        valid = np.asarray(self.valid_mode_mask)
        if valid.shape != expected_prefix or valid.dtype != np.bool_:
            raise JointRewardError("valid_mode_mask must be bool [3,10]")
        current_shape = rewards.shape
        for name in (
            "unsafe",
            "collision",
            "out_of_drivable",
            "clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != current_shape or value.dtype != np.bool_:
                raise JointRewardError(f"{name} must be bool [3,10,N]")
        for name in (
            "pretrain_unsafe",
            "pretrain_collision",
            "pretrain_out_of_drivable",
            "pretrain_clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != expected_prefix or value.dtype != np.bool_:
                raise JointRewardError(f"{name} must be bool [3,10]")
        for name, value in self.components.items():
            array = np.asarray(value)
            if array.shape != current_shape or not np.isfinite(array).all():
                raise JointRewardError(
                    f"vehicle reward component {name} must be finite [3,10,N]"
                )
        for name, value in self.pretrain_components.items():
            array = np.asarray(value)
            if array.shape != expected_prefix or not np.isfinite(array).all():
                raise JointRewardError(
                    f"pretrain reward component {name} must be finite [3,10]"
                )


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
        0.0,
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


def tracking_aware_half_extents(
    config: JointRewardConfig,
) -> tuple[float, float]:
    """Return conservative half extents under measured tracking error."""

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


def _footprint_points(
    local: np.ndarray,
    config: JointRewardConfig,
    *,
    tracking_aware: bool,
) -> np.ndarray:
    poses = np.asarray(local, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise JointRewardError("footprint poses must be finite [N,3]")
    if tracking_aware:
        half_length, half_width = tracking_aware_half_extents(config)
    else:
        half_length = 0.5 * config.vehicle_length_m
        half_width = 0.5 * config.vehicle_width_m
    bev_config = SemanticBEVConfig()
    longitudinal_resolution = (
        bev_config.x_max_m - bev_config.x_min_m
    ) / (bev_config.height - 1)
    lateral_resolution = (
        bev_config.y_max_m - bev_config.y_min_m
    ) / (bev_config.width - 1)
    longitudinal = np.linspace(
        -half_length,
        half_length,
        max(2, int(math.ceil(2.0 * half_length / longitudinal_resolution)) + 1),
        dtype=np.float64,
    )
    lateral = np.linspace(
        -half_width,
        half_width,
        max(2, int(math.ceil(2.0 * half_width / lateral_resolution)) + 1),
        dtype=np.float64,
    )
    longitudinal_grid, lateral_grid = np.meshgrid(
        longitudinal, lateral, indexing="ij"
    )
    offsets = np.column_stack(
        (longitudinal_grid.reshape(-1), lateral_grid.reshape(-1))
    )
    heading = poses[:, 2]
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    points = np.empty((len(local), len(offsets), 2), dtype=np.float64)
    for index, (longitudinal, lateral) in enumerate(offsets):
        points[:, index, 0] = (
            poses[:, 0] + cos_h * longitudinal - sin_h * lateral
        )
        points[:, index, 1] = (
            poses[:, 1] + sin_h * longitudinal + cos_h * lateral
        )
    return points


def _local_points_to_bev_coordinates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bev_config = SemanticBEVConfig()
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
    return row, col


def footprint_outside_drivable_series(
    local_poses: np.ndarray,
    drivable: np.ndarray,
    config: JointRewardConfig,
) -> np.ndarray:
    """Return physical-footprint raster violations for every dense sample."""

    drivable_values = np.asarray(drivable)
    bev_config = SemanticBEVConfig()
    if drivable_values.shape != (bev_config.height, bev_config.width):
        raise JointRewardError("drivable BEV must match SemanticBEVConfig")
    poses = np.asarray(local_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise JointRewardError("footprint poses must be finite [N,3]")
    half_length = 0.5 * config.vehicle_length_m
    half_width = 0.5 * config.vehicle_width_m
    offsets = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    result = np.zeros(len(poses), dtype=np.bool_)
    for index, pose in enumerate(poses):
        cosine = math.cos(float(pose[2]))
        sine = math.sin(float(pose[2]))
        corners = np.empty((4, 2), dtype=np.float64)
        corners[:, 0] = (
            pose[0] + cosine * offsets[:, 0] - sine * offsets[:, 1]
        )
        corners[:, 1] = (
            pose[1] + sine * offsets[:, 0] + cosine * offsets[:, 1]
        )
        row, col = _local_points_to_bev_coordinates(corners)
        if bool(
            np.any(row < 0.0)
            or np.any(row > bev_config.height - 1)
            or np.any(col < 0.0)
            or np.any(col > bev_config.width - 1)
        ):
            result[index] = True
            continue
        polygon = np.column_stack((np.rint(col), np.rint(row))).astype(
            np.int32
        )
        col_min = int(np.min(polygon[:, 0]))
        col_max = int(np.max(polygon[:, 0]))
        row_min = int(np.min(polygon[:, 1]))
        row_max = int(np.max(polygon[:, 1]))
        roi_mask = np.zeros(
            (row_max - row_min + 1, col_max - col_min + 1),
            dtype=np.uint8,
        )
        roi_polygon = polygon - np.asarray([col_min, row_min], dtype=np.int32)
        cv2.fillConvexPoly(roi_mask, roi_polygon, 1)
        roi_drivable = drivable_values[
            row_min : row_max + 1, col_min : col_max + 1
        ]
        result[index] = bool(np.any(roi_drivable[roi_mask != 0] == 0))
    return result


def drivable_signed_distance_m(drivable: np.ndarray) -> np.ndarray:
    """Build a metric signed-distance field from a semantic drivable raster."""

    values = np.asarray(drivable)
    bev_config = SemanticBEVConfig()
    if values.shape != (bev_config.height, bev_config.width):
        raise JointRewardError("drivable BEV must match SemanticBEVConfig")
    mask = values != 0
    row_resolution = (bev_config.x_max_m - bev_config.x_min_m) / (
        bev_config.height - 1
    )
    col_resolution = (bev_config.y_max_m - bev_config.y_min_m) / (
        bev_config.width - 1
    )
    sampling = (row_resolution, col_resolution)
    inside = distance_transform_edt(
        np.pad(mask, 1, mode="constant", constant_values=False),
        sampling=sampling,
    )[1:-1, 1:-1]
    if bool(np.any(mask)):
        outside = distance_transform_edt(~mask, sampling=sampling)
    else:
        outside = np.full(
            mask.shape,
            math.hypot(
                bev_config.x_max_m - bev_config.x_min_m,
                bev_config.y_max_m - bev_config.y_min_m,
            ),
            dtype=np.float64,
        )
    return np.ascontiguousarray(inside - outside, dtype=np.float64)


def footprint_road_margin_series(
    local_poses: np.ndarray,
    signed_distance_m: np.ndarray,
    config: JointRewardConfig,
    *,
    tracking_aware: bool,
) -> np.ndarray:
    """Return the minimum signed drivable margin across footprint samples."""

    field = np.asarray(signed_distance_m, dtype=np.float64)
    bev_config = SemanticBEVConfig()
    if (
        field.shape != (bev_config.height, bev_config.width)
        or not np.isfinite(field).all()
    ):
        raise JointRewardError("signed drivable distance must be finite BEV [H,W]")
    points_by_time = _footprint_points(
        local_poses, config, tracking_aware=tracking_aware
    )
    points = points_by_time.reshape(-1, 2)
    row, col = _local_points_to_bev_coordinates(points)
    inside = (
        (row >= 0.0)
        & (row <= bev_config.height - 1)
        & (col >= 0.0)
        & (col <= bev_config.width - 1)
    )
    sampled = np.empty(len(points), dtype=np.float64)
    if np.any(inside):
        indices = np.flatnonzero(inside)
        row_inside = row[indices]
        col_inside = col[indices]
        row_low = np.floor(row_inside).astype(np.int64)
        col_low = np.floor(col_inside).astype(np.int64)
        row_high = np.minimum(row_low + 1, bev_config.height - 1)
        col_high = np.minimum(col_low + 1, bev_config.width - 1)
        row_fraction = row_inside - row_low
        col_fraction = col_inside - col_low
        sampled[indices] = (
            field[row_low, col_low]
            * (1.0 - row_fraction)
            * (1.0 - col_fraction)
            + field[row_high, col_low]
            * row_fraction
            * (1.0 - col_fraction)
            + field[row_low, col_high]
            * (1.0 - row_fraction)
            * col_fraction
            + field[row_high, col_high] * row_fraction * col_fraction
        )
    if np.any(~inside):
        indices = np.flatnonzero(~inside)
        clamped_row = np.clip(row[indices], 0.0, bev_config.height - 1)
        clamped_col = np.clip(col[indices], 0.0, bev_config.width - 1)
        row_resolution = (bev_config.x_max_m - bev_config.x_min_m) / (
            bev_config.height - 1
        )
        col_resolution = (bev_config.y_max_m - bev_config.y_min_m) / (
            bev_config.width - 1
        )
        outside_distance = np.hypot(
            (row[indices] - clamped_row) * row_resolution,
            (col[indices] - clamped_col) * col_resolution,
        )
        sampled[indices] = -outside_distance
    return np.min(sampled.reshape(points_by_time.shape[:2]), axis=1)


def soft_threshold_risk(
    values: np.ndarray,
    *,
    warning_threshold: float,
    softness: float,
) -> np.ndarray:
    """Map a larger-is-safer metric to a stable continuous risk in [0,1]."""

    metric = np.asarray(values, dtype=np.float64)
    threshold = float(warning_threshold)
    width = float(softness)
    if (
        not np.isfinite(metric).all()
        or not math.isfinite(threshold)
        or threshold <= 0.0
        or not math.isfinite(width)
        or width <= 0.0
    ):
        raise JointRewardError("soft risk inputs and parameters must be finite/positive")
    scaled = (threshold - metric) / width
    softplus = np.maximum(scaled, 0.0) + np.log1p(
        np.exp(-np.abs(scaled))
    )
    return np.clip((width / threshold) * softplus, 0.0, 1.0)


def closing_ttc_from_gap_series(
    gap_series: np.ndarray,
    *,
    dt_s: float,
    closing_speed_epsilon_mps: float,
    no_risk_gap_m: float,
    no_risk_ttc_s: float,
) -> np.ndarray:
    """Return finite TTC using adjacent finite shared-corridor gap samples."""

    gap = np.asarray(gap_series, dtype=np.float64)
    if gap.ndim != 1 or not np.isfinite(gap).all() or gap.size == 0:
        raise JointRewardError("gap_series must be a non-empty finite vector")
    dt = float(dt_s)
    epsilon = float(closing_speed_epsilon_mps)
    gap_sentinel = float(no_risk_gap_m)
    sentinel = float(no_risk_ttc_s)
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
        or not math.isfinite(gap_sentinel)
        or gap_sentinel <= 0.0
        or not math.isfinite(sentinel)
        or sentinel <= 0.0
    ):
        raise JointRewardError("TTC parameters must be positive and finite")
    ttc = np.full(gap.shape, sentinel, dtype=np.float64)
    closing_speed = (gap[:-1] - gap[1:]) / dt
    adjacent_shared_corridor = (
        (gap[:-1] < gap_sentinel) & (gap[1:] < gap_sentinel)
    )
    closing = adjacent_shared_corridor & (closing_speed > epsilon)
    indices = np.flatnonzero(closing) + 1
    if len(indices):
        ttc[indices] = np.maximum(gap[indices], 0.0) / closing_speed[closing]
    return ttc


def ttc_risk_from_gap_series(
    gap_series: np.ndarray,
    *,
    dt_s: float,
    warning_threshold_s: float,
    softness_s: float,
    closing_speed_epsilon_mps: float,
    no_risk_gap_m: float,
    no_risk_ttc_s: float,
) -> np.ndarray:
    ttc = closing_ttc_from_gap_series(
        gap_series,
        dt_s=dt_s,
        closing_speed_epsilon_mps=closing_speed_epsilon_mps,
        no_risk_gap_m=no_risk_gap_m,
        no_risk_ttc_s=no_risk_ttc_s,
    )
    return soft_threshold_risk(
        ttc,
        warning_threshold=warning_threshold_s,
        softness=softness_s,
    )


def aggregate_temporal_risk(
    per_time_risk: np.ndarray,
    *,
    max_weight: float,
    mean_weight: float,
) -> float:
    values = np.asarray(per_time_risk, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise JointRewardError("temporal risk must be a non-empty [0,1] vector")
    if (
        not math.isfinite(max_weight)
        or not math.isfinite(mean_weight)
        or max_weight < 0.0
        or mean_weight < 0.0
        or not math.isclose(max_weight + mean_weight, 1.0)
    ):
        raise JointRewardError("temporal aggregation weights must sum to one")
    return float(max_weight * np.max(values) + mean_weight * np.mean(values))


def compose_joint_reward(
    *,
    progress_score: np.ndarray,
    formation_penalty: np.ndarray,
    gap_penalty: np.ndarray,
    ttc_penalty: np.ndarray,
    road_penalty: np.ndarray,
    comfort_penalty: np.ndarray,
    collision: np.ndarray,
    out_of_drivable: np.ndarray,
    clearance_violation: np.ndarray,
    config: JointRewardConfig,
    diagnostic_components: Mapping[str, np.ndarray] | None = None,
) -> JointRewardResult:
    """Compose the V2 continuous quality reward without an unsafe branch."""

    arrays = {
        "progress_score": np.asarray(progress_score, dtype=np.float64),
        "formation_penalty": np.asarray(formation_penalty, dtype=np.float64),
        "gap_penalty": np.asarray(gap_penalty, dtype=np.float64),
        "ttc_penalty": np.asarray(ttc_penalty, dtype=np.float64),
        "road_penalty": np.asarray(road_penalty, dtype=np.float64),
        "comfort_penalty": np.asarray(comfort_penalty, dtype=np.float64),
    }
    group_size = (
        arrays["progress_score"].shape[0]
        if arrays["progress_score"].ndim == 1
        else -1
    )
    if group_size <= 0 or any(
        value.shape != (group_size,)
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
        or np.any(value > 1.0)
        for value in arrays.values()
    ):
        raise JointRewardError("reward quality components must be [0,1] vectors [G]")
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

    unsafe = collision_values | out_values | clearance_values
    rewards = (
        config.progress_weight * arrays["progress_score"]
        - config.formation_weight * arrays["formation_penalty"]
        - config.gap_weight * arrays["gap_penalty"]
        - config.ttc_weight * arrays["ttc_penalty"]
        - config.road_weight * arrays["road_penalty"]
        - config.comfort_weight * arrays["comfort_penalty"]
        - config.collision_penalty * collision_values.astype(np.float64)
        - config.out_of_drivable_penalty * out_values.astype(np.float64)
    )
    if not np.isfinite(rewards).all():
        raise JointRewardError("composed rewards must be finite")
    reserved = set(arrays)
    overlap = reserved.intersection(diagnostic_components or {})
    if overlap:
        raise JointRewardError(
            f"diagnostic components cannot replace reward components: {sorted(overlap)}"
        )
    diagnostics = {
        str(name): np.asarray(value, dtype=np.float32)
        for name, value in (diagnostic_components or {}).items()
    }
    return JointRewardResult(
        rewards=rewards.astype(np.float32),
        unsafe=unsafe.astype(np.bool_),
        collision=collision_values,
        out_of_drivable=out_values,
        clearance_violation=clearance_values,
        components={
            **{name: value.astype(np.float32) for name, value in arrays.items()},
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
        progress_score = np.zeros(group_size, dtype=np.float64)
        formation_penalty = np.zeros(group_size, dtype=np.float64)
        gap_penalty = np.zeros(group_size, dtype=np.float64)
        ttc_penalty = np.zeros(group_size, dtype=np.float64)
        road_penalty = np.zeros(group_size, dtype=np.float64)
        comfort_penalty = np.zeros(group_size, dtype=np.float64)
        collision = np.zeros(group_size, dtype=np.bool_)
        out_of_drivable = np.zeros(group_size, dtype=np.bool_)
        clearance_violation = np.zeros(group_size, dtype=np.bool_)
        minimum_background_by_group = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_platoon_by_group = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_road_by_group = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_ttc_by_group = np.full(
            group_size, self.config.no_risk_ttc_s, dtype=np.float64
        )

        background = self._prediction_planner._predicted_obstacles(
            env,
            (getattr(env, "agents", {}) or {})[AGENT_IDS[0]],
            times,
            include_platoon=False,
            include_policy_branches=True,
        )
        background_by_actor: dict[
            str, list[tuple[str, np.ndarray, tuple[float, float]]]
        ] = {}
        for name, predicted, other_dimensions in background:
            actor = str(name).split(":policy_branch_", 1)[0]
            background_by_actor.setdefault(actor, []).append(
                (str(name), np.asarray(predicted, dtype=np.float64), other_dimensions)
            )
        half_length, half_width = tracking_aware_half_extents(self.config)
        tracking_dimensions = (2.0 * half_length, 2.0 * half_width)
        physical_dimensions = (
            self.config.vehicle_length_m,
            self.config.vehicle_width_m,
        )
        road_fields = [
            drivable_signed_distance_m(bev[role, int(BEVChannel.DRIVABLE)])
            for role in range(NUM_ROLES)
        ]

        for group in range(group_size):
            role_progress = []
            role_comfort = []
            interaction_gap_risks: list[np.ndarray] = []
            interaction_ttc_risks: list[np.ndarray] = []
            role_road_risks: list[np.ndarray] = []
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
                # The simulator reports collision/out only after an executed
                # step.  Keep those hard events on t=0.1..4.0 while continuous
                # gap/TTC/road risks also include the shared t=0 snapshot.
                physical_out = footprint_outside_drivable_series(
                    local[1:],
                    bev[role, int(BEVChannel.DRIVABLE)],
                    self.config,
                )
                out_of_drivable[group] |= bool(np.any(physical_out))
                road_margin = footprint_road_margin_series(
                    local,
                    road_fields[role],
                    self.config,
                    tracking_aware=True,
                )
                minimum_road_by_group[group] = min(
                    minimum_road_by_group[group], float(np.min(road_margin))
                )
                role_road_risks.append(
                    soft_threshold_risk(
                        road_margin,
                        warning_threshold=self.config.road_margin_warning_m,
                        softness=self.config.road_margin_softness_m,
                    )
                )
                motion_local = local[1:]
                speed = np.linalg.norm(
                    np.diff(
                        np.concatenate(
                            (np.zeros((1, 2)), motion_local[:, :2]), axis=0
                        ),
                        axis=0,
                    ),
                    axis=1,
                ) / self.config.interpolation_dt_s
                acceleration = np.diff(speed, prepend=speed[0]) / (
                    self.config.interpolation_dt_s
                )
                unwrapped_heading = np.unwrap(motion_local[:, 2])
                yaw_rate = np.diff(
                    unwrapped_heading, prepend=unwrapped_heading[0]
                ) / self.config.interpolation_dt_s
                role_comfort.append(
                    float(
                        np.clip(
                            0.5 * np.mean(np.abs(acceleration)) / 8.0
                            + 0.5 * np.mean(np.abs(yaw_rate)),
                            0.0,
                            1.0,
                        )
                    )
                )

                for actor, predictions in background_by_actor.items():
                    branch_gap_risks = []
                    branch_ttc_risks = []
                    for name, predicted, other_dimensions in predictions:
                        gap_series = shared_corridor_gap_series(
                            world,
                            tracking_dimensions,
                            predicted,
                            other_dimensions,
                            no_risk_gap_m=self.config.no_risk_gap_m,
                        )
                        branch_gap_risks.append(
                            soft_threshold_risk(
                                gap_series,
                                warning_threshold=(
                                    self.config.background_safe_gap_m
                                ),
                                softness=self.config.gap_softness_m,
                            )
                        )
                        ttc = closing_ttc_from_gap_series(
                            gap_series,
                            dt_s=self.config.interpolation_dt_s,
                            closing_speed_epsilon_mps=(
                                self.config.closing_speed_epsilon_mps
                            ),
                            no_risk_gap_m=self.config.no_risk_gap_m,
                            no_risk_ttc_s=self.config.no_risk_ttc_s,
                        )
                        minimum_ttc_by_group[group] = min(
                            minimum_ttc_by_group[group], float(np.min(ttc))
                        )
                        branch_ttc_risks.append(
                            soft_threshold_risk(
                                ttc,
                                warning_threshold=self.config.ttc_warning_s,
                                softness=self.config.ttc_softness_s,
                            )
                        )
                        if name == actor:
                            minimum_background_by_group[group] = min(
                                minimum_background_by_group[group],
                                float(np.min(gap_series)),
                            )
                            if obb_overlap_series(
                                world[1:],
                                physical_dimensions,
                                predicted[1:],
                                other_dimensions,
                                0.0,
                            ):
                                collision[group] = True
                    interaction_gap_risks.append(
                        np.max(np.stack(branch_gap_risks), axis=0)
                    )
                    interaction_ttc_risks.append(
                        np.max(np.stack(branch_ttc_risks), axis=0)
                    )

            pair_errors = []
            for leader, follower in ((0, 1), (1, 2), (0, 2)):
                first = dense_world[group, leader]
                second = dense_world[group, follower]
                if obb_overlap_series(
                    first[1:],
                    physical_dimensions,
                    second[1:],
                    physical_dimensions,
                    0.0,
                ):
                    collision[group] = True
                pair_gap = shared_corridor_gap_series(
                    first,
                    tracking_dimensions,
                    second,
                    tracking_dimensions,
                    no_risk_gap_m=self.config.no_risk_gap_m,
                )
                minimum_pair_gap = float(np.min(pair_gap))
                center_distance = np.linalg.norm(
                    first[1:, :2] - second[1:, :2], axis=1
                )
                minimum_platoon_by_group[group] = min(
                    minimum_platoon_by_group[group], minimum_pair_gap
                )
                if minimum_pair_gap < self.config.platoon_safe_gap_m:
                    clearance_violation[group] = True
                interaction_gap_risks.append(
                    soft_threshold_risk(
                        pair_gap,
                        warning_threshold=self.config.platoon_safe_gap_m,
                        softness=self.config.gap_softness_m,
                    )
                )
                pair_ttc = closing_ttc_from_gap_series(
                    pair_gap,
                    dt_s=self.config.interpolation_dt_s,
                    closing_speed_epsilon_mps=(
                        self.config.closing_speed_epsilon_mps
                    ),
                    no_risk_gap_m=self.config.no_risk_gap_m,
                    no_risk_ttc_s=self.config.no_risk_ttc_s,
                )
                minimum_ttc_by_group[group] = min(
                    minimum_ttc_by_group[group], float(np.min(pair_ttc))
                )
                interaction_ttc_risks.append(
                    soft_threshold_risk(
                        pair_ttc,
                        warning_threshold=self.config.ttc_warning_s,
                        softness=self.config.ttc_softness_s,
                    )
                )
                if follower == leader + 1:
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
            progress_score[group] = float(np.mean(role_progress))
            formation_penalty[group] = min(
                float(np.mean(pair_errors)) / self.config.formation_norm_m,
                1.0,
            )
            comfort_penalty[group] = float(np.mean(role_comfort))
            gap_penalty[group] = aggregate_temporal_risk(
                np.max(np.stack(interaction_gap_risks), axis=0),
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )
            ttc_penalty[group] = aggregate_temporal_risk(
                np.max(np.stack(interaction_ttc_risks), axis=0),
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )
            road_penalty[group] = aggregate_temporal_risk(
                np.max(np.stack(role_road_risks), axis=0),
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )
            clearance_violation[group] |= bool(
                minimum_background_by_group[group]
                < self.config.background_safe_gap_m
                or minimum_platoon_by_group[group]
                < self.config.platoon_safe_gap_m
            )

        return compose_joint_reward(
            progress_score=progress_score,
            formation_penalty=formation_penalty,
            gap_penalty=gap_penalty,
            ttc_penalty=ttc_penalty,
            road_penalty=road_penalty,
            comfort_penalty=comfort_penalty,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance_violation,
            config=self.config,
            diagnostic_components={
                "minimum_background_gap_m": minimum_background_by_group,
                "minimum_platoon_gap_m": minimum_platoon_by_group,
                "minimum_road_margin_m": minimum_road_by_group,
                "minimum_ttc_s": minimum_ttc_by_group,
            },
        )


class VehicleModeCounterfactualReward:
    """Score each ``(vehicle, mode)`` against one frozen teammate context.

    Candidate and same-mode frozen rewards differ only in the target vehicle's
    raw ``tau_d``.  The two teammates always use the deterministic frozen
    Stage-1 argmax trajectories.  This scorer deliberately does not read
    formation relations and never evaluates teammate-only events.
    """

    _COMPONENT_NAMES = (
        "progress_score",
        "gap_penalty",
        "ttc_penalty",
        "road_penalty",
        "comfort_penalty",
        "minimum_background_gap_m",
        "minimum_teammate_gap_m",
        "minimum_road_margin_m",
        "minimum_ttc_s",
    )

    def __init__(self, config: VehicleModeRewardConfig | None = None) -> None:
        from models.platoon_planner.platoon_normal_planner import (
            PlatoonNormalPlanner,
        )

        self.config = config or VehicleModeRewardConfig()
        self._prediction_planner = PlatoonNormalPlanner(
            background_safe_gap_m=self.config.background_safe_gap_m,
            platoon_safe_gap_m=self.config.platoon_safe_gap_m,
        )

    @staticmethod
    def _array_without_optional_batch(
        value: object,
        *,
        unbatched_ndim: int,
        name: str,
    ) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == unbatched_ndim + 1:
            if array.shape[0] != 1:
                raise JointRewardError(f"{name} only supports an optional B=1 axis")
            array = array[0]
        if array.ndim != unbatched_ndim:
            raise JointRewardError(f"{name} has the wrong rank")
        return array

    def _candidate_values(
        self,
        candidates: object,
        *,
        trajectories_per_mode: int | None = None,
    ) -> np.ndarray:
        values = self._array_without_optional_batch(
            candidates, unbatched_ndim=5, name="candidates"
        )
        sample_count = (
            self.config.trajectories_per_mode
            if trajectories_per_mode is None
            else int(trajectories_per_mode)
        )
        expected = (
            NUM_ROLES,
            NUM_MODES,
            sample_count,
            *TRAJECTORY_SHAPE,
        )
        if (
            values.shape != expected
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise JointRewardError(
                "candidates must be finite floating-point [3,10,N,8,3]"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _frozen_all_values(self, trajectories: object) -> np.ndarray:
        values = self._array_without_optional_batch(
            trajectories,
            unbatched_ndim=4,
            name="frozen_all_mode_trajectories",
        )
        if (
            values.shape != (NUM_ROLES, NUM_MODES, *TRAJECTORY_SHAPE)
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise JointRewardError(
                "frozen_all_mode_trajectories must be finite floating-point "
                "[3,10,8,3]"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _frozen_argmax_values(self, trajectories: object) -> np.ndarray:
        values = self._array_without_optional_batch(
            trajectories,
            unbatched_ndim=3,
            name="frozen_argmax_joint_trajectories",
        )
        if (
            values.shape != (NUM_ROLES, *TRAJECTORY_SHAPE)
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise JointRewardError(
                "frozen_argmax_joint_trajectories must be finite floating-point "
                "[3,8,3]"
            )
        return np.ascontiguousarray(values, dtype=np.float32)

    def _valid_values(self, valid_mode_mask: object) -> np.ndarray:
        values = self._array_without_optional_batch(
            valid_mode_mask, unbatched_ndim=2, name="valid_mode_mask"
        )
        if values.shape != (NUM_ROLES, NUM_MODES) or values.dtype != np.bool_:
            raise JointRewardError("valid_mode_mask must be bool [3,10]")
        return np.ascontiguousarray(values, dtype=np.bool_)

    def _background_by_actor(
        self,
        env: object,
        target_role: int,
        times: np.ndarray,
    ) -> dict[str, list[tuple[str, np.ndarray, tuple[float, float]]]]:
        agents = getattr(env, "agents", {}) or {}
        background = self._prediction_planner._predicted_obstacles(
            env,
            agents[AGENT_IDS[target_role]],
            times,
            include_platoon=False,
            include_policy_branches=True,
        )
        grouped: dict[str, list[tuple[str, np.ndarray, tuple[float, float]]]] = {}
        for name, predicted, dimensions in background:
            actor = str(name).split(":policy_branch_", 1)[0]
            grouped.setdefault(actor, []).append(
                (str(name), np.asarray(predicted, dtype=np.float64), dimensions)
            )
        return grouped

    def _score_target_group(
        self,
        *,
        target_role: int,
        target_trajectories: np.ndarray,
        frozen_argmax_joint_trajectories: np.ndarray,
        poses: list[np.ndarray],
        drivable: np.ndarray,
        road_field: np.ndarray,
        background_by_actor: Mapping[
            str, list[tuple[str, np.ndarray, tuple[float, float]]]
        ],
    ) -> dict[str, object]:
        group_size = target_trajectories.shape[0]
        joint = np.broadcast_to(
            frozen_argmax_joint_trajectories,
            (group_size, NUM_ROLES, *TRAJECTORY_SHAPE),
        ).copy()
        joint[:, target_role] = target_trajectories
        dense_local, _ = _dense_local_trajectories(joint, self.config)
        dense_world = np.empty_like(dense_local)
        for role in range(NUM_ROLES):
            dense_world[:, role] = _local_to_world(dense_local[:, role], poses[role])

        progress_score = np.zeros(group_size, dtype=np.float64)
        gap_penalty = np.zeros(group_size, dtype=np.float64)
        ttc_penalty = np.zeros(group_size, dtype=np.float64)
        road_penalty = np.zeros(group_size, dtype=np.float64)
        comfort_penalty = np.zeros(group_size, dtype=np.float64)
        collision = np.zeros(group_size, dtype=np.bool_)
        out_of_drivable = np.zeros(group_size, dtype=np.bool_)
        clearance_violation = np.zeros(group_size, dtype=np.bool_)
        minimum_background_gap = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_teammate_gap = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_road_margin = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_ttc = np.full(
            group_size, self.config.no_risk_ttc_s, dtype=np.float64
        )
        half_length, half_width = tracking_aware_half_extents(self.config)
        tracking_dimensions = (2.0 * half_length, 2.0 * half_width)
        physical_dimensions = (
            self.config.vehicle_length_m,
            self.config.vehicle_width_m,
        )

        for group in range(group_size):
            local = dense_local[group, target_role]
            world = dense_world[group, target_role]
            progress_score[group] = float(
                np.clip(
                    target_trajectories[group, -1, 0]
                    / self.config.progress_norm_m,
                    0.0,
                    1.0,
                )
            )
            out_of_drivable[group] = bool(
                np.any(
                    footprint_outside_drivable_series(
                        local[1:], drivable, self.config
                    )
                )
            )
            road_margin = footprint_road_margin_series(
                local,
                road_field,
                self.config,
                tracking_aware=True,
            )
            minimum_road_margin[group] = float(np.min(road_margin))
            road_risk = soft_threshold_risk(
                road_margin,
                warning_threshold=self.config.road_margin_warning_m,
                softness=self.config.road_margin_softness_m,
            )
            road_penalty[group] = aggregate_temporal_risk(
                road_risk,
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )

            motion_local = local[1:]
            speed = np.linalg.norm(
                np.diff(
                    np.concatenate(
                        (np.zeros((1, 2)), motion_local[:, :2]), axis=0
                    ),
                    axis=0,
                ),
                axis=1,
            ) / self.config.interpolation_dt_s
            acceleration = np.diff(speed, prepend=speed[0]) / (
                self.config.interpolation_dt_s
            )
            unwrapped_heading = np.unwrap(motion_local[:, 2])
            yaw_rate = np.diff(
                unwrapped_heading, prepend=unwrapped_heading[0]
            ) / self.config.interpolation_dt_s
            comfort_penalty[group] = float(
                np.clip(
                    0.5 * np.mean(np.abs(acceleration)) / 8.0
                    + 0.5 * np.mean(np.abs(yaw_rate)),
                    0.0,
                    1.0,
                )
            )

            interaction_gap_risks: list[np.ndarray] = []
            interaction_ttc_risks: list[np.ndarray] = []
            for actor, predictions in background_by_actor.items():
                branch_gap_risks = []
                branch_ttc_risks = []
                for name, predicted, other_dimensions in predictions:
                    gap_series = shared_corridor_gap_series(
                        world,
                        tracking_dimensions,
                        predicted,
                        other_dimensions,
                        no_risk_gap_m=self.config.no_risk_gap_m,
                    )
                    branch_gap_risks.append(
                        soft_threshold_risk(
                            gap_series,
                            warning_threshold=self.config.background_safe_gap_m,
                            softness=self.config.gap_softness_m,
                        )
                    )
                    ttc = closing_ttc_from_gap_series(
                        gap_series,
                        dt_s=self.config.interpolation_dt_s,
                        closing_speed_epsilon_mps=(
                            self.config.closing_speed_epsilon_mps
                        ),
                        no_risk_gap_m=self.config.no_risk_gap_m,
                        no_risk_ttc_s=self.config.no_risk_ttc_s,
                    )
                    minimum_ttc[group] = min(
                        minimum_ttc[group], float(np.min(ttc))
                    )
                    branch_ttc_risks.append(
                        soft_threshold_risk(
                            ttc,
                            warning_threshold=self.config.ttc_warning_s,
                            softness=self.config.ttc_softness_s,
                        )
                    )
                    if name == actor:
                        minimum_background_gap[group] = min(
                            minimum_background_gap[group],
                            float(np.min(gap_series)),
                        )
                        if obb_overlap_series(
                            world[1:],
                            physical_dimensions,
                            predicted[1:],
                            other_dimensions,
                            0.0,
                        ):
                            collision[group] = True
                interaction_gap_risks.append(
                    np.max(np.stack(branch_gap_risks), axis=0)
                )
                interaction_ttc_risks.append(
                    np.max(np.stack(branch_ttc_risks), axis=0)
                )

            for other_role in range(NUM_ROLES):
                if other_role == target_role:
                    continue
                teammate = dense_world[group, other_role]
                if obb_overlap_series(
                    world[1:],
                    physical_dimensions,
                    teammate[1:],
                    physical_dimensions,
                    0.0,
                ):
                    collision[group] = True
                pair_gap = shared_corridor_gap_series(
                    world,
                    tracking_dimensions,
                    teammate,
                    tracking_dimensions,
                    no_risk_gap_m=self.config.no_risk_gap_m,
                )
                minimum_teammate_gap[group] = min(
                    minimum_teammate_gap[group], float(np.min(pair_gap))
                )
                interaction_gap_risks.append(
                    soft_threshold_risk(
                        pair_gap,
                        warning_threshold=self.config.platoon_safe_gap_m,
                        softness=self.config.gap_softness_m,
                    )
                )
                pair_ttc = closing_ttc_from_gap_series(
                    pair_gap,
                    dt_s=self.config.interpolation_dt_s,
                    closing_speed_epsilon_mps=(
                        self.config.closing_speed_epsilon_mps
                    ),
                    no_risk_gap_m=self.config.no_risk_gap_m,
                    no_risk_ttc_s=self.config.no_risk_ttc_s,
                )
                minimum_ttc[group] = min(
                    minimum_ttc[group], float(np.min(pair_ttc))
                )
                interaction_ttc_risks.append(
                    soft_threshold_risk(
                        pair_ttc,
                        warning_threshold=self.config.ttc_warning_s,
                        softness=self.config.ttc_softness_s,
                    )
                )

            gap_penalty[group] = aggregate_temporal_risk(
                np.max(np.stack(interaction_gap_risks), axis=0),
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )
            ttc_penalty[group] = aggregate_temporal_risk(
                np.max(np.stack(interaction_ttc_risks), axis=0),
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )
            clearance_violation[group] = bool(
                minimum_background_gap[group] < self.config.background_safe_gap_m
                or minimum_teammate_gap[group] < self.config.platoon_safe_gap_m
            )

        rewards = (
            self.config.progress_weight * progress_score
            - self.config.gap_weight * gap_penalty
            - self.config.ttc_weight * ttc_penalty
            - self.config.road_weight * road_penalty
            - self.config.comfort_weight * comfort_penalty
            - self.config.collision_penalty * collision.astype(np.float64)
            - self.config.out_of_drivable_penalty
            * out_of_drivable.astype(np.float64)
        )
        unsafe = collision | out_of_drivable | clearance_violation
        return {
            "rewards": rewards.astype(np.float32),
            "unsafe": unsafe,
            "collision": collision,
            "out_of_drivable": out_of_drivable,
            "clearance_violation": clearance_violation,
            "components": {
                "progress_score": progress_score.astype(np.float32),
                "gap_penalty": gap_penalty.astype(np.float32),
                "ttc_penalty": ttc_penalty.astype(np.float32),
                "road_penalty": road_penalty.astype(np.float32),
                "comfort_penalty": comfort_penalty.astype(np.float32),
                "minimum_background_gap_m": minimum_background_gap.astype(
                    np.float32
                ),
                "minimum_teammate_gap_m": minimum_teammate_gap.astype(np.float32),
                "minimum_road_margin_m": minimum_road_margin.astype(np.float32),
                "minimum_ttc_s": minimum_ttc.astype(np.float32),
            },
        }

    def build_geometry_context(
        self,
        env: object,
        model_inputs: object,
        frozen_argmax: np.ndarray,
    ) -> RewardGeometryContext:
        """Build reusable reward geometry for one unchanged live state."""

        bev = _as_numpy_model_field(model_inputs, "bev")
        if bev.ndim == 5:
            if bev.shape[0] != 1:
                raise JointRewardError("model input BEV only supports B=1")
            bev = bev[0]
        if bev.shape != (NUM_ROLES, 8, 256, 256) or bev.dtype != np.uint8:
            raise JointRewardError("model input BEV contract mismatch")
        poses = tuple(_agent_pose(env, agent_id) for agent_id in AGENT_IDS)
        probe = frozen_argmax[None]
        _, times = _dense_local_trajectories(probe, self.config)
        backgrounds = tuple(
            self._background_by_actor(env, role, times)
            for role in range(NUM_ROLES)
        )
        road_fields = tuple(
            drivable_signed_distance_m(bev[role, int(BEVChannel.DRIVABLE)])
            for role in range(NUM_ROLES)
        )
        return RewardGeometryContext(
            bev=bev,
            poses=poses,
            backgrounds=backgrounds,
            road_fields=road_fields,
        )

    def score_pretrain(
        self,
        env: object,
        model_inputs: object,
        frozen_all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        *,
        geometry_context: RewardGeometryContext | None = None,
    ) -> VehicleModePretrainRewardResult:
        """Score all valid frozen modes once for one live-state cache."""

        frozen_all = self._frozen_all_values(frozen_all_mode_trajectories)
        frozen_argmax = self._frozen_argmax_values(
            frozen_argmax_joint_trajectories
        )
        valid = self._valid_values(valid_mode_mask)
        shape = (NUM_ROLES, NUM_MODES)
        rewards = np.zeros(shape, dtype=np.float32)
        unsafe = np.zeros(shape, dtype=np.bool_)
        collision = np.zeros(shape, dtype=np.bool_)
        out_of_drivable = np.zeros(shape, dtype=np.bool_)
        clearance = np.zeros(shape, dtype=np.bool_)
        components = {
            name: np.zeros(shape, dtype=np.float32)
            for name in self._COMPONENT_NAMES
        }
        context = (
            geometry_context
            if geometry_context is not None
            else self.build_geometry_context(env, model_inputs, frozen_argmax)
        )

        for role in range(NUM_ROLES):
            for mode in range(NUM_MODES):
                if not valid[role, mode]:
                    continue
                scored = self._score_target_group(
                    target_role=role,
                    target_trajectories=frozen_all[role, mode][None],
                    frozen_argmax_joint_trajectories=frozen_argmax,
                    poses=list(context.poses),
                    drivable=context.bev[role, int(BEVChannel.DRIVABLE)],
                    road_field=context.road_fields[role],
                    background_by_actor=context.backgrounds[role],
                )
                rewards[role, mode] = scored["rewards"][0]
                unsafe[role, mode] = scored["unsafe"][0]
                collision[role, mode] = scored["collision"][0]
                out_of_drivable[role, mode] = scored["out_of_drivable"][0]
                clearance[role, mode] = scored["clearance_violation"][0]
                scored_components = scored["components"]
                for name in self._COMPONENT_NAMES:
                    components[name][role, mode] = scored_components[name][0]

        return VehicleModePretrainRewardResult(
            rewards=rewards,
            valid_mode_mask=valid,
            unsafe=unsafe,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance,
            components=components,
        )

    def score_candidates(
        self,
        env: object,
        model_inputs: object,
        candidates: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        pretrain: VehicleModePretrainRewardResult,
        *,
        geometry_context: RewardGeometryContext | None = None,
        trajectories_per_mode: int | None = None,
    ) -> VehicleModeRewardResult:
        """Score one noise resample while reusing cached pretrain rewards."""

        candidate_values = self._candidate_values(
            candidates, trajectories_per_mode=trajectories_per_mode
        )
        frozen_argmax = self._frozen_argmax_values(
            frozen_argmax_joint_trajectories
        )
        valid = self._valid_values(valid_mode_mask)
        if not isinstance(pretrain, VehicleModePretrainRewardResult):
            raise JointRewardError(
                "pretrain must be VehicleModePretrainRewardResult"
            )
        if not np.array_equal(pretrain.valid_mode_mask, valid):
            raise JointRewardError("pretrain and candidate valid_mode_mask differ")
        current_shape = (
            NUM_ROLES,
            NUM_MODES,
            candidate_values.shape[2],
        )
        rewards = np.zeros(current_shape, dtype=np.float32)
        unsafe = np.zeros(current_shape, dtype=np.bool_)
        collision = np.zeros(current_shape, dtype=np.bool_)
        out_of_drivable = np.zeros(current_shape, dtype=np.bool_)
        clearance = np.zeros(current_shape, dtype=np.bool_)
        components = {
            name: np.zeros(current_shape, dtype=np.float32)
            for name in self._COMPONENT_NAMES
        }
        context = (
            geometry_context
            if geometry_context is not None
            else self.build_geometry_context(env, model_inputs, frozen_argmax)
        )

        for role in range(NUM_ROLES):
            for mode in range(NUM_MODES):
                if not valid[role, mode]:
                    continue
                scored = self._score_target_group(
                    target_role=role,
                    target_trajectories=candidate_values[role, mode],
                    frozen_argmax_joint_trajectories=frozen_argmax,
                    poses=list(context.poses),
                    drivable=context.bev[role, int(BEVChannel.DRIVABLE)],
                    road_field=context.road_fields[role],
                    background_by_actor=context.backgrounds[role],
                )
                rewards[role, mode] = scored["rewards"]
                unsafe[role, mode] = scored["unsafe"]
                collision[role, mode] = scored["collision"]
                out_of_drivable[role, mode] = scored["out_of_drivable"]
                clearance[role, mode] = scored["clearance_violation"]
                scored_components = scored["components"]
                for name in self._COMPONENT_NAMES:
                    components[name][role, mode] = scored_components[name]

        return VehicleModeRewardResult(
            rewards=rewards,
            pretrain_rewards=pretrain.rewards,
            valid_mode_mask=valid,
            unsafe=unsafe,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance,
            pretrain_unsafe=pretrain.unsafe,
            pretrain_collision=pretrain.collision,
            pretrain_out_of_drivable=pretrain.out_of_drivable,
            pretrain_clearance_violation=pretrain.clearance_violation,
            components=components,
            pretrain_components=pretrain.components,
        )

    def score_all_mode_trajectories(
        self,
        env: object,
        model_inputs: object,
        all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        pretrain: VehicleModePretrainRewardResult,
        *,
        geometry_context: RewardGeometryContext | None = None,
    ) -> VehicleModeRewardResult:
        """Score one deterministic trajectory for every vehicle-mode pair.

        This is the exact N=1 counterpart of :meth:`score_candidates` for
        validation.  It avoids materializing N identical trajectory copies.
        """

        all_mode_values = self._frozen_all_values(all_mode_trajectories)
        return self.score_candidates(
            env,
            model_inputs,
            all_mode_values[:, :, None],
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
            pretrain,
            geometry_context=geometry_context,
            trajectories_per_mode=1,
        )

    def score_counterfactuals(
        self,
        env: object,
        model_inputs: object,
        candidates: object,
        frozen_all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
    ) -> VehicleModeRewardResult:
        """One-shot scoring; online retries should use the split cache API."""

        pretrain = self.score_pretrain(
            env,
            model_inputs,
            frozen_all_mode_trajectories,
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
        )
        return self.score_candidates(
            env,
            model_inputs,
            candidates,
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
            pretrain,
        )


__all__ = [
    "GRPO_OPEN_REWARD_APPLICATION_CONTRACT",
    "GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256",
    "JOINT_REWARD_CONTRACT",
    "JOINT_REWARD_CONTRACT_SHA256",
    "VEHICLE_MODE_REWARD_CONTRACT",
    "VEHICLE_MODE_REWARD_CONTRACT_SHA256",
    "JointRewardConfig",
    "JointRewardError",
    "JointRewardResult",
    "JointTrajectoryProxyReward",
    "RewardGeometryContext",
    "VehicleModeCounterfactualReward",
    "VehicleModePretrainRewardResult",
    "VehicleModeRewardConfig",
    "VehicleModeRewardResult",
    "aggregate_temporal_risk",
    "closing_ttc_from_gap_series",
    "compose_joint_reward",
    "drivable_signed_distance_m",
    "footprint_outside_drivable_series",
    "footprint_road_margin_series",
    "joint_reward_config_sha256",
    "shared_corridor_gap_series",
    "soft_threshold_risk",
    "tracking_aware_half_extents",
    "ttc_risk_from_gap_series",
    "vehicle_mode_reward_config_sha256",
]
