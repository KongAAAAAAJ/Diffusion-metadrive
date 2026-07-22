"""Hard feasibility and expert-label contract for the BEV-only planner.

The feasibility mask deliberately excludes dynamic actors and expert decisions.
It answers only whether a fixed semantic mode is supported by road topology,
stays on the simulator drivable surface, and satisfies broad vehicle limits.
Dynamic collision risk belongs to the training objective and closed-loop reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np

from envs.observations.semantic_bev import (
    BEVChannel,
    SemanticBEVConfig,
    SemanticBEVRasterizer,
)


NUM_MODES = 10
TRAJECTORY_STEPS = 8
TRAJECTORY_DIM = 3
TRAJECTORY_SHAPE = (NUM_MODES, TRAJECTORY_STEPS, TRAJECTORY_DIM)


class ModeContractError(ValueError):
    """Raised when mode feasibility or expert labels violate the frozen contract."""


class ModeIndex(IntEnum):
    KEEP_HIGH = 0
    KEEP_MEDIUM = 1
    KEEP_LOW = 2
    LEFT_HIGH = 3
    LEFT_MEDIUM = 4
    LEFT_LOW = 5
    RIGHT_HIGH = 6
    RIGHT_MEDIUM = 7
    RIGHT_LOW = 8
    STOP = 9


MODE_NAMES = tuple(mode.name for mode in ModeIndex)
KEEP_MODES = tuple(int(mode) for mode in (ModeIndex.KEEP_HIGH, ModeIndex.KEEP_MEDIUM, ModeIndex.KEEP_LOW))
LEFT_MODES = tuple(int(mode) for mode in (ModeIndex.LEFT_HIGH, ModeIndex.LEFT_MEDIUM, ModeIndex.LEFT_LOW))
RIGHT_MODES = tuple(int(mode) for mode in (ModeIndex.RIGHT_HIGH, ModeIndex.RIGHT_MEDIUM, ModeIndex.RIGHT_LOW))


class RuleAction(IntEnum):
    LEFT = -1
    KEEP = 0
    RIGHT = 1


@dataclass(frozen=True)
class ModeTopology:
    """Route-aware lateral reachability derived from simulator topology."""

    left_reachable: bool
    right_reachable: bool

    def __post_init__(self) -> None:
        for name in ("left_reachable", "right_reachable"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ModeContractError(f"{name} must be bool, got {type(value).__name__}")
            object.__setattr__(self, name, bool(value))


@dataclass(frozen=True)
class HardModeMaskConfig:
    """Broad physical limits used only to reject impossible anchors."""

    dt_s: float = 0.5
    vehicle_length_m: float = 5.74
    vehicle_width_m: float = 2.3
    max_sweep_step_m: float = 0.5
    max_sweep_heading_step_rad: float = 0.05
    max_speed_mps: float = 100.0 / 3.6
    min_accel_mps2: float = -8.0
    max_accel_mps2: float = 5.0
    max_yaw_rate_rad_s: float = 1.0
    max_curvature_per_m: float = 0.25
    max_lateral_accel_mps2: float = 6.0
    max_heading_alignment_error_rad: float = 0.6
    min_forward_step_m: float = -1e-3
    movement_epsilon_m: float = 1e-3
    drivable_threshold: int = 1

    def __post_init__(self) -> None:
        positive = (
            "dt_s",
            "vehicle_length_m",
            "vehicle_width_m",
            "max_sweep_step_m",
            "max_sweep_heading_step_rad",
            "max_speed_mps",
            "max_yaw_rate_rad_s",
            "max_curvature_per_m",
            "max_lateral_accel_mps2",
            "max_heading_alignment_error_rad",
            "movement_epsilon_m",
        )
        for name in positive:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ModeContractError(f"{name} must be numeric") from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ModeContractError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        try:
            min_accel = float(self.min_accel_mps2)
            max_accel = float(self.max_accel_mps2)
            min_forward = float(self.min_forward_step_m)
        except (TypeError, ValueError) as exc:
            raise ModeContractError("acceleration and forward limits must be numeric") from exc
        if not np.isfinite(min_accel) or not np.isfinite(max_accel):
            raise ModeContractError("acceleration limits must be finite")
        if min_accel >= max_accel:
            raise ModeContractError("min_accel_mps2 must be smaller than max_accel_mps2")
        if not np.isfinite(min_forward) or min_forward > 0.0:
            raise ModeContractError("min_forward_step_m must be finite and non-positive")
        if isinstance(self.drivable_threshold, (bool, np.bool_)):
            raise ModeContractError("drivable_threshold must be an integer in [1, 255]")
        try:
            drivable_threshold = int(self.drivable_threshold)
        except (TypeError, ValueError) as exc:
            raise ModeContractError("drivable_threshold must be an integer in [1, 255]") from exc
        if drivable_threshold != self.drivable_threshold or not 1 <= drivable_threshold <= 255:
            raise ModeContractError("drivable_threshold must be an integer in [1, 255]")
        object.__setattr__(self, "min_accel_mps2", min_accel)
        object.__setattr__(self, "max_accel_mps2", max_accel)
        object.__setattr__(self, "min_forward_step_m", min_forward)
        object.__setattr__(self, "drivable_threshold", drivable_threshold)


def _frozen_bool_mask(value: np.ndarray, *, name: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (NUM_MODES,):
        raise ModeContractError(f"{name} must have shape [{NUM_MODES}], got {mask.shape}")
    result = np.ascontiguousarray(mask).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HardModeMaskResult:
    topology_mask: np.ndarray
    road_mask: np.ndarray
    kinematic_mask: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        for name in ("topology_mask", "road_mask", "kinematic_mask", "valid_mask"):
            object.__setattr__(self, name, _frozen_bool_mask(getattr(self, name), name=name))


def _validate_bev(bev: np.ndarray) -> np.ndarray:
    array = np.asarray(bev)
    expected = SemanticBEVConfig().shape
    if array.shape != expected:
        raise ModeContractError(f"bev must have shape {expected}, got {array.shape}")
    if array.dtype != np.uint8:
        raise ModeContractError(f"bev must have dtype uint8, got {array.dtype}")
    return np.ascontiguousarray(array)


def _validate_coarse_trajectories(coarse_trajectories: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(coarse_trajectories, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModeContractError("coarse_trajectories must be numeric") from exc
    if array.shape != TRAJECTORY_SHAPE:
        raise ModeContractError(
            f"coarse_trajectories must have shape {TRAJECTORY_SHAPE}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ModeContractError("coarse_trajectories contains non-finite values")
    return np.ascontiguousarray(array)


def _validate_expert_trajectory(expert_trajectory: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(expert_trajectory, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModeContractError("expert_trajectory must be numeric") from exc
    expected = (TRAJECTORY_STEPS, TRAJECTORY_DIM)
    if array.shape != expected:
        raise ModeContractError(f"expert_trajectory must have shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ModeContractError("expert_trajectory contains non-finite values")
    return np.ascontiguousarray(array)


def _validate_mode_mask(mode_valid_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mode_valid_mask)
    if array.shape != (NUM_MODES,):
        raise ModeContractError(
            f"mode_valid_mask must have shape [{NUM_MODES}], got {array.shape}"
        )
    if array.dtype != np.bool_:
        raise ModeContractError(f"mode_valid_mask must have dtype bool, got {array.dtype}")
    return np.ascontiguousarray(array)


def _wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _topology_mask(topology: ModeTopology) -> np.ndarray:
    mask = np.zeros((NUM_MODES,), dtype=bool)
    mask[list(KEEP_MODES)] = True
    mask[list(LEFT_MODES)] = topology.left_reachable
    mask[list(RIGHT_MODES)] = topology.right_reachable
    mask[ModeIndex.STOP] = True
    return mask


def _interpolated_poses(trajectory: np.ndarray, config: HardModeMaskConfig) -> np.ndarray:
    poses = np.concatenate(
        [np.zeros((1, TRAJECTORY_DIM), dtype=np.float64), trajectory], axis=0
    )
    samples = [poses[0]]
    for start, end in zip(poses[:-1], poses[1:]):
        distance = float(np.linalg.norm(end[:2] - start[:2]))
        heading_delta = float(_wrap_to_pi(end[2] - start[2]))
        spatial_steps = int(np.ceil(distance / config.max_sweep_step_m))
        angular_steps = int(np.ceil(abs(heading_delta) / config.max_sweep_heading_step_rad))
        num_steps = max(1, spatial_steps, angular_steps)
        for step in range(1, num_steps + 1):
            fraction = float(step) / float(num_steps)
            pose = np.empty((TRAJECTORY_DIM,), dtype=np.float64)
            pose[:2] = start[:2] + fraction * (end[:2] - start[:2])
            pose[2] = float(_wrap_to_pi(start[2] + fraction * heading_delta))
            samples.append(pose)
    return np.asarray(samples, dtype=np.float64)


def _footprint_corners(pose: np.ndarray, config: HardModeMaskConfig) -> np.ndarray:
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
    cos_h = float(np.cos(pose[2]))
    sin_h = float(np.sin(pose[2]))
    rotation = np.asarray([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    return offsets @ rotation.T + pose[None, :2]


def _trajectory_stays_drivable(
    trajectory: np.ndarray,
    drivable: np.ndarray,
    config: HardModeMaskConfig,
) -> bool:
    rasterizer = SemanticBEVRasterizer()
    swept_footprint = np.zeros_like(drivable, dtype=np.uint8)
    height, width = drivable.shape
    for pose in _interpolated_poses(trajectory, config):
        corners = _footprint_corners(pose, config)
        pixels = rasterizer.ego_to_pixel(corners.astype(np.float32))
        if (
            np.any(pixels[:, 0] < 0.0)
            or np.any(pixels[:, 0] > width - 1)
            or np.any(pixels[:, 1] < 0.0)
            or np.any(pixels[:, 1] > height - 1)
        ):
            return False
        polygon = np.rint(pixels).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(swept_footprint, [polygon], color=1, lineType=cv2.LINE_8)
    occupied = swept_footprint.astype(bool)
    return bool(np.any(occupied) and np.all(drivable[occupied]))


def _road_mask(bev: np.ndarray, trajectories: np.ndarray, config: HardModeMaskConfig) -> np.ndarray:
    drivable = bev[BEVChannel.DRIVABLE] >= int(config.drivable_threshold)
    return np.asarray(
        [
            _trajectory_stays_drivable(trajectory, drivable, config)
            for trajectory in trajectories
        ],
        dtype=bool,
    )


def _trajectory_is_kinematic(
    trajectory: np.ndarray,
    ego_speed_mps: float,
    config: HardModeMaskConfig,
) -> bool:
    poses = np.concatenate(
        [np.zeros((1, TRAJECTORY_DIM), dtype=np.float64), trajectory], axis=0
    )
    segments = np.diff(poses[:, :2], axis=0)
    distances = np.linalg.norm(segments, axis=1)
    heading_delta = _wrap_to_pi(np.diff(poses[:, 2]))
    previous_heading = poses[:-1, 2]
    forward = segments[:, 0] * np.cos(previous_heading) + segments[:, 1] * np.sin(previous_heading)
    if np.any(forward < config.min_forward_step_m):
        return False

    speeds = distances / config.dt_s
    if np.any(speeds > config.max_speed_mps + 1e-6):
        return False
    accelerations = np.diff(np.concatenate([[ego_speed_mps], speeds])) / config.dt_s
    if np.any(accelerations < config.min_accel_mps2 - 1e-6):
        return False
    if np.any(accelerations > config.max_accel_mps2 + 1e-6):
        return False

    yaw_rate = np.abs(heading_delta) / config.dt_s
    if np.any(yaw_rate > config.max_yaw_rate_rad_s + 1e-6):
        return False
    lateral_acceleration = speeds * yaw_rate
    if np.any(lateral_acceleration > config.max_lateral_accel_mps2 + 1e-6):
        return False

    moving = distances > config.movement_epsilon_m
    if np.any(np.abs(heading_delta[moving]) / distances[moving] > config.max_curvature_per_m + 1e-6):
        return False
    if np.any(moving):
        segment_heading = np.arctan2(segments[moving, 1], segments[moving, 0])
        middle_heading = previous_heading[moving] + 0.5 * heading_delta[moving]
        alignment_error = np.abs(_wrap_to_pi(segment_heading - middle_heading))
        if np.any(alignment_error > config.max_heading_alignment_error_rad + 1e-6):
            return False
    return True


def _kinematic_mask(
    trajectories: np.ndarray,
    ego_speed_mps: float,
    config: HardModeMaskConfig,
) -> np.ndarray:
    return np.asarray(
        [
            _trajectory_is_kinematic(trajectory, ego_speed_mps, config)
            for trajectory in trajectories
        ],
        dtype=bool,
    )


def build_hard_mode_valid_mask(
    bev: np.ndarray,
    coarse_trajectories: np.ndarray,
    ego_speed_mps: float,
    topology: ModeTopology,
    config: HardModeMaskConfig | None = None,
) -> HardModeMaskResult:
    """Build an expert-independent hard mask for the fixed ten modes."""

    bev_array = _validate_bev(bev)
    trajectories = _validate_coarse_trajectories(coarse_trajectories)
    if not isinstance(topology, ModeTopology):
        raise ModeContractError("topology must be a ModeTopology instance")
    try:
        speed = float(ego_speed_mps)
    except (TypeError, ValueError) as exc:
        raise ModeContractError("ego_speed_mps must be numeric") from exc
    if not np.isfinite(speed) or speed < 0.0:
        raise ModeContractError("ego_speed_mps must be finite and non-negative")
    cfg = config or HardModeMaskConfig()
    if not isinstance(cfg, HardModeMaskConfig):
        raise ModeContractError("config must be a HardModeMaskConfig instance")

    topology_mask = _topology_mask(topology)
    road_mask = _road_mask(bev_array, trajectories, cfg)
    kinematic_mask = _kinematic_mask(trajectories, speed, cfg)
    valid_mask = topology_mask & road_mask & kinematic_mask
    # STOP is the sole safety invariant. Invalid global tensors are rejected
    # above, but a stopped ego must retain an action even at a road boundary.
    valid_mask[ModeIndex.STOP] = True
    return HardModeMaskResult(
        topology_mask=topology_mask,
        road_mask=road_mask,
        kinematic_mask=kinematic_mask,
        valid_mask=valid_mask,
    )


def _candidate_modes_for_action(rule_action: RuleAction) -> tuple[int, ...]:
    if rule_action == RuleAction.LEFT:
        return LEFT_MODES
    if rule_action == RuleAction.KEEP:
        return KEEP_MODES + (int(ModeIndex.STOP),)
    if rule_action == RuleAction.RIGHT:
        return RIGHT_MODES
    raise ModeContractError(f"unsupported RuleMaker action: {int(rule_action)}")


def label_gt_mode(
    rule_action: int | RuleAction,
    expert_trajectory: np.ndarray,
    coarse_trajectories: np.ndarray,
    mode_valid_mask: np.ndarray,
    heading_error_weight: float = 0.2,
) -> int:
    """Choose a valid speed tier inside the RuleMaker-selected lateral group."""

    if isinstance(rule_action, (bool, np.bool_)) or not isinstance(
        rule_action, (int, np.integer, RuleAction)
    ):
        raise ModeContractError(f"RuleMaker action must be one of -1, 0, 1; got {rule_action!r}")
    try:
        action = RuleAction(int(rule_action))
    except (TypeError, ValueError) as exc:
        raise ModeContractError(f"RuleMaker action must be one of -1, 0, 1; got {rule_action!r}") from exc
    expert = _validate_expert_trajectory(expert_trajectory)
    coarse = _validate_coarse_trajectories(coarse_trajectories)
    valid_mask = _validate_mode_mask(mode_valid_mask)
    weight = float(heading_error_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ModeContractError("heading_error_weight must be finite and non-negative")

    candidates = tuple(index for index in _candidate_modes_for_action(action) if valid_mask[index])
    if not candidates:
        raise ModeContractError(
            f"RuleMaker action {action.name} has no hard-valid mode; the joint sample must be discarded"
        )

    scores = []
    for index in candidates:
        xy_error = np.linalg.norm(expert[:, :2] - coarse[index, :, :2], axis=-1).mean()
        heading_error = np.abs(_wrap_to_pi(expert[:, 2] - coarse[index, :, 2])).mean()
        scores.append(float(xy_error + weight * heading_error))
    selected = int(candidates[int(np.argmin(np.asarray(scores, dtype=np.float64)))])
    if not bool(valid_mask[selected]):
        raise AssertionError("GT mode selection escaped the hard-valid candidate set")
    return selected
