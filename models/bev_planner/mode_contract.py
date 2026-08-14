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


@dataclass(frozen=True)
class TrajectoryKinematicResult:
    """Immutable fixed-time trajectory feasibility result.

    The vectors contain one value for each future sample at
    ``0.5, 1.0, ..., 4.0`` seconds.  Shape/type failures are contract errors;
    physical failures are reported through ``violations``.
    """

    sample_times_s: np.ndarray
    segment_distance_m: np.ndarray
    forward_step_m: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    yaw_rate_rad_s: np.ndarray
    curvature_per_m: np.ndarray
    lateral_acceleration_mps2: np.ndarray
    heading_alignment_error_rad: np.ndarray
    cumulative_distance_m: np.ndarray
    reachable_min_distance_m: np.ndarray
    reachable_max_distance_m: np.ndarray
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "sample_times_s",
            "segment_distance_m",
            "forward_step_m",
            "speed_mps",
            "acceleration_mps2",
            "yaw_rate_rad_s",
            "curvature_per_m",
            "lateral_acceleration_mps2",
            "heading_alignment_error_rad",
            "cumulative_distance_m",
            "reachable_min_distance_m",
            "reachable_max_distance_m",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (TRAJECTORY_STEPS,) or not np.isfinite(value).all():
                raise ModeContractError(
                    f"{name} must be finite [{TRAJECTORY_STEPS}]"
                )
            frozen = np.ascontiguousarray(value)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if (
            not isinstance(self.violations, tuple)
            or len(set(self.violations)) != len(self.violations)
            or any(not isinstance(value, str) for value in self.violations)
        ):
            raise ModeContractError("violations must be a unique tuple of strings")

    @property
    def valid(self) -> bool:
        return not self.violations


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


def _reachable_distance(
    current_speed_mps: float,
    acceleration_mps2: float,
    config: HardModeMaskConfig,
) -> np.ndarray:
    """Integrate the constant-acceleration reachability envelope exactly.

    A clipped trapezoid over a whole 0.5 s interval overestimates the minimum
    distance when a low-speed vehicle stops before the interval ends.  Normal
    planner profiles use continuous stop-and-hold kinematics, so the envelope
    must use the same time semantics.
    """

    speed = float(current_speed_mps)
    distance = 0.0
    values = []
    for _ in range(TRAJECTORY_STEPS):
        dt = float(config.dt_s)
        acceleration = float(acceleration_mps2)
        if acceleration < 0.0 and speed + acceleration * dt < 0.0:
            active_dt = speed / -acceleration
            distance += speed * active_dt + 0.5 * acceleration * active_dt**2
            next_speed = 0.0
        elif (
            acceleration > 0.0
            and speed + acceleration * dt > config.max_speed_mps
        ):
            active_dt = (config.max_speed_mps - speed) / acceleration
            active_dt = float(np.clip(active_dt, 0.0, dt))
            distance += (
                speed * active_dt
                + 0.5 * acceleration * active_dt**2
                + config.max_speed_mps * (dt - active_dt)
            )
            next_speed = float(config.max_speed_mps)
        else:
            distance += speed * dt + 0.5 * acceleration * dt**2
            next_speed = float(
                np.clip(
                    speed + acceleration * dt,
                    0.0,
                    config.max_speed_mps,
                )
            )
        values.append(distance)
        speed = next_speed
    return np.asarray(values, dtype=np.float64)


def validate_trajectory_kinematics(
    trajectory: np.ndarray,
    current_speed_mps: float,
    origin_pose: np.ndarray,
    config: HardModeMaskConfig | None = None,
) -> TrajectoryKinematicResult:
    """Validate one fixed-time local or world-frame trajectory.

    ``origin_pose`` must use the same coordinate frame as ``trajectory``.
    This makes the calculation usable both for ego-local dataset tensors and
    world-frame Normal-planner candidates without changing the physics.
    """

    try:
        values = np.asarray(trajectory, dtype=np.float64)
        origin = np.asarray(origin_pose, dtype=np.float64)
        speed0 = float(current_speed_mps)
    except (TypeError, ValueError) as exc:
        raise ModeContractError("trajectory kinematic inputs must be numeric") from exc
    if values.shape != (TRAJECTORY_STEPS, TRAJECTORY_DIM):
        raise ModeContractError(
            f"trajectory must have shape [{TRAJECTORY_STEPS},{TRAJECTORY_DIM}]"
        )
    if origin.shape != (TRAJECTORY_DIM,):
        raise ModeContractError("origin_pose must have shape [3]")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(speed0)
        or speed0 < 0.0
    ):
        raise ModeContractError(
            "trajectory, origin_pose and current_speed_mps must be finite; "
            "speed must be non-negative"
        )
    cfg = config or HardModeMaskConfig()
    if not isinstance(cfg, HardModeMaskConfig):
        raise ModeContractError("config must be a HardModeMaskConfig instance")

    poses = np.concatenate(
        [origin[None], values], axis=0
    )
    segments = np.diff(poses[:, :2], axis=0)
    chord_distances = np.linalg.norm(segments, axis=1)
    heading_delta = _wrap_to_pi(np.diff(poses[:, 2]))
    # Fixed-time waypoints on a curved path delimit an arc, while their XY
    # difference is only its chord.  Using the chord as longitudinal travel
    # makes a physically reachable, hard-braking curve appear shorter than
    # the minimum reachable distance.  Recover the constant-curvature arc
    # implied by the endpoint tangents; the straight-line limit is exact.
    half_angle = 0.5 * np.abs(heading_delta)
    arc_scale = np.ones_like(chord_distances)
    curved = half_angle > 1.0e-8
    arc_scale[curved] = half_angle[curved] / np.sin(half_angle[curved])
    distances = chord_distances * arc_scale
    previous_heading = poses[:-1, 2]
    forward = (
        segments[:, 0] * np.cos(previous_heading)
        + segments[:, 1] * np.sin(previous_heading)
    )
    speeds = distances / cfg.dt_s
    accelerations = (
        np.diff(np.concatenate([[speed0], speeds])) / cfg.dt_s
    )
    yaw_rate = np.abs(heading_delta) / cfg.dt_s
    lateral_acceleration = speeds * yaw_rate
    curvature = np.zeros_like(distances)
    moving = distances > cfg.movement_epsilon_m
    curvature[moving] = np.abs(heading_delta[moving]) / distances[moving]
    alignment_error = np.zeros_like(distances)
    if np.any(moving):
        segment_heading = np.arctan2(segments[moving, 1], segments[moving, 0])
        middle_heading = previous_heading[moving] + 0.5 * heading_delta[moving]
        alignment_error[moving] = np.abs(
            _wrap_to_pi(segment_heading - middle_heading)
        )
    cumulative = np.cumsum(distances)
    reachable_min = _reachable_distance(speed0, cfg.min_accel_mps2, cfg)
    reachable_max = _reachable_distance(speed0, cfg.max_accel_mps2, cfg)
    epsilon = 1.0e-6
    violations = []
    checks = (
        (
            "first_waypoint_at_time_zero",
            (
                speed0 > 0.5
                and distances[0] <= cfg.movement_epsilon_m
                and reachable_min[0] > cfg.movement_epsilon_m
            ),
        ),
        ("non_forward_motion", np.any(forward < cfg.min_forward_step_m)),
        ("speed_limit", np.any(speeds > cfg.max_speed_mps + epsilon)),
        (
            "acceleration_below_min",
            np.any(accelerations < cfg.min_accel_mps2 - epsilon),
        ),
        (
            "acceleration_above_max",
            np.any(accelerations > cfg.max_accel_mps2 + epsilon),
        ),
        (
            "yaw_rate_limit",
            np.any(yaw_rate > cfg.max_yaw_rate_rad_s + epsilon),
        ),
        (
            "curvature_limit",
            np.any(curvature > cfg.max_curvature_per_m + epsilon),
        ),
        (
            "lateral_acceleration_limit",
            np.any(
                lateral_acceleration
                > cfg.max_lateral_accel_mps2 + epsilon
            ),
        ),
        (
            "heading_alignment",
            np.any(
                alignment_error
                > cfg.max_heading_alignment_error_rad + epsilon
            ),
        ),
        (
            "outside_reachable_distance",
            np.any(cumulative < reachable_min - 1.0e-3)
            or np.any(cumulative > reachable_max + 1.0e-3),
        ),
    )
    for name, failed in checks:
        if bool(failed):
            violations.append(name)
    return TrajectoryKinematicResult(
        sample_times_s=np.arange(1, TRAJECTORY_STEPS + 1, dtype=np.float64)
        * cfg.dt_s,
        segment_distance_m=distances,
        forward_step_m=forward,
        speed_mps=speeds,
        acceleration_mps2=accelerations,
        yaw_rate_rad_s=yaw_rate,
        curvature_per_m=curvature,
        lateral_acceleration_mps2=lateral_acceleration,
        heading_alignment_error_rad=alignment_error,
        cumulative_distance_m=cumulative,
        reachable_min_distance_m=reachable_min,
        reachable_max_distance_m=reachable_max,
        violations=tuple(violations),
    )


def _kinematic_mask(
    trajectories: np.ndarray,
    ego_speed_mps: float,
    config: HardModeMaskConfig,
) -> np.ndarray:
    return np.asarray(
        [
            validate_trajectory_kinematics(
                trajectory,
                ego_speed_mps,
                np.zeros((TRAJECTORY_DIM,), dtype=np.float64),
                config,
            ).valid
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


def mode_indices_for_rule_action(
    rule_action: int | RuleAction,
) -> tuple[int, ...]:
    """Return the frozen mode group represented by a RuleMaker action."""

    if isinstance(rule_action, (bool, np.bool_)) or not isinstance(
        rule_action, (int, np.integer, RuleAction)
    ):
        raise ModeContractError(
            f"RuleMaker action must be one of -1, 0, 1; got {rule_action!r}"
        )
    try:
        action = RuleAction(int(rule_action))
    except (TypeError, ValueError) as exc:
        raise ModeContractError(
            f"RuleMaker action must be one of -1, 0, 1; got {rule_action!r}"
        ) from exc
    if action == RuleAction.LEFT:
        return LEFT_MODES
    if action == RuleAction.KEEP:
        return KEEP_MODES + (int(ModeIndex.STOP),)
    if action == RuleAction.RIGHT:
        return RIGHT_MODES
    raise AssertionError("unreachable RuleMaker action")


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

    candidates = tuple(index for index in mode_indices_for_rule_action(action) if valid_mask[index])
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


def label_gt_mode_from_trajectory(
    expert_trajectory: np.ndarray,
    coarse_trajectories: np.ndarray,
    mode_valid_mask: np.ndarray,
    heading_error_weight: float = 0.2,
) -> int:
    """Quantize an expert trajectory against every hard-valid semantic mode.

    This is reserved for route-topology transitions whose execution action is
    KEEP even though the physical trajectory contains a lateral manoeuvre.
    """

    expert = _validate_expert_trajectory(expert_trajectory)
    coarse = _validate_coarse_trajectories(coarse_trajectories)
    valid_mask = _validate_mode_mask(mode_valid_mask)
    weight = float(heading_error_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ModeContractError("heading_error_weight must be finite and non-negative")

    candidates = tuple(int(index) for index in np.flatnonzero(valid_mask))
    if not candidates:
        raise ModeContractError(
            "expert trajectory has no hard-valid mode; the joint sample must be discarded"
        )
    scores = []
    for index in candidates:
        xy_error = np.linalg.norm(expert[:, :2] - coarse[index, :, :2], axis=-1).mean()
        heading_error = np.abs(_wrap_to_pi(expert[:, 2] - coarse[index, :, 2])).mean()
        scores.append(float(xy_error + weight * heading_error))
    selected = int(candidates[int(np.argmin(np.asarray(scores, dtype=np.float64)))])
    if not bool(valid_mask[selected]):
        raise AssertionError("trajectory GT mode selection escaped the hard-valid candidate set")
    return selected
