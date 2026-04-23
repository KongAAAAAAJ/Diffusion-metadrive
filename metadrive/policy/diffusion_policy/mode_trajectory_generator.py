from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np

from metadrive.policy.diffusion_policy.mode_context import DynamicObstacle, ModeContext
from metadrive.policy.diffusion_policy.mode_feasibility import (
    GeometricFeasibilityChecker,
    TrafficFeasibilityChecker,
    build_mode_valid_mask,
)
from metadrive.policy.diffusion_policy.mode_definitions import (
    BehaviorType,
    ModeSlot,
    SpeedProfile,
    build_mode_slots,
)


def _polyline_arc_lengths(polyline: np.ndarray) -> np.ndarray:
    deltas = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)], axis=0)


def _sample_polyline_at_distances(polyline: np.ndarray, distances: np.ndarray) -> np.ndarray:
    polyline = np.asarray(polyline, dtype=np.float32)
    if polyline.shape[0] == 1:
        return np.repeat(polyline, len(distances), axis=0)
    arc_lengths = _polyline_arc_lengths(polyline)
    total_length = float(arc_lengths[-1])
    if total_length <= 1e-6:
        return np.repeat(polyline[:1], len(distances), axis=0)
    # Compute the extrapolation direction from the last segment for distances
    # beyond the polyline.  Linear extrapolation prevents the fold-back artefact
    # that occurs when clamping causes trajectory points to pile up at the
    # polyline endpoint while the quintic blend still transitions laterally.
    last_dir = polyline[-1] - polyline[-2]
    last_dir_norm = float(np.linalg.norm(last_dir))
    if last_dir_norm > 1e-6:
        last_unit = last_dir / last_dir_norm
    else:
        last_unit = np.array([1.0, 0.0], dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)
    sampled = []
    for distance in distances:
        d = float(distance)
        if d > total_length:
            # Extrapolate linearly beyond the polyline endpoint.
            sampled.append(polyline[-1] + (d - total_length) * last_unit)
        else:
            d = max(d, 0.0)
            segment_idx = int(np.searchsorted(arc_lengths, d, side="right") - 1)
            segment_idx = max(0, min(segment_idx, polyline.shape[0] - 2))
            start_s = float(arc_lengths[segment_idx])
            end_s = float(arc_lengths[segment_idx + 1])
            ratio = 0.0 if end_s - start_s <= 1e-6 else float((d - start_s) / (end_s - start_s))
            sampled.append(polyline[segment_idx] + ratio * (polyline[segment_idx + 1] - polyline[segment_idx]))
    return np.asarray(sampled, dtype=np.float32)


def _resample_to_fixed_steps(polyline: np.ndarray, num_steps: int, speed_mps: float, dt: float) -> np.ndarray:
    distances = np.cumsum(np.full((num_steps,), max(float(speed_mps), 0.0) * float(dt), dtype=np.float32))
    return _sample_polyline_at_distances(polyline, distances)


def _quintic_blend(progress: np.ndarray) -> np.ndarray:
    p = np.clip(progress, 0.0, 1.0)
    return (10.0 * p**3 - 15.0 * p**4 + 6.0 * p**5).astype(np.float32)


@dataclass
class ModeTrajectoryOutput:
    coarse_trajectories: np.ndarray
    mode_valid_mask: np.ndarray


class ModeTrajectoryGenerator:
    HORIZON_STEPS = 8
    DT = 0.5

    def __init__(
        self,
        horizon_steps: int = HORIZON_STEPS,
        dt: float = DT,
        keep_lane_high_speed_mps: float = 13.0,
        keep_lane_medium_speed_mps: float = 8.0,
        keep_lane_low_speed_mps: float = 3.0,
        emergency_decel_mps2: float = 4.5,
        lane_change_progress_steps: Optional[int] = None,
        keep_lane_level_count: int = 3,
        lane_change_left_level_count: int = 3,
        lane_change_right_level_count: int = 3,
        emergency_stop_level_count: int = 1,
        mode_slots: Optional[Sequence[ModeSlot]] = None,
        collision_prediction_enabled: bool = True,
        collision_check_clearance_m: float = 0.8,
        ego_collision_radius_m: float = 2.6,
        collision_speed_scale_candidates: Sequence[float] = (1.0, 0.85, 0.7, 0.55, 0.4),
        geometric_checker: Optional[GeometricFeasibilityChecker] = None,
        traffic_checker: Optional[TrafficFeasibilityChecker] = None,
    ) -> None:
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.keep_lane_high_speed_mps = float(keep_lane_high_speed_mps)
        self.keep_lane_medium_speed_mps = float(keep_lane_medium_speed_mps)
        self.keep_lane_low_speed_mps = float(keep_lane_low_speed_mps)
        self.emergency_decel_mps2 = float(emergency_decel_mps2)
        self.lane_change_progress_steps = int(lane_change_progress_steps or horizon_steps)
        self.keep_lane_level_count = int(keep_lane_level_count)
        self.lane_change_left_level_count = int(lane_change_left_level_count)
        self.lane_change_right_level_count = int(lane_change_right_level_count)
        self.emergency_stop_level_count = int(emergency_stop_level_count)
        self.mode_slots = tuple(
            mode_slots
            if mode_slots is not None
            else build_mode_slots(
                keep_lane_count=self.keep_lane_level_count,
                lane_change_left_count=self.lane_change_left_level_count,
                lane_change_right_count=self.lane_change_right_level_count,
                emergency_stop_count=self.emergency_stop_level_count,
            )
        )
        self.num_mode_slots = len(self.mode_slots)
        self.collision_prediction_enabled = bool(collision_prediction_enabled)
        self.collision_check_clearance_m = float(collision_check_clearance_m)
        self.ego_collision_radius_m = float(ego_collision_radius_m)
        self.collision_speed_scale_candidates = tuple(float(scale) for scale in collision_speed_scale_candidates)
        self.geometric_checker = geometric_checker or GeometricFeasibilityChecker()
        self.traffic_checker = traffic_checker or TrafficFeasibilityChecker()

    def _interpolated_speed(self, ctx: ModeContext, slot: ModeSlot) -> float:
        ego_speed = max(float(ctx.ego_speed_mps), 0.0)
        low = min(ego_speed, self.keep_lane_low_speed_mps)
        high = max(ego_speed, self.keep_lane_high_speed_mps)
        speed = low + float(slot.level_fraction) * (high - low)
        return max(float(speed), 0.0)

    def _desired_speed(self, ctx: ModeContext, slot: ModeSlot) -> float:
        ego_speed = max(float(ctx.ego_speed_mps), 0.0)
        if slot.semantic_group == "KEEP" and slot.name == "KEEP_HIGH":
            return max(ego_speed, self.keep_lane_high_speed_mps)
        if slot.semantic_group == "KEEP" and slot.name == "KEEP_MEDIUM":
            front_speed = float(ctx.front_object_speed_mps)
            if ctx.front_object_distance >= 0.0:
                return max(min(ego_speed, front_speed if front_speed > 0.0 else ego_speed), 1.0)
            return max(min(ego_speed, self.keep_lane_medium_speed_mps), 1.0)
        if slot.semantic_group == "KEEP" and slot.name == "KEEP_LOW":
            return min(ego_speed, self.keep_lane_low_speed_mps)
        if slot.name.startswith("KEEP_LEVEL_") or slot.name.startswith("LEFT_LC_LEVEL_") or slot.name.startswith("RIGHT_LC_LEVEL_"):
            return self._interpolated_speed(ctx, slot)
        if slot.speed_profile == SpeedProfile.HIGH:
            return max(ego_speed, self.keep_lane_high_speed_mps)
        if slot.speed_profile == SpeedProfile.MEDIUM:
            return max(min(ego_speed, self.keep_lane_medium_speed_mps), 1.0)
        return min(ego_speed, self.keep_lane_low_speed_mps)

    def _generate_keep_lane(self, ctx: ModeContext, speed_mps: float) -> np.ndarray:
        return _resample_to_fixed_steps(ctx.current_lane_polyline, self.horizon_steps, speed_mps, self.dt)

    def _generate_lane_change(self, ctx: ModeContext, target_polyline: np.ndarray, speed_mps: float) -> np.ndarray:
        # Enforce a minimum forward speed so the trajectory has enough forward
        # distance for a smooth lateral transition (~1 lane width over 8 steps).
        # Without this floor, near-zero ego speed produces near-vertical
        # trajectories with extreme heading kinks.
        effective_speed = max(speed_mps, 3.0)
        base = self._generate_keep_lane(ctx, effective_speed)
        target = _resample_to_fixed_steps(target_polyline, self.horizon_steps, effective_speed, self.dt)
        progress = np.linspace(1.0 / self.lane_change_progress_steps, 1.0, self.horizon_steps, dtype=np.float32)
        blend = _quintic_blend(progress)[:, None]
        return (1.0 - blend) * base + blend * target

    def _generate_emergency_stop(self, ctx: ModeContext) -> np.ndarray:
        speed = max(float(ctx.ego_speed_mps), 0.0)
        xs = []
        distance = 0.0
        for step in range(self.horizon_steps):
            current_speed = max(speed - self.emergency_decel_mps2 * self.dt * step, 0.0)
            distance += current_speed * self.dt
            xs.append(distance)
        return np.stack([np.asarray(xs, dtype=np.float32), np.zeros((self.horizon_steps,), dtype=np.float32)], axis=1)

    @staticmethod
    def _lane_change_target(ctx: ModeContext, slot: ModeSlot) -> np.ndarray | None:
        if slot.lateral_direction == "left":
            return ctx.left_lane_polyline if ctx.left_lane_polyline is not None else ctx.left_branch_polyline
        if slot.lateral_direction == "right":
            return ctx.right_lane_polyline if ctx.right_lane_polyline is not None else ctx.right_branch_polyline
        return None

    def _predict_obstacle_positions(self, obstacle: DynamicObstacle) -> np.ndarray:
        times = (np.arange(self.horizon_steps, dtype=np.float32) + 1.0) * float(self.dt)
        return obstacle.initial_position_xy[None, :] + times[:, None] * obstacle.velocity_xy[None, :]

    def _is_collision_free(self, ctx: ModeContext, trajectory: np.ndarray) -> bool:
        if not self.collision_prediction_enabled or not ctx.dynamic_obstacles:
            return True
        trajectory = np.asarray(trajectory, dtype=np.float32)
        ego_radius = float(self.ego_collision_radius_m)
        clearance = float(self.collision_check_clearance_m)
        for obstacle in ctx.dynamic_obstacles:
            predicted = self._predict_obstacle_positions(obstacle)
            distances = np.linalg.norm(trajectory - predicted, axis=1)
            if np.any(distances < ego_radius + float(obstacle.radius_m) + clearance):
                return False
        return True

    def _generate_slot_trajectory(self, ctx: ModeContext, slot: ModeSlot, speed_mps: float) -> np.ndarray | None:
        if slot.semantic_group == "KEEP":
            return self._generate_keep_lane(ctx, speed_mps)
        if slot.behavior_type == BehaviorType.LANE_CHANGE:
            target_polyline = self._lane_change_target(ctx, slot)
            if target_polyline is None:
                return None
            return self._generate_lane_change(ctx, target_polyline, speed_mps)
        if slot.semantic_group == "STOP":
            return self._generate_emergency_stop(ctx)
        return None

    def _generate_collision_aware_trajectory(self, ctx: ModeContext, slot: ModeSlot, base_speed_mps: float) -> np.ndarray | None:
        # KEEP_LOW and STOP always produce a trajectory
        # regardless of surrounding obstacles so the vehicle always has at
        # least one actionable mode.  Faster KEEP modes (HIGH / MEDIUM)
        # are subject to collision checking — when the ego drives fast toward
        # a close front vehicle the trajectory would overlap, so those modes
        # should become invalid to nudge the planner toward deceleration.
        if slot.semantic_group == "KEEP":
            trajectory = self._generate_slot_trajectory(ctx, slot, base_speed_mps)
            if slot.level_fraction > 0.0:
                if trajectory is not None and not self._is_collision_free(ctx, trajectory):
                    return None
            return trajectory
        scales = self.collision_speed_scale_candidates
        for scale in scales:
            candidate = self._generate_slot_trajectory(ctx, slot, base_speed_mps * float(scale))
            if candidate is None:
                return None
            if self._is_collision_free(ctx, candidate):
                return candidate
        return None

    def generate(self, ctx: ModeContext) -> ModeTrajectoryOutput:
        feasibility = build_mode_valid_mask(ctx, self.geometric_checker, self.traffic_checker, self.mode_slots)
        valid_mask = feasibility.valid_mask.copy()
        coarse = np.zeros((self.num_mode_slots, self.horizon_steps, 2), dtype=np.float32)

        for slot in self.mode_slots:
            if not bool(valid_mask[slot.index]) and slot.semantic_group != "STOP":
                continue
            trajectory = self._generate_collision_aware_trajectory(ctx, slot, self._desired_speed(ctx, slot))
            if trajectory is None:
                valid_mask[slot.index] = False
                continue
            coarse[slot.index] = trajectory

        for slot in self.mode_slots:
            if slot.semantic_group == "STOP":
                valid_mask[slot.index] = True

        coarse[~valid_mask] = 0.0
        return ModeTrajectoryOutput(coarse_trajectories=coarse, mode_valid_mask=valid_mask.astype(bool))
