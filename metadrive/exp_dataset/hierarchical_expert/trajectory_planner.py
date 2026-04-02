from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.exp_dataset.hierarchical_expert.driving_style import DrivingStyleProfile


@dataclass
class LaneChangeTrajectoryConfig:
    # Default expert collection/evaluation setup fixes normal lane changes to 4.0s
    # so dataset generation and offline evaluation follow the same timing.
    min_duration: float = 4.0
    max_duration: float = 4.0
    num_sample_points: int = 60
    max_curvature_factor: float = 0.8
    terminal_lateral_tolerance: float = 0.3
    terminal_heading_tolerance: float = 0.15
    max_heading_jump: float = 0.2
    corridor_margin: float = 0.2


class QuinticLaneChangePlanner:
    def __init__(self, style: DrivingStyleProfile, config: LaneChangeTrajectoryConfig = None):
        self.style = style
        self.config = config or LaneChangeTrajectoryConfig()

    def _solve_quintic_coefficients(self, d0, d0_dot, d0_ddot, df, df_dot, df_ddot, T) -> np.ndarray:
        a0 = d0
        a1 = d0_dot
        a2 = d0_ddot / 2.0
        matrix = np.asarray(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ],
            dtype=np.float64,
        )
        rhs = np.asarray(
            [
                df - (a0 + a1 * T + a2 * T**2),
                df_dot - (a1 + 2 * a2 * T),
                df_ddot - (2 * a2),
            ],
            dtype=np.float64,
        )
        a3, a4, a5 = np.linalg.solve(matrix, rhs)
        return np.asarray([a0, a1, a2, a3, a4, a5], dtype=np.float64)

    def _evaluate_quintic(self, coeffs, t) -> Tuple[float, float, float]:
        d = sum(coeff * (t**idx) for idx, coeff in enumerate(coeffs))
        d_dot = sum(idx * coeff * (t**(idx - 1)) for idx, coeff in enumerate(coeffs) if idx >= 1)
        d_ddot = sum(idx * (idx - 1) * coeff * (t**(idx - 2)) for idx, coeff in enumerate(coeffs) if idx >= 2)
        return float(d), float(d_dot), float(d_ddot)

    def _compute_duration(self, ego_speed, lateral_distance, urgency) -> float:
        base = abs(lateral_distance) / max(self.style.desired_lateral_speed, 0.5)
        speed_factor = float(np.clip(ego_speed / 20.0, 0.8, 1.5))
        urgency_factor = 1.0 - 0.3 * float(np.clip(urgency, 0.0, 1.0))
        duration = base * speed_factor * urgency_factor
        return float(np.clip(duration, self.config.min_duration, self.config.max_duration))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float(math.atan2(math.sin(angle), math.cos(angle)))

    def _validate_curvature(self, coeffs, T, ego_speed, wheel_base=2.8) -> bool:
        times = np.linspace(0.0, T, 30)
        curvature_limit = np.tan(0.3149) / wheel_base * self.config.max_curvature_factor
        longitudinal_speed = max(float(ego_speed), 1e-3)
        lateral_speed_limit = max(self.style.desired_lateral_speed * 4.0, 1.5)
        for t in times:
            _, d_dot, d_ddot = self._evaluate_quintic(coeffs, float(t))
            if abs(d_dot) > lateral_speed_limit:
                return False
            curvature = abs(longitudinal_speed * d_ddot) / max((longitudinal_speed**2 + d_dot**2) ** 1.5, 1e-6)
            if curvature >= curvature_limit:
                return False
        return True

    def _sample_frenet_points(
        self,
        source_lane,
        start_longitudinal: float,
        ego_speed: float,
        start_lateral: float,
        target_offset: float,
        duration: float,
        clip_longitudinal: bool = True,
    ) -> np.ndarray:
        travel_distance = max(float(ego_speed) * float(duration), 0.0)
        num_points = max(int(travel_distance / 1.5), 20)
        num_points = min(num_points, self.config.num_sample_points)
        times = np.linspace(0.0, duration, num_points)
        coeffs = self._solve_quintic_coefficients(
            float(start_lateral),
            0.0,
            0.0,
            float(target_offset),
            0.0,
            0.0,
            float(duration),
        )
        source_length = float(source_lane.length)
        points = []
        for t in times:
            longitudinal = float(start_longitudinal) + float(ego_speed) * float(t)
            if clip_longitudinal:
                longitudinal = float(np.clip(longitudinal, 0.0, source_length))
            lateral, _, _ = self._evaluate_quintic(coeffs, float(t))
            points.append(np.asarray(source_lane.position(longitudinal, lateral), dtype=np.float64))
        return np.asarray(points, dtype=np.float64)

    def _validate_world_path_curvature(self, points: np.ndarray, wheel_base: float = 2.8) -> bool:
        if len(points) < 3:
            return False
        deltas = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        valid_segments = segment_lengths > 1e-6
        if np.count_nonzero(valid_segments) < 2:
            return False
        filtered_deltas = deltas[valid_segments]
        filtered_lengths = segment_lengths[valid_segments]
        headings = np.unwrap(np.arctan2(filtered_deltas[:, 1], filtered_deltas[:, 0]))
        if len(headings) < 2:
            return True
        curvature_limit = np.tan(0.3149) / wheel_base * self.config.max_curvature_factor
        heading_deltas = np.diff(headings)
        avg_lengths = 0.5 * (filtered_lengths[:-1] + filtered_lengths[1:])
        discrete_curvature = np.abs(heading_deltas) / np.maximum(avg_lengths, 1e-6)
        return bool(np.all(discrete_curvature < curvature_limit))

    def _validate_world_path(
        self,
        points: np.ndarray,
        source_lane,
        target_lane,
        target_longitudinal: float,
        *,
        enforce_curvature: bool = True,
    ) -> bool:
        if len(points) < 3:
            return False
        if enforce_curvature and not self._validate_world_path_curvature(points):
            return False

        end_point = points[-1]
        _, final_lateral = target_lane.local_coordinates(end_point)
        if abs(float(final_lateral)) >= self.config.terminal_lateral_tolerance:
            return False

        deltas = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        valid_segments = segment_lengths > 1e-6
        if np.count_nonzero(valid_segments) == 0:
            return False
        filtered_deltas = deltas[valid_segments]
        terminal_delta = filtered_deltas[-1]
        terminal_heading = float(np.arctan2(terminal_delta[1], terminal_delta[0]))
        target_heading = float(target_lane.heading_theta_at(float(np.clip(target_longitudinal, 0.0, float(target_lane.length)))))
        if abs(self._wrap_to_pi(terminal_heading - target_heading)) >= self.config.terminal_heading_tolerance:
            return False

        headings = np.unwrap(np.arctan2(filtered_deltas[:, 1], filtered_deltas[:, 0]))
        if len(headings) >= 2 and np.max(np.abs(np.diff(headings))) >= self.config.max_heading_jump:
            return False

        max_corridor_lateral = max(float(source_lane.width), float(target_lane.width)) / 2.0 + self.config.corridor_margin
        for point in points:
            _, source_lateral = source_lane.local_coordinates(point)
            _, target_lateral = target_lane.local_coordinates(point)
            if min(abs(float(source_lateral)), abs(float(target_lateral))) > max_corridor_lateral:
                return False
        return True

    def _build_point_lane(self, points: np.ndarray, width: float) -> PointLane:
        return PointLane(center_line_points=np.asarray(points, dtype=np.float64), width=float(width))

    def plan(self, ego_position, ego_heading, ego_speed, source_lane, target_lane, direction, urgency=0.0) -> Optional[PointLane]:
        del direction
        longitudinal, lateral = source_lane.local_coordinates(ego_position)
        source_remaining = max(float(source_lane.length) - float(longitudinal), 0.0)
        min_required = float(ego_speed) * self.config.min_duration * 0.3
        allow_short_mandatory = urgency > 0.0 and source_remaining < max(min_required, 5.0)
        if not allow_short_mandatory and source_remaining < max(min_required, 5.0):
            return None
        enforce_curvature = source_remaining > max(float(source_lane.width) * 1.5, 5.0)
        reference_position = source_lane.position(np.clip(longitudinal, 0.0, float(source_lane.length)), 0.0)
        target_longitudinal, _ = target_lane.local_coordinates(reference_position)
        target_longitudinal = float(np.clip(target_longitudinal, 0.0, float(target_lane.length)))
        target_position = target_lane.position(target_longitudinal, 0.0)
        _, target_offset = source_lane.local_coordinates(target_position)
        duration = self._compute_duration(ego_speed, target_offset - lateral, urgency)
        for _ in range(3):
            points = self._sample_frenet_points(
                source_lane=source_lane,
                start_longitudinal=float(longitudinal),
                ego_speed=float(ego_speed),
                start_lateral=float(lateral),
                target_offset=float(target_offset),
                duration=duration,
                clip_longitudinal=not allow_short_mandatory,
            )
            terminal_target_longitudinal, _ = target_lane.local_coordinates(points[-1])
            terminal_target_longitudinal = float(np.clip(terminal_target_longitudinal, 0.0, float(target_lane.length)))
            if self._validate_world_path(
                points,
                source_lane,
                target_lane,
                terminal_target_longitudinal,
                enforce_curvature=enforce_curvature,
            ):
                return self._build_point_lane(points, width=float(source_lane.width))
            duration = min(duration * 1.3, self.config.max_duration * 2.0)
        return None

    def plan_abort(self, ego_position, ego_heading, ego_speed, source_lane) -> PointLane:
        longitudinal, lateral = source_lane.local_coordinates(ego_position)
        duration = float(np.clip(abs(lateral) / max(self.style.desired_lateral_speed * 0.8, 0.3), 2.0, 5.0))
        for _ in range(3):
            points = self._sample_frenet_points(
                source_lane=source_lane,
                start_longitudinal=float(longitudinal),
                ego_speed=float(ego_speed),
                start_lateral=float(lateral),
                target_offset=0.0,
                duration=duration,
            )
            terminal_target_longitudinal, _ = source_lane.local_coordinates(points[-1])
            terminal_target_longitudinal = float(np.clip(terminal_target_longitudinal, 0.0, float(source_lane.length)))
            if self._validate_world_path(points, source_lane, source_lane, terminal_target_longitudinal):
                return self._build_point_lane(points, width=float(source_lane.width))
            duration = min(duration * 1.3, 8.0)
        return self._build_point_lane(points, width=float(source_lane.width))
