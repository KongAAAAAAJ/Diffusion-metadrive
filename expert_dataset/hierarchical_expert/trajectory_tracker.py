from __future__ import annotations

import math

import numpy as np

from expert_dataset.expert_idm_policy import ExpertIDMPolicy


class PurePursuitTracker:
    PP_LOOKAHEAD_SPEED_GAIN: float = 0.65
    PP_MIN_LOOKAHEAD: float = 3.5
    PP_MAX_LOOKAHEAD: float = 10.0
    PP_WHEELBASE: float = 2.8
    MAX_STEERING: float = ExpertIDMPolicy.MAX_STEERING
    MAX_STEER_DELTA_PER_STEP: float = 0.025
    END_SEGMENT_LOOKAHEAD_MULTIPLIER: float = 1.25
    END_SEGMENT_THRESHOLD: float = 6.0

    def __init__(self) -> None:
        self._last_steering = None

    def _compute_lookahead(self, ego_speed: float, current_s: float, trajectory_length: float) -> float:
        lookahead = float(
            np.clip(
                self.PP_LOOKAHEAD_SPEED_GAIN * float(ego_speed),
                self.PP_MIN_LOOKAHEAD,
                self.PP_MAX_LOOKAHEAD,
            )
        )
        remaining = float(trajectory_length) - float(current_s)
        if remaining < self.END_SEGMENT_THRESHOLD:
            lookahead = min(lookahead * self.END_SEGMENT_LOOKAHEAD_MULTIPLIER, self.PP_MAX_LOOKAHEAD)
        return float(lookahead)

    def _apply_rate_limit(self, steering: float) -> float:
        if self._last_steering is None:
            self._last_steering = float(steering)
            return float(steering)
        lower = self._last_steering - self.MAX_STEER_DELTA_PER_STEP
        upper = self._last_steering + self.MAX_STEER_DELTA_PER_STEP
        limited = float(np.clip(steering, lower, upper))
        self._last_steering = limited
        return limited

    def reset(self) -> None:
        self._last_steering = None

    def compute_steering(self, ego_position: np.ndarray, ego_heading: float, ego_speed: float, trajectory) -> float:
        current_s, _ = trajectory.local_coordinates(ego_position)
        current_s = float(current_s)
        lookahead = self._compute_lookahead(ego_speed=ego_speed, current_s=current_s, trajectory_length=float(trajectory.length))
        lookahead_s = min(current_s + lookahead, float(trajectory.length))
        target_point = trajectory.position(lookahead_s, 0.0)

        dx = float(target_point[0]) - float(ego_position[0])
        dy = float(target_point[1]) - float(ego_position[1])

        cos_h = math.cos(float(ego_heading))
        sin_h = math.sin(float(ego_heading))
        local_y = -dx * sin_h + dy * cos_h

        ld_sq = dx * dx + dy * dy
        if ld_sq < 1e-6:
            return 0.0

        # Pure Pursuit控制律：steering = atan(2 * L * sin(alpha) / ld)，其中alpha = atan(local_y / lookahead)，L为轴距，ld为当前点与目标点的距离
        curvature = 2.0 * local_y / ld_sq
        steering = math.atan(curvature * self.PP_WHEELBASE)
        steering = float(np.clip(steering, -self.MAX_STEERING, self.MAX_STEERING))
        return self._apply_rate_limit(steering)

    def is_trajectory_ended(self, ego_position: np.ndarray, trajectory, threshold: float = 2.0) -> bool:
        current_s, _ = trajectory.local_coordinates(ego_position)
        return float(current_s) >= float(trajectory.length) - float(threshold)
