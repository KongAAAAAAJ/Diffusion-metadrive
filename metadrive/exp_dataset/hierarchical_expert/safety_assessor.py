from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

from metadrive.exp_dataset.hierarchical_expert.driving_style import DrivingStyleProfile


@dataclass
class GapInfo:
    front_distance: float
    front_speed: float
    rear_distance: float
    rear_speed: float
    front_ttc: float
    rear_ttc: float


class SafetyAssessor:
    def __init__(self, style: DrivingStyleProfile):
        self.style = style

    def compute_gap_info(self, direction, ego_speed, perception) -> GapInfo:
        if direction < 0:
            if not perception.left_lane_exist():
                return GapInfo(999.0, 0.0, 999.0, 0.0, inf, inf)
            front_obj = perception.left_front_object() if perception.has_left_front_object() else None
            rear_obj = perception.left_back_object() if perception.has_left_back_object() else None
            front_distance = float(perception.left_front_min_distance())
            rear_distance = float(perception.left_back_min_distance())
        else:
            if not perception.right_lane_exist():
                return GapInfo(999.0, 0.0, 999.0, 0.0, inf, inf)
            front_obj = perception.right_front_object() if perception.has_right_front_object() else None
            rear_obj = perception.right_back_object() if perception.has_right_back_object() else None
            front_distance = float(perception.right_front_min_distance())
            rear_distance = float(perception.right_back_min_distance())

        front_speed = float(getattr(front_obj, "speed_km_h", 0.0) / 3.6) if front_obj is not None else 0.0
        rear_speed = float(getattr(rear_obj, "speed_km_h", 0.0) / 3.6) if rear_obj is not None else 0.0

        front_ttc = inf
        if front_obj is not None and ego_speed > front_speed:
            front_ttc = float(front_distance / max(ego_speed - front_speed, 0.01))

        rear_ttc = inf
        if rear_obj is not None and rear_speed > ego_speed:
            rear_ttc = float(rear_distance / max(rear_speed - ego_speed, 0.01))

        return GapInfo(front_distance, front_speed, rear_distance, rear_speed, front_ttc, rear_ttc)

    def _rss_safe_distance(self, v_rear, v_front, reaction_time, a_max_brake, a_min_brake) -> float:
        distance = v_rear * reaction_time + v_rear**2 / max(2.0 * a_min_brake, 1e-6) - v_front**2 / max(2.0 * a_max_brake, 1e-6)
        return float(max(distance, 2.0))

    def is_gap_acceptable(self, direction, ego_speed, perception) -> bool:
        gap = self.compute_gap_info(direction, ego_speed, perception)
        rear_safe = self._rss_safe_distance(
            gap.rear_speed,
            ego_speed,
            self.style.reaction_time,
            self.style.comfortable_decel,
            self.style.comfortable_decel * 0.75,
        )
        return bool(
            gap.front_ttc >= self.style.min_front_ttc and
            gap.rear_ttc >= self.style.min_rear_ttc and
            gap.rear_distance >= rear_safe
        )

    def gap_acceptance_score(self, direction, ego_speed, perception) -> float:
        gap = self.compute_gap_info(direction, ego_speed, perception)
        rear_safe = self._rss_safe_distance(
            gap.rear_speed,
            ego_speed,
            self.style.reaction_time,
            self.style.comfortable_decel,
            self.style.comfortable_decel * 0.75,
        )
        front_ratio = gap.front_ttc / max(self.style.min_front_ttc, 1e-6)
        rear_ratio = gap.rear_ttc / max(self.style.min_rear_ttc, 1e-6)
        rss_ratio = gap.rear_distance / max(rear_safe, 1e-6)
        score = min(front_ratio, rear_ratio, rss_ratio) - 1.0
        return float(np.clip(score, -1.0, 1.0))

    def monitor_ongoing(self, direction, ego_speed, perception) -> bool:
        gap = self.compute_gap_info(direction, ego_speed, perception)
        if gap.front_distance < 3.0 or gap.rear_distance < 3.0:
            return False

        rear_safe = self._rss_safe_distance(
            gap.rear_speed,
            ego_speed,
            self.style.reaction_time,
            self.style.comfortable_decel,
            self.style.comfortable_decel * 0.75,
        )
        return bool(
            gap.front_ttc >= self.style.min_front_ttc * 0.6 and
            gap.rear_ttc >= self.style.min_rear_ttc * 0.6 and
            gap.rear_distance >= rear_safe * 0.6
        )
