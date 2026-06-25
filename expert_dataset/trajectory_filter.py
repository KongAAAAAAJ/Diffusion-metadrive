from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def _world_future_to_local(current_pose: np.ndarray, future_pose: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    future_pose = np.asarray(future_pose, dtype=np.float32)
    dx = float(future_pose[0] - current_pose[0])
    dy = float(future_pose[1] - current_pose[1])
    heading = float(current_pose[2])
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    d_heading = float(future_pose[2] - current_pose[2])
    return np.asarray([local_x, local_y, d_heading], dtype=np.float32)


@dataclass
class FilterVerdict:
    passed: bool
    rule_name: str
    reason: str = ""
    missing_reference_lane_points: int = 0


@dataclass
class FilterResult:
    passed: bool
    verdicts: List[FilterVerdict]

    @property
    def rejection_reasons(self) -> List[str]:
        return [verdict.reason for verdict in self.verdicts if not verdict.passed and verdict.reason]

    @property
    def missing_reference_lane_points(self) -> int:
        return int(sum(verdict.missing_reference_lane_points for verdict in self.verdicts))


class TrajectoryFilterRule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def check_sample(self, sample: Dict[str, np.ndarray]) -> FilterVerdict:
        raise NotImplementedError


class TrajectoryFilterPipeline:
    def __init__(self, rules: List[TrajectoryFilterRule]):
        self.rules = list(rules)

    def check_sample(self, sample: Dict[str, np.ndarray]) -> FilterResult:
        verdicts = [rule.check_sample(sample) for rule in self.rules]
        return FilterResult(
            passed=all(verdict.passed for verdict in verdicts),
            verdicts=verdicts,
        )


class OutOfRoadByReferenceLaneRule(TrajectoryFilterRule):
    def __init__(
        self,
        out_of_road_margin_ratio: float = 0.0,
        out_of_road_missing_lane_policy: str = "skip_point",
    ):
        self.out_of_road_margin_ratio = float(out_of_road_margin_ratio)
        self.out_of_road_missing_lane_policy = str(out_of_road_missing_lane_policy)

    @property
    def name(self) -> str:
        return "out_of_road"

    def check_sample(self, sample: Dict[str, np.ndarray]) -> FilterVerdict:
        future_ego = np.asarray(sample["future_ego_pose_world"], dtype=np.float32)
        future_reference = np.asarray(sample["future_reference_pose_world"], dtype=np.float32)
        future_lane_width = np.asarray(sample["future_lane_width"], dtype=np.float32)
        future_lane_index = np.asarray(sample["future_reference_lane_index"], dtype=np.int16)

        missing_reference_lane_points = 0
        for idx in range(future_ego.shape[0]):
            if int(future_lane_index[idx]) < 0:
                missing_reference_lane_points += 1
                if self.out_of_road_missing_lane_policy == "reject_sample":
                    return FilterVerdict(
                        passed=False,
                        rule_name=self.name,
                        reason=self.name,
                        missing_reference_lane_points=missing_reference_lane_points,
                    )
                continue

            local_pose = _world_future_to_local(future_reference[idx], future_ego[idx])
            lateral = float(abs(local_pose[1]))
            threshold = 0.5 * float(future_lane_width[idx]) * (1.0 + self.out_of_road_margin_ratio)
            if lateral > threshold:
                return FilterVerdict(
                    passed=False,
                    rule_name=self.name,
                    reason=self.name,
                    missing_reference_lane_points=missing_reference_lane_points,
                )

        return FilterVerdict(
            passed=True,
            rule_name=self.name,
            missing_reference_lane_points=missing_reference_lane_points,
        )
