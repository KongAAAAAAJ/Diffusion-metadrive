from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from expert_dataset.hierarchical_expert.driving_style import DrivingStyleProfile
from expert_dataset.hierarchical_expert.safety_assessor import GapInfo, SafetyAssessor


@dataclass
class MockObject:
    speed_km_h: float


class MockFrontBackObjects:
    def __init__(self, left_front=None, left_back=None, right_front=None, right_back=None, left_exist=True, right_exist=True):
        self._left_front = left_front
        self._left_back = left_back
        self._right_front = right_front
        self._right_back = right_back
        self._left_exist = left_exist
        self._right_exist = right_exist

    def left_lane_exist(self):
        return self._left_exist

    def right_lane_exist(self):
        return self._right_exist

    def has_left_front_object(self):
        return self._left_front is not None

    def has_left_back_object(self):
        return self._left_back is not None

    def has_right_front_object(self):
        return self._right_front is not None

    def has_right_back_object(self):
        return self._right_back is not None

    def left_front_object(self):
        return self._left_front

    def left_back_object(self):
        return self._left_back

    def right_front_object(self):
        return self._right_front

    def right_back_object(self):
        return self._right_back

    def left_front_min_distance(self):
        return 50.0

    def left_back_min_distance(self):
        return 40.0

    def right_front_min_distance(self):
        return 50.0

    def right_back_min_distance(self):
        return 40.0


def make_perception(front_distance=50.0, front_speed=25.0, rear_distance=40.0, rear_speed=15.0, direction=-1):
    front = MockObject(speed_km_h=front_speed * 3.6)
    rear = MockObject(speed_km_h=rear_speed * 3.6)
    if direction < 0:
        perception = MockFrontBackObjects(left_front=front, left_back=rear)
        perception.left_front_min_distance = lambda: front_distance
        perception.left_back_min_distance = lambda: rear_distance
        return perception
    perception = MockFrontBackObjects(right_front=front, right_back=rear)
    perception.right_front_min_distance = lambda: front_distance
    perception.right_back_min_distance = lambda: rear_distance
    return perception


def test_rss_safe_distance_matches_formula():
    assessor = SafetyAssessor(DrivingStyleProfile())
    rss = assessor._rss_safe_distance(20.0, 15.0, 0.5, 2.0, 1.5)

    assert np.isclose(rss, 87.0833333333, atol=1e-6)


def test_compute_gap_info_ttc_rules():
    assessor = SafetyAssessor(DrivingStyleProfile())
    perception = make_perception(front_distance=50.0, front_speed=25.0, rear_distance=40.0, rear_speed=35.0, direction=-1)

    gap = assessor.compute_gap_info(-1, ego_speed=30.0, perception=perception)
    slower_front_gap = assessor.compute_gap_info(-1, ego_speed=20.0, perception=perception)

    assert isinstance(gap, GapInfo)
    assert np.isclose(gap.front_ttc, 10.0)
    assert np.isinf(slower_front_gap.front_ttc)


def test_is_gap_acceptable_rejects_small_front_ttc():
    assessor = SafetyAssessor(DrivingStyleProfile())
    perception = make_perception(front_distance=10.0, front_speed=20.0, rear_distance=60.0, rear_speed=10.0, direction=-1)

    assert assessor.is_gap_acceptable(-1, ego_speed=30.0, perception=perception) is False


def test_is_gap_acceptable_accepts_safe_case():
    assessor = SafetyAssessor(DrivingStyleProfile())
    perception = make_perception(front_distance=50.0, front_speed=20.0, rear_distance=150.0, rear_speed=20.0, direction=-1)

    assert assessor.is_gap_acceptable(-1, ego_speed=30.0, perception=perception) is True


def test_monitor_ongoing_is_more_permissive_than_initial_gate():
    assessor = SafetyAssessor(DrivingStyleProfile())
    perception = make_perception(front_distance=20.0, front_speed=20.0, rear_distance=150.0, rear_speed=20.0, direction=-1)

    assert assessor.monitor_ongoing(-1, ego_speed=30.0, perception=perception) is True
    assert assessor.is_gap_acceptable(-1, ego_speed=30.0, perception=perception) is False


def test_gap_acceptance_score_is_clipped():
    assessor = SafetyAssessor(DrivingStyleProfile())
    perception = make_perception(front_distance=5.0, front_speed=29.0, rear_distance=5.0, rear_speed=40.0, direction=-1)

    score = assessor.gap_acceptance_score(-1, ego_speed=30.0, perception=perception)
    assert -1.0 <= score <= 1.0


def test_compute_gap_info_handles_missing_neighbor_lane():
    assessor = SafetyAssessor(DrivingStyleProfile())
    perception = MockFrontBackObjects(left_exist=False, right_exist=True)

    gap = assessor.compute_gap_info(-1, ego_speed=20.0, perception=perception)
    assert gap.front_distance == 999.0
    assert gap.rear_distance == 999.0
    assert np.isinf(gap.front_ttc)
    assert np.isinf(gap.rear_ttc)
