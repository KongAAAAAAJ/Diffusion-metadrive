from __future__ import annotations

import numpy as np

from models.diffusion.mode_context import ModeContext
from models.diffusion.mode_feasibility import (
    GeometricFeasibilityChecker,
    TrafficFeasibilityChecker,
    build_mode_valid_mask,
)


def _straight_polyline(y_offset: float = 0.0, length: float = 40.0, num_points: int = 21) -> np.ndarray:
    xs = np.linspace(0.0, length, num_points, dtype=np.float32)
    ys = np.full((num_points,), y_offset, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def _context(**overrides) -> ModeContext:
    base = dict(
        ego_speed_mps=10.0,
        ego_heading=0.0,
        current_lane_polyline=_straight_polyline(0.0),
        current_lane_width=4.0,
        left_lane_polyline=_straight_polyline(4.0),
        right_lane_polyline=_straight_polyline(-4.0),
        left_branch_polyline=_straight_polyline(3.0),
        right_branch_polyline=_straight_polyline(-3.0),
        front_object_distance=20.0,
        front_object_speed_mps=8.0,
        left_lane_gap=20.0,
        right_lane_gap=20.0,
        current_ref_lane_count=2,
        next_ref_lane_count=2,
        has_left_adjacent=True,
        has_right_adjacent=True,
        has_left_branch=True,
        has_right_branch=True,
    )
    base.update(overrides)
    return ModeContext(**base)


def test_geometric_feasibility_disables_left_lane_change_without_left_lane():
    ctx = _context(left_lane_polyline=None, has_left_adjacent=False, left_branch_polyline=None, has_left_branch=False)

    mask = GeometricFeasibilityChecker().evaluate(ctx)

    assert not mask[3]
    assert not mask[4]
    assert not mask[5]
    assert mask[6]
    assert mask[7]
    assert mask[8]


def test_traffic_feasibility_disables_left_lane_change_for_small_gap():
    ctx = _context(left_lane_gap=5.0)
    geometric_mask = GeometricFeasibilityChecker().evaluate(ctx)

    mask = TrafficFeasibilityChecker(lane_change_min_gap_m=12.0).evaluate(ctx, geometric_mask)

    assert not mask[3]
    assert not mask[4]
    assert not mask[5]
    assert mask[6]
    assert mask[7]
    assert mask[8]


def test_geometric_feasibility_allows_branch_only_lane_change_for_s7_merge_like_case():
    ctx = _context(left_lane_polyline=None, has_left_adjacent=False, left_branch_polyline=_straight_polyline(3.0), has_left_branch=True)

    mask = GeometricFeasibilityChecker().evaluate(ctx)

    assert mask[3]
    assert mask[4]
    assert mask[5]


def test_build_mode_valid_mask_disables_medium_keep_lane_without_front_vehicle():
    ctx = _context(front_object_distance=-1.0, front_object_speed_mps=0.0)

    result = build_mode_valid_mask(ctx)

    assert not result.valid_mask[1]
    assert result.valid_mask[0]
    assert result.valid_mask[9]
