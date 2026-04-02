from __future__ import annotations

import numpy as np

from metadrive.exp_dataset.trajectory_filter import (
    OutOfRoadByReferenceLaneRule,
    TrajectoryFilterPipeline,
    TrajectoryFilterRule,
)


def _make_sample(
    future_ego_pose_world: np.ndarray,
    future_reference_pose_world: np.ndarray,
    future_lane_width: np.ndarray,
    future_reference_lane_index: np.ndarray,
):
    return {
        "future_ego_pose_world": np.asarray(future_ego_pose_world, dtype=np.float32),
        "future_reference_pose_world": np.asarray(future_reference_pose_world, dtype=np.float32),
        "future_lane_width": np.asarray(future_lane_width, dtype=np.float32),
        "future_reference_lane_index": np.asarray(future_reference_lane_index, dtype=np.int16),
    }


def test_out_of_road_rule_passes_when_points_within_lane_bounds():
    sample = _make_sample(
        future_ego_pose_world=[[1.0, 0.5, 0.0], [2.0, -0.6, 0.0]],
        future_reference_pose_world=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        future_lane_width=[4.0, 4.0],
        future_reference_lane_index=[0, 0],
    )

    result = OutOfRoadByReferenceLaneRule().check_sample(sample)

    assert result.passed is True
    assert result.rule_name == "out_of_road"
    assert result.reason == ""


def test_out_of_road_rule_rejects_when_point_exceeds_half_lane_width():
    sample = _make_sample(
        future_ego_pose_world=[[1.0, 2.1, 0.0], [2.0, 0.0, 0.0]],
        future_reference_pose_world=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        future_lane_width=[4.0, 4.0],
        future_reference_lane_index=[0, 0],
    )

    result = OutOfRoadByReferenceLaneRule().check_sample(sample)

    assert result.passed is False
    assert result.reason == "out_of_road"


def test_out_of_road_margin_ratio_increases_tolerance():
    sample = _make_sample(
        future_ego_pose_world=[[1.0, 2.1, 0.0]],
        future_reference_pose_world=[[1.0, 0.0, 0.0]],
        future_lane_width=[4.0],
        future_reference_lane_index=[0],
    )

    result = OutOfRoadByReferenceLaneRule(out_of_road_margin_ratio=0.1).check_sample(sample)

    assert result.passed is True


def test_out_of_road_rule_skips_points_with_missing_lane_when_configured():
    sample = _make_sample(
        future_ego_pose_world=[[1.0, 10.0, 0.0]],
        future_reference_pose_world=[[1.0, 0.0, 0.0]],
        future_lane_width=[4.0],
        future_reference_lane_index=[-1],
    )

    result = OutOfRoadByReferenceLaneRule(out_of_road_missing_lane_policy="skip_point").check_sample(sample)

    assert result.passed is True
    assert result.missing_reference_lane_points == 1


def test_pipeline_aggregates_rule_results():
    class AlwaysPassRule(TrajectoryFilterRule):
        @property
        def name(self) -> str:
            return "always_pass"

        def check_sample(self, sample):
            from metadrive.exp_dataset.trajectory_filter import FilterVerdict

            return FilterVerdict(passed=True, rule_name=self.name)

    sample = _make_sample(
        future_ego_pose_world=[[1.0, 2.1, 0.0]],
        future_reference_pose_world=[[1.0, 0.0, 0.0]],
        future_lane_width=[4.0],
        future_reference_lane_index=[0],
    )

    result = TrajectoryFilterPipeline(
        [AlwaysPassRule(), OutOfRoadByReferenceLaneRule()]
    ).check_sample(sample)

    assert result.passed is False
    assert result.rejection_reasons == ["out_of_road"]
