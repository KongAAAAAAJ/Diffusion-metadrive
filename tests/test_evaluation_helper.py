from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import evaluation_helper as helper


def test_evaluation_helper_aggregates_pdms_records_with_team_mean() -> None:
    records = [
        {
            "agent0": {"reward": 1.0, "progress": 2.0},
            "agent1": {"reward": 3.0, "progress": 4.0},
        },
        {
            "agent0": {"reward": 5.0, "progress": 6.0},
            "agent1": {"reward": 7.0, "progress": 8.0},
        },
    ]

    aggregated = helper._aggregate_pdms_records(records)

    assert aggregated["agent0"] == {"reward": 3.0, "progress": 4.0}
    assert aggregated["agent1"] == {"reward": 5.0, "progress": 6.0}
    assert aggregated["__team__"] == {"reward": 4.0, "progress": 5.0}


class _FakeLane:
    def local_coordinates(self, position):
        return float(position[0]) * 2.0, float(position[1])


def test_collect_episode_step_record_records_lane_s_from_vehicle_lane() -> None:
    vehicle = SimpleNamespace(
        position=np.asarray([12.5, 1.0], dtype=np.float32),
        velocity=np.asarray([10.0, 0.0], dtype=np.float32),
        speed_km_h=36.0,
        heading_theta=0.2,
        lane=_FakeLane(),
    )
    env = SimpleNamespace(agents={"agent0": vehicle}, _preview_control_debug={})

    record = helper._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=0,
        actions={"agent0": np.asarray([0.1, 0.2], dtype=np.float32)},
        info={},
        pdms={},
        planning_debug={},
        previous_speed_mps={},
        dt=0.5,
    )

    assert record["vehicles"]["agent0"]["lane_s"] == pytest.approx(25.0)


def test_collect_episode_step_record_acceleration_uses_vehicle_velocity_and_step_dt() -> None:
    vehicle = SimpleNamespace(
        position=np.asarray([0.0, 0.0], dtype=np.float32),
        velocity=np.asarray([10.0, 0.0], dtype=np.float32),
        speed_km_h=36.0,
        heading_theta=0.0,
        lane=None,
    )
    env = SimpleNamespace(agents={"agent0": vehicle}, _preview_control_debug={})
    previous_velocity_mps: dict[str, np.ndarray] = {}

    first = helper._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=0,
        actions={"agent0": np.asarray([0.0, 0.0], dtype=np.float32)},
        info={},
        pdms={},
        planning_debug={},
        previous_speed_mps=previous_velocity_mps,
        dt=0.1,
    )
    vehicle.velocity = np.asarray([11.0, 0.0], dtype=np.float32)
    second = helper._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=1,
        actions={"agent0": np.asarray([0.0, 0.0], dtype=np.float32)},
        info={},
        pdms={},
        planning_debug={},
        previous_speed_mps=previous_velocity_mps,
        dt=0.1,
    )

    assert first["vehicles"]["agent0"]["accel_mps2"] is None
    assert second["vehicles"]["agent0"]["accel_mps2"] == pytest.approx(10.0)
    assert second["vehicles"]["agent0"]["accel_x_mps2"] == pytest.approx(10.0)
    assert second["vehicles"]["agent0"]["accel_y_mps2"] == pytest.approx(0.0)
    assert second["vehicles"]["agent0"]["accel_norm_mps2"] == pytest.approx(10.0)


def test_collect_episode_step_record_projects_acceleration_onto_vehicle_heading() -> None:
    vehicle = SimpleNamespace(
        position=np.asarray([0.0, 0.0], dtype=np.float32),
        velocity=np.asarray([10.0, 0.0], dtype=np.float32),
        speed_km_h=36.0,
        heading_theta=np.pi,
        lane=None,
    )
    env = SimpleNamespace(agents={"agent0": vehicle}, _preview_control_debug={})
    previous_velocity_mps: dict[str, np.ndarray] = {}

    helper._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=0,
        actions={"agent0": np.asarray([0.0, 0.0], dtype=np.float32)},
        info={},
        pdms={},
        planning_debug={},
        previous_speed_mps=previous_velocity_mps,
        dt=0.1,
    )
    vehicle.velocity = np.asarray([9.0, 0.0], dtype=np.float32)
    record = helper._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=1,
        actions={"agent0": np.asarray([0.0, 0.0], dtype=np.float32)},
        info={},
        pdms={},
        planning_debug={},
        previous_speed_mps=previous_velocity_mps,
        dt=0.1,
    )

    assert record["vehicles"]["agent0"]["accel_mps2"] == pytest.approx(10.0)


def test_collect_episode_step_record_handles_missing_velocity_without_acceleration() -> None:
    vehicle = SimpleNamespace(
        position=np.asarray([0.0, 0.0], dtype=np.float32),
        speed_km_h=36.0,
        heading_theta=0.0,
        lane=None,
    )
    env = SimpleNamespace(agents={"agent0": vehicle}, _preview_control_debug={})

    record = helper._collect_episode_step_record(
        env=env,
        agent_ids=["agent0"],
        step_idx=0,
        actions={"agent0": np.asarray([0.0, 0.0], dtype=np.float32)},
        info={},
        pdms={},
        planning_debug={},
        previous_speed_mps={},
        dt=0.1,
    )

    assert record["vehicles"]["agent0"]["accel_mps2"] is None


def test_following_distance_series_uses_agent_order_and_lane_s_difference() -> None:
    step_records = [
        {
            "vehicles": {
                "agent0": {"x": 0.0, "lane_s": 100.0},
                "agent1": {"x": 500.0, "lane_s": 88.0},
                "agent2": {"x": -500.0, "lane_s": 70.0},
            }
        },
        {
            "vehicles": {
                "agent0": {"x": 0.0, "lane_s": 104.0},
                "agent1": {"x": 500.0, "lane_s": None},
                "agent2": {"x": -500.0, "lane_s": 75.0},
            }
        },
    ]

    pairs = helper._following_distance_series(step_records, ["agent0", "agent1", "agent2"])

    assert pairs[0][0] == "agent0-agent1"
    np.testing.assert_allclose(pairs[0][1], np.asarray([12.0, np.nan], dtype=np.float32), equal_nan=True)
    assert pairs[1][0] == "agent1-agent2"
    np.testing.assert_allclose(pairs[1][1], np.asarray([18.0, np.nan], dtype=np.float32), equal_nan=True)
