from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.platoon_env import PlatoonEnv


class SurrogateHarness:
    evaluate_trajectory_group = PlatoonEnv.evaluate_trajectory_group
    _agent_pose = PlatoonEnv._agent_pose
    _agent_speed_km_h = PlatoonEnv._agent_speed_km_h
    _agent_velocity_ms = PlatoonEnv._agent_velocity_ms
    _vehicle_length_m = PlatoonEnv._vehicle_length_m
    _desired_center_spacing_m = PlatoonEnv._desired_center_spacing_m
    get_formation_relation_state = PlatoonEnv.get_formation_relation_state
    _surrogate_lateral_offset = PlatoonEnv._surrogate_lateral_offset
    _surrogate_min_gap = PlatoonEnv._surrogate_min_gap
    _surrogate_formation_error = PlatoonEnv._surrogate_formation_error


def _load_anchor_trajectories() -> np.ndarray:
    anchors = np.load("expert_dataset/metadrive_anchors.npy").astype(np.float32)
    headings = np.zeros((anchors.shape[0], anchors.shape[1], 1), dtype=np.float32)
    return np.concatenate([anchors, headings], axis=-1)


def _make_vehicle(x: float, speed_km_h: float = 25.0) -> SimpleNamespace:
    return SimpleNamespace(
        position=np.asarray([x, 0.0], dtype=np.float32),
        heading_theta=0.0,
        speed_km_h=float(speed_km_h),
        velocity=np.asarray([speed_km_h / 3.6, 0.0], dtype=np.float32),
        LENGTH=5.74,
        lane=None,
    )


def _build_stub_env() -> SurrogateHarness:
    env = SurrogateHarness()
    env._agent_ids = ["agent0", "agent1", "agent2"]
    env.platoon_config = SimpleNamespace(
        initial_speed_km_h=25.0,
        headway_time_s=0.5,
        vehicle_length_m=5.74,
        trajectory_dt=0.5,
    )
    env.agents = {
        "agent0": _make_vehicle(0.0),
        "agent1": _make_vehicle(9.21),
        "agent2": _make_vehicle(18.42),
    }
    env._road_half_width = lambda agent_id: 100.0
    return env


def test_motion_prediction_eliminates_false_crashes():
    env = _build_stub_env()
    result = env.evaluate_trajectory_group("agent0", _load_anchor_trajectories())
    assert sum(bool(flag) for flag in result["crash_flags"]) == 0


def test_slow_anchor_correctly_crashes_middle_agent():
    env = _build_stub_env()
    trajectories = _load_anchor_trajectories()[0:1]
    result = env.evaluate_trajectory_group("agent1", trajectories)
    assert result["crash_flags"] == [True]


def test_lead_agent_no_crash_on_straight_anchors():
    env = _build_stub_env()
    trajectories = _load_anchor_trajectories()[[4, 7]]
    result = env.evaluate_trajectory_group("agent2", trajectories)
    assert result["crash_flags"] == [False, False]


def test_surrogate_does_not_mutate_env_state():
    env = _build_stub_env()
    before = env.get_formation_relation_state("agent0").copy()
    env.evaluate_trajectory_group("agent0", _load_anchor_trajectories())
    after = env.get_formation_relation_state("agent0")
    assert np.allclose(before, after)


def test_velocity_fallback():
    env = _build_stub_env()
    env.agents["agent0"] = SimpleNamespace(
        position=np.asarray([0.0, 0.0], dtype=np.float32),
        heading_theta=np.pi / 2,
        speed_km_h=36.0,
        LENGTH=5.74,
        lane=None,
    )
    velocity = env._agent_velocity_ms("agent0")
    assert np.allclose(velocity, np.asarray([0.0, 10.0], dtype=np.float32), atol=1e-4)


def test_collision_rate_improvement():
    env = _build_stub_env()
    trajectories = _load_anchor_trajectories()
    crash_count = 0
    total = 0
    for agent_id in ("agent0", "agent1", "agent2"):
        result = env.evaluate_trajectory_group(agent_id, trajectories)
        crash_count += sum(bool(flag) for flag in result["crash_flags"])
        total += len(result["crash_flags"])
    assert crash_count / total < 0.30
