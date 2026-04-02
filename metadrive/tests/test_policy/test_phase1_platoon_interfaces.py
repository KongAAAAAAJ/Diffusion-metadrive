from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_platoon_module_with_torch_stub(module_name: str):
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.device = object
    torch_stub.as_tensor = lambda value, dtype=None: np.asarray(value, dtype=np.float32 if dtype is not None else None)
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    try:
        return _load_module(module_name, REPO_ROOT / "envs/platoon_env.py")
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch


def test_platoon_env_requires_real_metadrive_runtime():
    module = _load_module("phase1_platoon_env", REPO_ROOT / "envs/platoon_env.py")
    if module._METADRIVE_IMPORT_ERROR is not None:
        with pytest.raises(RuntimeError):
            module.PlatoonEnv()


def test_platoon_env_infers_trajectory_and_low_level_action_modes():
    module = _load_platoon_module_with_torch_stub("phase1_platoon_env_step")
    env = module.PlatoonEnv.__new__(module.PlatoonEnv)
    env.config = {"control_mode": "physics"}

    trajectory_actions = {
        "agent0": np.zeros((8, 3), dtype=np.float32),
        "agent1": np.zeros((8, 3), dtype=np.float32),
        "agent2": np.zeros((8, 3), dtype=np.float32),
    }
    assert env._infer_control_mode(trajectory_actions) == "trajectory"

    low_level_actions = {
        "agent0": np.zeros((2,), dtype=np.float32),
        "agent1": np.zeros((2,), dtype=np.float32),
        "agent2": np.zeros((2,), dtype=np.float32),
    }
    assert env._infer_control_mode(low_level_actions) == "low_level"


def test_platoon_env_teleport_mode_short_circuits_shape_inference():
    module = _load_platoon_module_with_torch_stub("phase1_platoon_env_teleport_mode")
    env = module.PlatoonEnv.__new__(module.PlatoonEnv)
    env.config = {"control_mode": "teleport"}

    teleport_actions = {
        "agent0": np.zeros((8, 3), dtype=np.float32),
        "agent1": np.zeros((8, 3), dtype=np.float32),
        "agent2": np.zeros((8, 3), dtype=np.float32),
    }
    assert env._infer_control_mode(teleport_actions) == "teleport"


def test_platoon_env_build_info_dict_handles_teleport_trajectory_actions():
    module = _load_platoon_module_with_torch_stub("phase1_platoon_env_info_teleport")
    env = module.PlatoonEnv.__new__(module.PlatoonEnv)
    env._agent_ids = ["agent0"]
    env.agents = {"agent0": object()}
    env._last_actions = {}
    env._last_info = {}
    env._compute_min_gap = lambda: 6.0
    env.get_formation_relation_state = lambda agent_id: np.zeros((12,), dtype=np.float32)
    env._compute_agent_formation_error = lambda agent_id: 0.5
    env._compute_progress = lambda agent_id: 1.25
    env._agent_speed_km_h = lambda agent_id: 20.0

    info = env._build_info_dict(
        "teleport",
        actions={"agent0": np.zeros((8, 3), dtype=np.float32)},
        base_info={"agent0": {}},
    )

    assert info["agent0"]["control_mode"] == "teleport"
    assert info["agent0"]["jerk"] == 0.0
    assert info["agent0"]["delta_steering"] == 0.0


def test_platoon_env_initial_spacing_and_target_gap_include_vehicle_length():
    module = _load_module("phase1_platoon_env_spacing", REPO_ROOT / "envs/platoon_env.py")
    env = module.PlatoonEnv.__new__(module.PlatoonEnv)
    env.platoon_config = module.PlatoonEnvConfig(initial_speed_km_h=25.0, headway_time_s=0.5, vehicle_length_m=4.5)
    env._agent_ids = ["agent0", "agent1", "agent2"]
    env.agents = {}

    expected_gap = 4.5 + 25.0 / 3.6 * 0.5
    assert np.isclose(env._desired_center_spacing_m(), expected_gap, atol=1e-5)

    poses = {
        "agent0": np.asarray([20.0, 0.0, 0.0], dtype=np.float32),
        "agent1": np.asarray([20.0 - expected_gap, 0.0, 0.0], dtype=np.float32),
        "agent2": np.asarray([20.0 - 2.0 * expected_gap, 0.0, 0.0], dtype=np.float32),
    }
    speeds = {agent_id: 25.0 for agent_id in env._agent_ids}
    env._agent_pose = lambda agent_id: poses[agent_id]
    env._agent_speed_km_h = lambda agent_id: speeds[agent_id]

    relation = env.get_formation_relation_state("agent0")
    assert relation[4] < 0.0
    assert np.isclose(abs(relation[4]), expected_gap, atol=1e-5)


def test_platoon_env_collision_on_one_vehicle_terminates_whole_platoon():
    module = _load_module("phase1_platoon_env_termination", REPO_ROOT / "envs/platoon_env.py")

    env = module.PlatoonEnv.__new__(module.PlatoonEnv)
    env._agent_ids = ["agent0", "agent1", "agent2"]
    terminated, truncated = env._enforce_platoon_episode_end(
        {
            "agent0": False,
            "agent1": False,
            "agent2": False,
            "__all__": False,
        },
        {
            "agent0": False,
            "agent1": False,
            "agent2": False,
            "__all__": False,
        },
        {
            "agent0": {"crash_vehicle": False},
            "agent1": {"crash_vehicle": True},
            "agent2": {"crash_vehicle": False},
        },
    )

    assert terminated["agent0"] is True
    assert terminated["agent1"] is True
    assert terminated["agent2"] is True
    assert terminated["__all__"] is True
    assert truncated["__all__"] is False


def test_platoon_env_out_of_road_on_one_vehicle_terminates_whole_platoon():
    module = _load_module("phase1_platoon_env_oor_termination", REPO_ROOT / "envs/platoon_env.py")

    env = module.PlatoonEnv.__new__(module.PlatoonEnv)
    env._agent_ids = ["agent0", "agent1", "agent2"]
    terminated, truncated = env._enforce_platoon_episode_end(
        {
            "agent0": False,
            "agent1": False,
            "agent2": False,
            "__all__": False,
        },
        {
            "agent0": False,
            "agent1": False,
            "agent2": False,
            "__all__": False,
        },
        {
            "agent0": {"out_of_road": False},
            "agent1": {"out_of_road": True},
            "agent2": {"out_of_road": False},
        },
    )

    assert terminated["agent0"] is True
    assert terminated["agent1"] is True
    assert terminated["agent2"] is True
    assert terminated["__all__"] is True
    assert truncated["__all__"] is False


def test_platoon_metrics_compute_expected_keys():
    module = _load_module("phase1_platoon_metrics", REPO_ROOT / "evaluation/platoon_metrics.py")

    metrics = module.PlatoonMetrics(formation_error_threshold=2.0)
    metrics.start_episode()
    metrics.update({
        "agent0": {"arrive_dest": True, "crash": False, "out_of_road": False, "formation_error": 1.0, "min_gap": 8.0},
        "agent1": {"arrive_dest": True, "crash": False, "out_of_road": False, "formation_error": 1.5, "min_gap": 7.0},
        "agent2": {"arrive_dest": True, "crash": False, "out_of_road": False, "formation_error": 1.2, "min_gap": 6.0},
    })
    metrics.end_episode()

    summary = metrics.compute()
    assert set(summary) >= {
        "success_rate",
        "collision_rate",
        "formation_error",
        "recovery_time",
        "min_inter_vehicle_gap",
    }


def test_hazard_scenarios_return_three_named_configs():
    module = _load_module("phase1_hazard_scenarios", REPO_ROOT / "scenarios/hazard_scenarios.py")

    configs = module.get_hazard_scenario_configs()
    names = {cfg["name"] for cfg in configs}
    assert {"static_obstacle_detour", "dynamic_cut_in", "bottleneck_narrow_bridge"} <= names


def test_verify_phase1_parse_args_supports_episodes_render_and_top_down():
    module = _load_module("phase1_verify_script", REPO_ROOT / "scripts/verify_phase1.py")
    args = module.parse_args(["--episodes", "5", "--render", "0", "--top-down", "1"])
    assert args.episodes == 5
    assert args.render == 0
    assert args.top_down == 1
