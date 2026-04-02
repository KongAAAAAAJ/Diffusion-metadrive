from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


class _StubConfig(dict):
    def update(self, other=None, *args, **kwargs):
        if other is not None:
            super().update(other)
        return self


def _load_base_multi_env_module():
    module_name = "base_multi_env_teleport_test"
    path = REPO_ROOT / "metadrive/envs/diffusion_envs/base_multi_env.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    multi_agent_module = types.ModuleType("metadrive.envs.marl_envs.multi_agent_metadrive")

    class _MultiAgentMetaDrive:
        @staticmethod
        def default_config():
            return _StubConfig()

        @property
        def action_space(self):
            return "physics_action_space"

    multi_agent_module.MultiAgentMetaDrive = _MultiAgentMetaDrive
    register("metadrive.envs.marl_envs.multi_agent_metadrive", multi_agent_module)

    hybrid_module = types.ModuleType("metadrive.envs.diffusion_envs.custom_hybrid_map")
    hybrid_module.MAHybridPGMapManager = object
    register("metadrive.envs.diffusion_envs.custom_hybrid_map", hybrid_module)

    state_obs_module = types.ModuleType("metadrive.obs.state_obs")
    state_obs_module.LidarStateObservation = object
    register("metadrive.obs.state_obs", state_obs_module)

    diff_obs_module = types.ModuleType("metadrive.obs.diff_obs.top_down_state_obs_multi_channel")
    diff_obs_module.TopDownLidarStateObservation = object
    diff_obs_module.DatasetCollectObservation = object
    register("metadrive.obs.diff_obs.top_down_state_obs_multi_channel", diff_obs_module)

    rgb_module = types.ModuleType("metadrive.component.sensors.rgb_camera")
    rgb_module.RGBCamera = object
    register("metadrive.component.sensors.rgb_camera", rgb_module)

    utils_module = types.ModuleType("metadrive.utils")
    utils_module.Config = _StubConfig
    utils_module.merge_dicts = lambda left, right, **kwargs: {**right, **left}
    register("metadrive.utils", utils_module)

    traffic_module = types.ModuleType("metadrive.manager.traffic_manager")
    traffic_module.TrafficMode = SimpleNamespace(Trigger="Trigger")
    register("metadrive.manager.traffic_manager", traffic_module)

    engine_utils_module = types.ModuleType("metadrive.engine.engine_utils")
    engine_utils_module.initialize_global_config = lambda config: None
    register("metadrive.engine.engine_utils", engine_utils_module)

    transfuser_module = types.ModuleType("metadrive.policy.diffusion_policy.transfuser_config")
    transfuser_module.build_transfuser_config = lambda size: {"size": size}
    transfuser_module.transfuser_config_to_dict = lambda config: dict(config)
    register("metadrive.policy.diffusion_policy.transfuser_config", transfuser_module)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in stubbed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


class _StubVehicle:
    def __init__(self, position=(0.0, 0.0)):
        self.position = np.asarray(position, dtype=np.float32)
        self.calls = []

    def teleport_to(self, position, heading_theta, velocity_2d=None):
        self.calls.append(
            {
                "position": tuple(position),
                "heading_theta": float(heading_theta),
                "velocity_2d": None if velocity_2d is None else tuple(velocity_2d),
            }
        )
        self.position = np.asarray(position, dtype=np.float32)


def test_teleport_action_space_uses_trajectory_box_for_multi_agent_env():
    module = _load_base_multi_env_module()
    env = module.BaseMultiEnv.__new__(module.BaseMultiEnv)
    env.config = {
        "control_mode": "teleport",
        "teleport_trajectory_steps": 8,
        "teleport_trajectory_dim": 3,
        "agent_configs": {"agent0": {}, "agent1": {}},
    }
    env.is_multi_agent = True
    env.agents = {}

    space = env.action_space

    assert set(space.spaces.keys()) == {"agent0", "agent1"}
    assert space["agent0"].shape == (8, 3)


def test_teleport_step_uses_selected_waypoint_heading_from_trajectory():
    module = _load_base_multi_env_module()
    env = module.BaseMultiEnv.__new__(module.BaseMultiEnv)
    vehicle = _StubVehicle()
    captured = {}
    env.config = {
        "teleport_waypoint_index": 1,
    }
    env.agents = {"agent0": vehicle}
    env.in_stop = False
    env._preprocess_actions = lambda actions: actions
    env._teleport_step_simulator = lambda low_level_actions: {"simulated": low_level_actions}
    env._get_step_return = lambda actions, engine_info: captured.update(
        {"actions": actions, "engine_info": engine_info}
    ) or ("obs", "reward", "terminated", "truncated", "info")

    trajectory = np.asarray([[1.0, 2.0, 0.1], [3.0, 4.0, 0.5]], dtype=np.float32)
    result = env._teleport_step({"agent0": trajectory})

    assert result == ("obs", "reward", "terminated", "truncated", "info")
    assert vehicle.calls[0]["position"] == (3.0, 4.0)
    assert np.isclose(vehicle.calls[0]["heading_theta"], 0.5)
    assert np.allclose(captured["actions"]["agent0"], trajectory)


def test_teleport_step_infers_heading_from_next_waypoint_when_heading_is_missing():
    module = _load_base_multi_env_module()
    env = module.BaseMultiEnv.__new__(module.BaseMultiEnv)
    vehicle = _StubVehicle(position=(1.0, 1.0))
    env.config = {
        "teleport_waypoint_index": 0,
    }
    env.agents = {"agent0": vehicle}
    env.in_stop = False
    env._preprocess_actions = lambda actions: actions
    env._teleport_step_simulator = lambda low_level_actions: {}
    env._get_step_return = lambda actions, engine_info: None

    trajectory = np.asarray([[2.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    env._teleport_step({"agent0": trajectory})

    expected_heading = np.arctan2(4.0, 3.0)
    assert np.isclose(vehicle.calls[0]["heading_theta"], expected_heading)


def test_teleport_step_simulator_runs_before_step_manager_steps_and_after_step():
    module = _load_base_multi_env_module()
    env = module.BaseMultiEnv.__new__(module.BaseMultiEnv)
    events = []
    manager = SimpleNamespace(step=lambda: events.append("manager.step"))
    actions = {"agent0": np.zeros(2, dtype=np.float32)}
    env.engine = SimpleNamespace(
        before_step=lambda current_actions: events.append(("before_step", tuple(current_actions.keys()))) or {"before": True},
        managers={"agent_manager": manager},
        step_physics_world=lambda: events.append("physics"),
        after_step=lambda: events.append("after_step") or {"after": True},
        task_manager=SimpleNamespace(step=lambda: events.append("task_manager.step")),
    )

    engine_info = env._teleport_step_simulator(actions)

    assert events == [
        ("before_step", ("agent0",)),
        "manager.step",
        "physics",
        "task_manager.step",
        "after_step",
    ]
    assert engine_info == {"after": True, "before": True}
