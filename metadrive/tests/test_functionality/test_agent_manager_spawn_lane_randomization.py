from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_agent_manager_module():
    module_name = "agent_manager_spawn_lane_randomization_test"
    path = REPO_ROOT / "metadrive/manager/agent_manager.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    constants_module = types.ModuleType("metadrive.constants")
    constants_module.DEFAULT_AGENT = "agent0"
    register("metadrive.constants", constants_module)

    logger_module = types.ModuleType("metadrive.engine.logger")
    logger_module.get_logger = lambda: SimpleNamespace()
    register("metadrive.engine.logger", logger_module)

    base_manager_module = types.ModuleType("metadrive.manager.base_manager")

    class _BaseAgentManager:
        def __init__(self, init_observations=None):
            self.engine = None
            self.np_random = np.random.RandomState(0)

        def before_reset(self):
            return None

        def reset(self):
            return None

        def after_reset(self):
            return None

    base_manager_module.BaseAgentManager = _BaseAgentManager
    register("metadrive.manager.base_manager", base_manager_module)

    ai_module = types.ModuleType("metadrive.policy.AI_protect_policy")
    ai_module.AIProtectPolicy = type("AIProtectPolicy", (), {})
    register("metadrive.policy.AI_protect_policy", ai_module)

    idm_module = types.ModuleType("metadrive.policy.idm_policy")
    idm_module.TrajectoryIDMPolicy = type("TrajectoryIDMPolicy", (), {})
    register("metadrive.policy.idm_policy", idm_module)

    manual_module = types.ModuleType("metadrive.policy.manual_control_policy")
    manual_module.ManualControlPolicy = type("ManualControlPolicy", (), {})
    manual_module.TakeoverPolicy = type("TakeoverPolicy", (), {})
    manual_module.TakeoverPolicyWithoutBrake = type("TakeoverPolicyWithoutBrake", (), {})
    register("metadrive.policy.manual_control_policy", manual_module)

    replay_module = types.ModuleType("metadrive.policy.replay_policy")
    replay_module.ReplayTrafficParticipantPolicy = type("ReplayTrafficParticipantPolicy", (), {})
    register("metadrive.policy.replay_policy", replay_module)

    scenario_module = types.ModuleType("metadrive.exp_dataset.scenario_definitions")
    scenario_module.SCENARIO_BY_ID = {}
    register("metadrive.exp_dataset.scenario_definitions", scenario_module)

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


def test_random_spawn_lane_in_single_agent_uses_actual_spawn_road_lane_count():
    module = _load_agent_manager_module()
    manager = module.VehicleAgentManager.__new__(module.VehicleAgentManager)
    manager.np_random = np.random.RandomState(123)
    manager.engine = SimpleNamespace(
        global_config={
            "is_multi_agent": False,
            "random_spawn_lane_index": True,
            "agent_configs": {"agent0": {"spawn_lane_index": ("A", "B", 0)}},
        },
        current_map=SimpleNamespace(
            config={"lane_num": 3},
            road_network=SimpleNamespace(graph={"A": {"B": [object()]}}),
        ),
    )

    manager.random_spawn_lane_in_single_agent()

    assert manager.engine.global_config["agent_configs"]["agent0"]["spawn_lane_index"] == ("A", "B", 0)


def test_random_spawn_lane_in_single_agent_still_randomizes_on_multilane_roads():
    module = _load_agent_manager_module()
    manager = module.VehicleAgentManager.__new__(module.VehicleAgentManager)
    manager.np_random = np.random.RandomState(1)
    manager.engine = SimpleNamespace(
        global_config={
            "is_multi_agent": False,
            "random_spawn_lane_index": True,
            "agent_configs": {"agent0": {"spawn_lane_index": ("A", "B", 0)}},
        },
        current_map=SimpleNamespace(
            config={"lane_num": 3},
            road_network=SimpleNamespace(graph={"A": {"B": [object(), object(), object()]}}),
        ),
    )

    manager.random_spawn_lane_in_single_agent()

    lane_index = manager.engine.global_config["agent_configs"]["agent0"]["spawn_lane_index"]
    assert lane_index[:2] == ("A", "B")
    assert lane_index[2] in {0, 1, 2}


def test_random_spawn_lane_in_single_agent_respects_scenario_spawn_lane_configuration():
    module = _load_agent_manager_module()
    scenario_module = types.ModuleType("metadrive.exp_dataset.scenario_definitions")
    scenario_module.SCENARIO_BY_ID = {
        "S9_narrow_channel_negotiation": SimpleNamespace(
            ego_spawn_lane_preference=None,
            ego_spawn_lane_probabilities={"rightmost": 0.4, "middle": 0.4, "leftmost": 0.2},
        )
    }
    previous = sys.modules.get("metadrive.exp_dataset.scenario_definitions")
    sys.modules["metadrive.exp_dataset.scenario_definitions"] = scenario_module
    manager = module.VehicleAgentManager.__new__(module.VehicleAgentManager)
    manager.np_random = np.random.RandomState(1)
    manager.engine = SimpleNamespace(
        global_config={
            "is_multi_agent": False,
            "random_spawn_lane_index": True,
            "scenario_id": "S9_narrow_channel_negotiation",
            "agent_configs": {"agent0": {"spawn_lane_index": ("A", "B", 2)}},
        },
        current_map=SimpleNamespace(
            config={"lane_num": 3},
            road_network=SimpleNamespace(graph={"A": {"B": [object(), object(), object()]}}),
        ),
    )
    try:
        manager.random_spawn_lane_in_single_agent()
        assert manager.engine.global_config["agent_configs"]["agent0"]["spawn_lane_index"] == ("A", "B", 2)
    finally:
        if previous is None:
            sys.modules.pop("metadrive.exp_dataset.scenario_definitions", None)
        else:
            sys.modules["metadrive.exp_dataset.scenario_definitions"] = previous
