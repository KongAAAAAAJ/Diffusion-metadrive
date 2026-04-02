from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_custom_traffic_manager_module():
    module_name = "custom_traffic_manager_test"
    path = REPO_ROOT / "envs/traffic_manager.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    traffic_module = types.ModuleType("metadrive.manager.traffic_manager")

    class _PGTrafficManager:
        def __init__(self):
            self.engine = SimpleNamespace(global_config={})
            self._traffic_vehicles = []
            self.before_step_called = False
            self.before_reset_called = False

        def before_step(self):
            self.before_step_called = True
            return {"parent": True}

        def before_reset(self):
            self.before_reset_called = True

    traffic_module.PGTrafficManager = _PGTrafficManager
    register("metadrive.manager.traffic_manager", traffic_module)

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


def test_before_reset_reads_target_speed_from_global_config():
    module = _load_custom_traffic_manager_module()
    manager = module.CustomTrafficManager()
    manager.engine.global_config = {"traffic_target_speed": 25.0}

    manager.before_reset()

    assert manager.before_reset_called is True
    assert manager._custom_target_speed == 25.0


def test_before_step_caps_background_policy_target_speed_without_raising_lower_limits():
    module = _load_custom_traffic_manager_module()
    manager = module.CustomTrafficManager()
    manager._custom_target_speed = 25.0

    fast_vehicle = SimpleNamespace(name="fast")
    creep_vehicle = SimpleNamespace(name="creep")
    manager._traffic_vehicles = [fast_vehicle, creep_vehicle]

    policies = {
        "fast": SimpleNamespace(target_speed=30.0),
        "creep": SimpleNamespace(target_speed=5.0),
    }
    manager.engine = SimpleNamespace(
        get_policy=lambda name: policies[name],
        global_config={},
    )

    result = manager.before_step()

    assert result == {"parent": True}
    assert policies["fast"].target_speed == 25.0
    assert policies["creep"].target_speed == 5.0


def test_before_step_applies_cap_to_newly_added_respawn_vehicle():
    module = _load_custom_traffic_manager_module()
    manager = module.CustomTrafficManager()
    manager._custom_target_speed = 22.0

    new_vehicle = SimpleNamespace(name="respawned")
    policy = SimpleNamespace(target_speed=30.0)
    manager._traffic_vehicles = [new_vehicle]
    manager.engine = SimpleNamespace(
        get_policy=lambda name: policy,
        global_config={},
    )

    manager.before_step()

    assert policy.target_speed == 22.0


def test_before_step_caps_normal_speed_so_policy_act_cannot_restore_original_speed():
    module = _load_custom_traffic_manager_module()
    manager = module.CustomTrafficManager()
    manager._custom_target_speed = 24.0

    vehicle = SimpleNamespace(name="lane_change_vehicle")

    class _Policy:
        def __init__(self):
            self.NORMAL_SPEED = 30.0
            self.target_speed = 30.0

        def act(self):
            self.target_speed = self.NORMAL_SPEED
            return [0.0, 0.0]

    policy = _Policy()
    manager._traffic_vehicles = [vehicle]
    manager.engine = SimpleNamespace(
        get_policy=lambda name: policy,
        global_config={},
    )

    manager.before_step()
    policy.act()

    assert policy.NORMAL_SPEED == 24.0
    assert policy.target_speed == 24.0
