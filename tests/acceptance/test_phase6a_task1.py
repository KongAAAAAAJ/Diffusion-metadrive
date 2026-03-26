from __future__ import annotations

import numpy as np
import pytest

try:
    from envs.platoon_env import PlatoonEnv

    METADRIVE_AVAILABLE = True
except Exception:
    METADRIVE_AVAILABLE = False

from train.train_platoon_rl import ToyEnv


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_get_state_contains_traffic_states():
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        state = env.get_state()
        assert "traffic_states" in state
        assert isinstance(state["traffic_states"], dict)
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_traffic_state_format():
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        state = env.get_state()
        for vehicle_name, vehicle_state in state["traffic_states"].items():
            assert isinstance(vehicle_name, str)
            assert "position" in vehicle_state
            assert "heading" in vehicle_state
            assert "velocity" in vehicle_state
            pos = np.asarray(vehicle_state["position"])
            vel = np.asarray(vehicle_state["velocity"])
            assert pos.ndim == 1
            assert pos.shape[0] >= 2
            assert vel.ndim == 1
            assert vel.shape[0] >= 2
            assert isinstance(vehicle_state["heading"], float)
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_set_state_restores_traffic_positions():
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        saved = env.get_state()
        dummy_action = {f"agent{i}": np.zeros((8, 3), dtype=np.float32) for i in range(3)}
        for _ in range(3):
            env.step(dummy_action)
        env.set_state(saved)
        restored = env.get_state()
        for vehicle_name in saved["traffic_states"]:
            if vehicle_name not in restored["traffic_states"]:
                continue
            pos_saved = np.asarray(saved["traffic_states"][vehicle_name]["position"], dtype=np.float64)[:2]
            pos_restored = np.asarray(restored["traffic_states"][vehicle_name]["position"], dtype=np.float64)[:2]
            assert np.linalg.norm(pos_saved - pos_restored) < 1.0
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_agent_state_still_works():
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1})
    try:
        env.reset()
        saved = env.get_state()
        assert "vehicle_states" in saved
        assert len(saved["vehicle_states"]) == 3
        dummy_action = {f"agent{i}": np.zeros((8, 3), dtype=np.float32) for i in range(3)}
        env.step(dummy_action)
        env.set_state(saved)
        restored = env.get_state()
        for agent_id in saved["vehicle_states"]:
            pos_saved = np.asarray(saved["vehicle_states"][agent_id]["position"], dtype=np.float64)[:2]
            pos_restored = np.asarray(restored["vehicle_states"][agent_id]["position"], dtype=np.float64)[:2]
            assert np.linalg.norm(pos_saved - pos_restored) < 0.5
    finally:
        env.close()


def test_toy_env_compatible():
    env = ToyEnv(num_agents=2, mode="platoon")
    state = env.get_state()
    env.set_state(state)
