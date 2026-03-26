from __future__ import annotations

import numpy as np
import pytest
import torch

from train.train_platoon_rl import ToyEnv


class DummyObs:
    def observe(self, vehicle):
        return {"obs": np.asarray([vehicle.position[0], vehicle.position[1]], dtype=np.float32)}


class DummyVehicle:
    def __init__(self, x: float, y: float, heading: float, velocity=(0.0, 0.0)):
        self.position = np.asarray([x, y], dtype=np.float64)
        self.heading_theta = float(heading)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.steering = 0.0
        self.speed_km_h = float(np.linalg.norm(self.velocity) * 3.6)
        self.lane = None

    def set_position(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def set_heading_theta(self, heading):
        self.heading_theta = float(heading)

    def set_velocity(self, velocity):
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.speed_km_h = float(np.linalg.norm(self.velocity) * 3.6)


class PandaOrigin:
    def __init__(self):
        self.pos_calls = []
        self.h_calls = []
        self._pos = [0.0, 0.0, 0.0]

    def setPos(self, x, y, z):
        self.pos_calls.append((x, y, z))
        self._pos = [x, y, z]

    def setH(self, h):
        self.h_calls.append(h)

    def getPos(self):
        return self._pos


class PandaNode:
    def __init__(self):
        self.velocity_calls = []

    def setLinearVelocity(self, vec):
        self.velocity_calls.append(vec)


class PandaChassis:
    def __init__(self):
        self._node = PandaNode()

    def node(self):
        return self._node


class PandaVehicle:
    def __init__(self):
        self.position = np.asarray([1.0, 2.0], dtype=np.float64)
        self.heading_theta = 0.0
        self.velocity = np.asarray([0.0, 0.0], dtype=np.float64)
        self.origin = PandaOrigin()
        self.chassis = PandaChassis()
        self.steering = 0.0
        self.lane = None


class BrokenVehicle:
    def __init__(self):
        self.position = np.asarray([0.0, 0.0], dtype=np.float64)
        self.heading_theta = 0.0
        self.velocity = np.asarray([0.0, 0.0], dtype=np.float64)
        self.lane = None


class MinimalStateEnv:
    def __init__(self, vehicles=None):
        self._agent_ids = ["agent0", "agent1"]
        self.agents = vehicles or {
            "agent0": DummyVehicle(1.0, 2.0, 0.1, (3.0, 0.0)),
            "agent1": DummyVehicle(4.0, 5.0, -0.2, (2.0, 0.0)),
        }
        self._last_actions = {agent_id: np.zeros((2,), dtype=np.float32) for agent_id in self._agent_ids}
        self._last_progress_refs = {
            agent_id: (None, 0.0, np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float32).copy())
            for agent_id, vehicle in self.agents.items()
        }
        self._last_info = {agent_id: {"raw_obs": {"obs": np.zeros((1,), dtype=np.float32)}} for agent_id in self._agent_ids}
        self.observations = {agent_id: DummyObs() for agent_id in self._agent_ids}

    def _format_agent_observation(self, agent_id: str, agent_obs: object) -> dict[str, np.ndarray]:
        obs = dict(agent_obs)
        obs["formation_relation_state"] = np.full((12,), float(len(agent_id)), dtype=np.float32)
        return obs

    def _augment_observations(self, obs):
        return {agent_id: self._format_agent_observation(agent_id, agent_obs) for agent_id, agent_obs in obs.items()}


def _bind_env_state_methods(env: MinimalStateEnv):
    from envs.platoon_env import PlatoonEnv

    env.get_state = PlatoonEnv.get_state.__get__(env, MinimalStateEnv)
    env.set_state = PlatoonEnv.set_state.__get__(env, MinimalStateEnv)
    env.get_current_obs = PlatoonEnv.get_current_obs.__get__(env, MinimalStateEnv)
    return env


def test_platoon_env_state_round_trip_and_observation_refresh():
    env = _bind_env_state_methods(MinimalStateEnv())
    state = env.get_state()
    assert "vehicle_states" in state
    assert set(state["vehicle_states"].keys()) == {"agent0", "agent1"}

    env.agents["agent0"].set_position([9.0, 9.0])
    env.agents["agent0"].set_heading_theta(1.2)
    env.agents["agent0"].set_velocity([0.5, 0.5])
    env.set_state(state)
    restored = env.get_state()

    np.testing.assert_allclose(restored["vehicle_states"]["agent0"]["position"], state["vehicle_states"]["agent0"]["position"], atol=1e-2)
    obs = env.get_current_obs()
    assert set(obs.keys()) == {"agent0", "agent1"}
    assert all(isinstance(value, np.ndarray) for value in obs["agent0"].values())


def test_panda3d_fallback_path_is_used_when_standard_vehicle_api_missing():
    env = _bind_env_state_methods(MinimalStateEnv({"agent0": PandaVehicle(), "agent1": PandaVehicle()}))
    state = env.get_state()
    env.set_state(state)
    agent0 = env.agents["agent0"]
    assert agent0.origin.pos_calls
    assert agent0.origin.h_calls
    assert agent0.chassis.node().velocity_calls


def test_missing_restore_api_raises_clear_runtime_error():
    env = _bind_env_state_methods(MinimalStateEnv({"agent0": BrokenVehicle(), "agent1": BrokenVehicle()}))
    state = env.get_state()
    with pytest.raises(RuntimeError) as exc_info:
        env.set_state(state)
    message = str(exc_info.value)
    assert "agent0" in message
    assert "position" in message or "heading" in message or "velocity" in message


def test_toy_env_state_stubs_exist_and_are_callable():
    env = ToyEnv(num_agents=2, mode="toy-single")
    state = env.get_state()
    assert isinstance(state, dict)
    env.set_state(state)
    obs = env.get_current_obs()
    assert set(obs.keys()) == {"agent0", "agent1"}
    assert torch.is_tensor(obs["agent0"]["status"])
