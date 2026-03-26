from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from envs.platoon_env import PlatoonEnv, obs_to_tensor


def test_lidar_state_mode_regression():
    env = PlatoonEnv({"use_render": False, "observation_mode": "lidar_state"})
    try:
        obs = env.reset()
        assert set(obs.keys()) == {"agent0", "agent1", "agent2"}
        agent0 = obs["agent0"]
        assert "formation_relation_state" in agent0
        assert "obs" in agent0
        assert tuple(np.asarray(agent0["formation_relation_state"]).shape) == (12,)
    finally:
        env.close()


def test_multimodal_reset_contains_required_modalities():
    env = PlatoonEnv({"use_render": False, "observation_mode": "multimodal"})
    try:
        obs = env.reset()
        agent0 = obs["agent0"]
        assert set(agent0.keys()) == {"camera", "lidar", "status", "formation_relation_state"}
        assert tuple(np.asarray(agent0["camera"]).shape) == (3, 256, 1024)
        assert tuple(np.asarray(agent0["lidar"]).shape) == (1, 256, 256)
        assert tuple(np.asarray(agent0["status"]).shape) == (8,)
        assert tuple(np.asarray(agent0["formation_relation_state"]).shape) == (12,)
    finally:
        env.close()


def test_obs_to_tensor_returns_float32_and_supports_device():
    sample = {
        "camera": np.zeros((3, 256, 1024), dtype=np.float32),
        "lidar": np.zeros((1, 256, 256), dtype=np.float32),
        "status": np.zeros((8,), dtype=np.float32),
        "formation_relation_state": np.zeros((12,), dtype=np.float32),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_obs = obs_to_tensor(sample, device=device)
    assert set(tensor_obs.keys()) == set(sample.keys())
    for value in tensor_obs.values():
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float32
        assert value.device.type == device
