from __future__ import annotations

import time

import torch
from torch import nn

from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from metadrive.policy.diffusion_policy.transfuser_model_v2 import V2TransfuserModel
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.relation_encoder import RelationEncoder


def _dummy_batch(config, num_vehicles: int = 3):
    return {
        f"agent{i}": {
            "camera": torch.zeros((3, config.camera_height, config.camera_width), dtype=torch.float32),
            "lidar": torch.zeros((1, config.lidar_resolution_height, config.lidar_resolution_width), dtype=torch.float32),
            "status": torch.zeros((8,), dtype=torch.float32),
            "formation_relation_state": torch.zeros((12,), dtype=torch.float32),
        }
        for i in range(num_vehicles)
    }


def test_platoon_diffusion_planner_contract():
    config = build_transfuser_config("small", plan_anchor_path="metadrive/exp_dataset/metadrive_anchors_ppo.npy")
    planner = PlatoonDiffusionPlanner(config, num_vehicles=3).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = planner.to(device)

    assert isinstance(planner, nn.Module)
    assert isinstance(planner.model, V2TransfuserModel)
    assert isinstance(planner.relation_encoder, RelationEncoder)
    assert sum(isinstance(module, V2TransfuserModel) for module in planner.modules()) == 1

    batch = _dummy_batch(config, num_vehicles=3)
    batch = {
        agent_id: {key: value.to(device) for key, value in sample.items()}
        for agent_id, sample in batch.items()
    }
    with torch.no_grad():
        planner(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = planner(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == {"agent0", "agent1", "agent2"}
    for value in outputs.values():
        assert isinstance(value, torch.Tensor)
        assert tuple(value.shape) == (8, 3)
    assert latency_ms < 100.0, latency_ms
