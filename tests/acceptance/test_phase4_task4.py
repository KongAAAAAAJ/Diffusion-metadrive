from __future__ import annotations

import time
from pathlib import Path

import torch

from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon


CKPT_PATH = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt")


def _dummy_batch(config):
    return {
        f"agent{i}": {
            "camera": torch.zeros((3, config.camera_height, config.camera_width), dtype=torch.float32),
            "lidar": torch.zeros((1, config.lidar_resolution_height, config.lidar_resolution_width), dtype=torch.float32),
            "status": torch.zeros((8,), dtype=torch.float32),
            "formation_relation_state": torch.zeros((12,), dtype=torch.float32),
        }
        for i in range(3)
    }


def test_phase4_end_to_end_integration():
    assert CKPT_PATH.exists(), CKPT_PATH
    config = build_transfuser_config("small", plan_anchor_path="metadrive/exp_dataset/metadrive_anchors_ppo.npy")
    planner = PlatoonDiffusionPlanner(config, num_vehicles=3).eval()
    planner = migrate_single_to_platoon(str(CKPT_PATH), planner)
    batch = _dummy_batch(config)

    if torch.cuda.is_available():
        planner = planner.cuda()
        batch = {
            agent_id: {key: value.cuda() for key, value in sample.items()}
            for agent_id, sample in batch.items()
        }
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        planner(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = planner(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

    gpu_peak_gb = 0.0
    if torch.cuda.is_available():
        gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    assert set(outputs.keys()) == {"agent0", "agent1", "agent2"}
    stacked = torch.stack([outputs[agent_id].detach().cpu() for agent_id in sorted(outputs.keys())], dim=0)
    assert tuple(stacked.shape) == (3, 8, 3)
    assert latency_ms < 50.0, latency_ms
    assert gpu_peak_gb < 6.0, gpu_peak_gb
    assert not torch.isnan(stacked).any()
    assert not torch.isinf(stacked).any()
    assert float(stacked[..., 0].min()) >= -10.0
    assert float(stacked[..., 0].max()) <= 100.0
    assert float(stacked[..., 1].min()) >= -30.0
    assert float(stacked[..., 1].max()) <= 30.0

    with torch.no_grad():
        repeat_outputs = [planner(batch) for _ in range(3)]
    base = torch.stack([repeat_outputs[0][agent_id].detach().cpu() for agent_id in sorted(repeat_outputs[0].keys())], dim=0)
    for repeated in repeat_outputs[1:]:
        cur = torch.stack([repeated[agent_id].detach().cpu() for agent_id in sorted(repeated.keys())], dim=0)
        assert torch.allclose(base, cur, atol=1e-6)
