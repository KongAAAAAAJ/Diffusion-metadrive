from __future__ import annotations

from pathlib import Path

import torch

from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon


CKPT_PATH = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt")


def _load_single_state_dict():
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {key.removeprefix("_transfuser_model."): value for key, value in state_dict.items() if key.startswith("_transfuser_model.")}


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


def test_weight_migration_contract():
    assert CKPT_PATH.exists(), CKPT_PATH
    config = build_transfuser_config("small", plan_anchor_path="metadrive/exp_dataset/metadrive_anchors_ppo.npy")
    planner = PlatoonDiffusionPlanner(config, num_vehicles=3).eval()
    planner = migrate_single_to_platoon(str(CKPT_PATH), planner)

    single_state = _load_single_state_dict()
    planner_state = planner.model.state_dict()

    assert "_status_encoding.weight" in planner_state
    for key, value in single_state.items():
        if key == "_status_encoding.weight":
            continue
        if key in planner_state and tuple(planner_state[key].shape) == tuple(value.shape):
            assert torch.allclose(planner_state[key], value, atol=1e-6), key

    assert torch.allclose(planner_state["_status_encoding.weight"][:, :8], single_state["_status_encoding.weight"], atol=1e-6)
    assert torch.count_nonzero(planner_state["_status_encoding.weight"][:, 8:]) > 0
    relation_nonzero = sum(int(torch.count_nonzero(param).item()) for param in planner.relation_encoder.parameters())
    assert relation_nonzero > 0

    with torch.no_grad():
        outputs = planner(_dummy_batch(config))
    assert set(outputs.keys()) == {"agent0", "agent1", "agent2"}

