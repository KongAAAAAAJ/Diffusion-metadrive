from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from envs.selector_platoon_env import _build_default_candidate_generator, _to_torch_batch
from train.train_selector import estimate_env_steps_per_iteration, resolve_max_iterations


def test_estimate_env_steps_uses_train_batch_size():
    cfg = {
        "train_batch_size": 4000,
        "rollout_fragment_length": 500,
        "num_rollout_workers": 4,
        "num_envs_per_worker": 1,
    }
    assert estimate_env_steps_per_iteration(cfg) == 4000


def test_estimate_env_steps_fragment_dominant():
    cfg = {
        "train_batch_size": 1000,
        "rollout_fragment_length": 500,
        "num_rollout_workers": 4,
        "num_envs_per_worker": 1,
    }
    assert estimate_env_steps_per_iteration(cfg) == 2000


def test_smoke_config_estimate_uses_train_batch_size():
    cfg = yaml.safe_load(Path("configs/train/platoon_mappo_smoke.yaml").read_text(encoding="utf-8"))
    assert estimate_env_steps_per_iteration(cfg) == 64


def test_resolve_max_iterations_with_fixed_estimate():
    cfg = {
        "total_env_steps": 20000,
        "train_batch_size": 4000,
        "rollout_fragment_length": 500,
        "num_rollout_workers": 4,
        "num_envs_per_worker": 1,
    }
    assert resolve_max_iterations(cfg) == 5


def test_to_torch_batch_respects_device():
    batch = {"x": np.zeros((2, 3), dtype=np.float32)}
    result = _to_torch_batch(batch, device=torch.device("cpu"))
    assert result["x"].device.type == "cpu"


def test_build_default_candidate_generator_falls_back_to_cpu(monkeypatch):
    class FakePlanner:
        def __init__(self, config=None, num_vehicles=0):
            self.config = config
            self.num_vehicles = num_vehicles
            self.moved_to = None

        def freeze_for_selector(self):
            return None

        def to(self, device):
            self.moved_to = torch.device(device)
            return self

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "metadrive.policy.diffusion_policy.transfuser_config.build_transfuser_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "models.platoon.platoon_diffusion_planner.PlatoonDiffusionPlanner",
        FakePlanner,
    )
    monkeypatch.setattr(
        "models.platoon.weight_migration.migrate_single_to_platoon",
        lambda ckpt_path, planner: planner,
    )

    planner = _build_default_candidate_generator(
        {
            "pretrained_ckpt": "/tmp/fake.ckpt",
            "anchor_path": "metadrive/exp_dataset/anchors.npy",
            "planner_device": "cuda",
            "num_agents": 3,
        }
    )

    assert isinstance(planner, FakePlanner)
    assert planner.moved_to.type == "cpu"


def test_phase6_configs_request_cuda_planner():
    main_cfg = yaml.safe_load(Path("configs/train/selector.yaml").read_text(encoding="utf-8"))
    smoke_cfg = yaml.safe_load(Path("configs/train/platoon_mappo_smoke.yaml").read_text(encoding="utf-8"))

    assert main_cfg["planner_device"] == "cuda"
    assert smoke_cfg["planner_device"] == "cuda"
    assert main_cfg["num_gpus"] == pytest.approx(0.5)
