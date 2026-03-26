from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

import train.train_platoon_rl as entry
from evaluation.reward_terms import compute_team_reward
from train.ma_grpo_trainer import MultiAgentGRPOTrainer


def test_lr_bidirectional_adaptation():
    param = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([param], lr=5e-5)
    config = {"kl_target": 8.0, "lr": 5e-5, "lr_decay_factor": 0.8, "lr_grow_factor": 1.05}

    for _ in range(10):
        lr = entry._adapt_learning_rate(optimizer, {"kl": 15.0}, config)
    decayed_lr = lr
    assert decayed_lr < 5e-5
    assert 1e-6 <= decayed_lr <= 5e-5

    for _ in range(50):
        lr = entry._adapt_learning_rate(optimizer, {"kl": 2.0}, config)
    assert decayed_lr < lr <= 5e-5
    assert abs(lr - 5e-5) / 5e-5 < 0.2


def test_bn_frozen_in_backbone():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model._backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
            self.model._tf_decoder = nn.Linear(4, 4)
            self.head_bn = nn.BatchNorm1d(4)

        def forward(self, x):
            x = self.model._backbone(x)
            x = x.mean(dim=(-1, -2))
            x = self.head_bn(x)
            return self.model._tf_decoder(x)

    model = DummyModel()
    entry._apply_freeze_config(
        model,
        {
            "freeze_backbone": True,
            "freeze_tf_decoder": False,
            "freeze_trajectory_head": False,
        },
    )

    backbone_bn = model.model._backbone[1]
    assert isinstance(backbone_bn, nn.BatchNorm2d)
    assert backbone_bn.training is False
    assert backbone_bn.track_running_stats is False
    assert model.head_bn.track_running_stats is True

    output = model(torch.zeros(2, 3, 8, 8))
    assert output.shape == (2, 4)


def test_beta_reg_uses_actual_total_steps():
    trainer = MultiAgentGRPOTrainer(
        model=entry.ToyPlanner(num_agents=1),
        ref_model=entry.ToyPlanner(num_agents=1),
        env=entry.ToyEnv(num_agents=1, mode="toy-single"),
        config={
            "lr": 5e-5,
            "total_steps": 10000,
            "beta_reg_max": 1.0,
            "beta_reg_min": 0.1,
            "beta_reg_warmup_frac": 0.3,
            "beta_reg_decay_frac": 0.4,
        },
    )

    trainer.global_step = 0
    assert trainer._get_beta_reg() == 1.0
    trainer.global_step = 3000
    assert trainer._get_beta_reg() == 1.0
    trainer.global_step = 5000
    assert 0.1 < trainer._get_beta_reg() < 1.0
    trainer.global_step = 7000
    assert trainer._get_beta_reg() == 0.1
    trainer.global_step = 9999
    assert trainer._get_beta_reg() == 0.1


def test_team_reward_proportional_crash_penalty():
    config = {
        "w_team_collision": 10.0,
        "w_team_formation": 0.5,
        "w_team_safety": 1.0,
        "w_team_efficiency": 0.3,
        "d_norm": 10.0,
        "delta_s_max": 10.0,
    }

    def _agent_infos(crash: bool):
        return [
            {"formation_error": 1.0, "progress": 2.0, "crash": crash},
            {"formation_error": 1.0, "progress": 2.0, "crash": crash},
        ]

    reward_0 = compute_team_reward(
        {"agent0": _agent_infos(False), "agent1": _agent_infos(False), "agent2": _agent_infos(False)},
        config,
    )
    reward_1 = compute_team_reward(
        {"agent0": _agent_infos(True), "agent1": _agent_infos(False), "agent2": _agent_infos(False)},
        config,
    )
    reward_3 = compute_team_reward(
        {"agent0": _agent_infos(True), "agent1": _agent_infos(True), "agent2": _agent_infos(True)},
        config,
    )

    assert reward_0 > reward_1 > reward_3
    assert abs((reward_1 - reward_0) + (10.0 / 3.0)) < 1e-4
    assert abs((reward_3 - reward_0) + 10.0) < 1e-4


def test_config_consistency():
    config = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text(encoding="utf-8"))
    assert "kl_threshold" not in config
    assert config["kl_target"] == 8.0
    assert config["lr_decay_factor"] == 0.8
    assert config["lr_grow_factor"] == 1.05
    assert config["max_grad_norm"] == 10.0


def test_no_regression_on_existing_training_loop(tmp_path):
    runtime = entry.build_runtime(
        mode="toy-single",
        config_path="configs/train/platoon_grpo_v2.yaml",
        steps=5,
        render=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        ckpt_path="unused",
        num_agents=1,
        run_training=False,
    )
    trainer = runtime["trainer"]
    obs = trainer.env.reset()
    trainer.current_obs = obs

    try:
        for _ in range(5):
            rollouts = trainer.collect_group_samples(group_size=int(runtime["config"].get("group_size", 2)), obs=obs)
            metrics = trainer.update(rollouts)
            assert {"loss", "kl", "mean_reward"} <= set(metrics.keys())
            assert torch.isfinite(torch.tensor(metrics["loss"]))
            obs = trainer.step_env_with_best(rollouts)
    finally:
        runtime["env"].close()
