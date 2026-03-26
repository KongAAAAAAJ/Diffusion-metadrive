from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from torch import nn

import train.train_platoon_rl as entry


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone_weight = nn.Parameter(torch.ones(2))
        self._bev_upscale_weight = nn.Parameter(torch.ones(2))
        self._tf_decoder_weight = nn.Parameter(torch.ones(2))
        self._trajectory_head_weight = nn.Parameter(torch.ones(2))
        self.relation_encoder_weight = nn.Parameter(torch.ones(2))
        self._status_encoding_weight = nn.Parameter(torch.ones(2))


class DummyWriter:
    def __init__(self):
        self.logged = []

    def add_scalar(self, key, value, step):
        self.logged.append((key, value, step))


def test_v2_config_exists_and_contains_required_fields():
    path = Path('configs/train/platoon_grpo_v2.yaml')
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    for key in ['beta_reg_max', 'beta_reg_min', 'lambda_local', 'lambda_team', 'joint_top_k', 'num_joint_groups', 'use_closedloop', 'freeze_backbone']:
        assert key in data
    reward = data['reward_config']
    for key in ['w_team_formation', 'w_team_safety', 'w_team_efficiency', 'w_team_collision']:
        assert key in reward


def test_apply_freeze_config_freezes_backbone_but_not_relation_or_status():
    model = DummyModel()
    entry._apply_freeze_config(model, {
        'freeze_backbone': True,
        'freeze_tf_decoder': False,
        'freeze_trajectory_head': False,
    })
    params = dict(model.named_parameters())
    assert params['_backbone_weight'].requires_grad is False
    assert params['_bev_upscale_weight'].requires_grad is False
    assert params['relation_encoder_weight'].requires_grad is True
    assert params['_status_encoding_weight'].requires_grad is True


def test_log_metrics_prefixes_train_namespace():
    writer = DummyWriter()
    entry._log_metrics(writer, {'ref_reg_loss': 1.2, 'beta_reg': 0.5, 'flag': True}, 3)
    assert ('train/ref_reg_loss', 1.2, 3) in writer.logged
    assert ('train/beta_reg', 0.5, 3) in writer.logged
    assert all(item[0] != 'train/flag' for item in writer.logged)


def test_parse_args_accepts_platoon_closedloop():
    args = entry.parse_args(['--mode', 'platoon-closedloop', '--steps', '5'])
    assert args.mode == 'platoon-closedloop'
    assert args.steps == 5




def test_build_runtime_runs_platoon_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, 'PlatoonEnv', lambda cfg: entry.ToyEnv(num_agents=int(cfg.get('num_agents', 3)), mode='platoon'))
    monkeypatch.setattr(entry, 'PlatoonDiffusionPlanner', lambda *args, **kwargs: entry.ToyPlanner(num_agents=3))
    monkeypatch.setattr(entry, 'migrate_single_to_platoon', lambda ckpt, model: model)
    monkeypatch.setattr(entry, 'build_transfuser_config', lambda *args, **kwargs: {'dummy': True})

    summary = entry.build_runtime(
        mode='platoon',
        config_path='configs/train/platoon_grpo_v2.yaml',
        steps=2,
        render=False,
        checkpoint_dir=str(tmp_path / 'ckpt'),
        log_dir=str(tmp_path / 'logs'),
        ckpt_path='dummy.ckpt',
        run_training=True,
    )
    assert 'loss' in summary
    assert torch.isfinite(torch.tensor(float(summary['loss'][0])))
    assert 'ref_reg_loss' in summary
    for key in ['profile_step_total', 'profile_collect', 'profile_joint', 'profile_update', 'profile_env_step', 'profile_logging', 'profile_checkpoint']:
        assert key in summary
        assert len(summary[key]) == 2
        assert torch.isfinite(torch.tensor(float(summary[key][0])))
    assert 'profile_summary' in summary
    assert 'step_total_mean' in summary['profile_summary']

def test_build_runtime_supports_closedloop_mode_with_monkeypatched_runtime(monkeypatch, tmp_path):
    class DummyEnv(entry.ToyEnv):
        def __init__(self, num_agents: int, mode: str):
            super().__init__(num_agents=num_agents, mode=mode)
        def close(self):
            return None

    class DummyPlanner(entry.ToyPlanner):
        pass

    monkeypatch.setattr(entry, 'PlatoonEnv', lambda cfg: DummyEnv(num_agents=int(cfg.get('num_agents', 3)), mode='platoon-closedloop'))
    monkeypatch.setattr(entry, 'PlatoonDiffusionPlanner', lambda *args, **kwargs: DummyPlanner(num_agents=3))
    monkeypatch.setattr(entry, 'migrate_single_to_platoon', lambda ckpt, model: model)
    monkeypatch.setattr(entry, 'build_transfuser_config', lambda *args, **kwargs: {'dummy': True})

    runtime = entry.build_runtime(
        mode='platoon-closedloop',
        config_path='configs/train/platoon_grpo_v2.yaml',
        steps=1,
        render=False,
        checkpoint_dir=str(tmp_path / 'ckpt'),
        log_dir=str(tmp_path / 'logs'),
        ckpt_path='dummy.ckpt',
        num_agents=3,
        run_training=False,
    )
    assert runtime['config']['use_closedloop'] is True


def test_allocate_rl_run_paths_counts_existing_runs(tmp_path):
    root = tmp_path / 'diffusion_rl'
    (root / 'run_1').mkdir(parents=True)
    (root / 'run_2').mkdir(parents=True)
    (root / 'run_4').mkdir(parents=True)
    run_dir, ckpt_dir, log_dir = entry._allocate_rl_run_paths(root)
    assert run_dir == root / 'run_4'
    assert ckpt_dir == root / 'run_4' / 'checkpoints'
    assert log_dir == root / 'run_4' / 'tb'


def test_resolve_output_dirs_auto_allocates_run_dir_for_rl(monkeypatch, tmp_path):
    root = tmp_path / 'diffusion_rl'
    monkeypatch.setattr(entry, 'DEFAULT_RL_OUTPUT_ROOT', root)
    checkpoint_dir, log_dir = entry._resolve_output_dirs('platoon-closedloop', None, None)
    assert checkpoint_dir == root / 'run_1' / 'checkpoints'
    assert log_dir == root / 'run_1' / 'tb'


def test_resolve_output_dirs_respects_explicit_overrides(monkeypatch, tmp_path):
    root = tmp_path / 'diffusion_rl'
    monkeypatch.setattr(entry, 'DEFAULT_RL_OUTPUT_ROOT', root)
    checkpoint_dir, log_dir = entry._resolve_output_dirs('platoon-closedloop', str(tmp_path / 'ckpt'), str(tmp_path / 'logs'))
    assert checkpoint_dir == tmp_path / 'ckpt'
    assert log_dir == tmp_path / 'logs'
    assert not root.exists()


def test_build_runtime_auto_assigns_run_dir(monkeypatch, tmp_path):
    root = tmp_path / 'diffusion_rl'
    monkeypatch.setattr(entry, 'DEFAULT_RL_OUTPUT_ROOT', root)
    monkeypatch.setattr(entry, 'PlatoonEnv', lambda cfg: entry.ToyEnv(num_agents=int(cfg.get('num_agents', 3)), mode='platoon-closedloop'))
    monkeypatch.setattr(entry, 'PlatoonDiffusionPlanner', lambda *args, **kwargs: entry.ToyPlanner(num_agents=3))
    monkeypatch.setattr(entry, 'migrate_single_to_platoon', lambda ckpt, model: model)
    monkeypatch.setattr(entry, 'build_transfuser_config', lambda *args, **kwargs: {'dummy': True})

    summary = entry.build_runtime(
        mode='platoon-closedloop',
        config_path='configs/train/platoon_grpo_v2.yaml',
        steps=1,
        render=False,
        checkpoint_dir=None,
        log_dir=None,
        ckpt_path='dummy.ckpt',
        num_agents=3,
        run_training=True,
    )
    assert Path(summary['checkpoint_dir']) == root / 'run_1' / 'checkpoints'
    assert Path(summary['log_dir']) == root / 'run_1' / 'tb'


def test_run_marl_train_script_does_not_force_repo_local_output_dirs():
    script = Path('scripts/run_marl_train.sh').read_text(encoding='utf-8')
    assert '--checkpoint-dir checkpoints/platoon_rl' not in script
    assert '--log-dir logs/platoon_rl' not in script
