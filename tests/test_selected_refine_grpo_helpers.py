import torch
import torch.nn as nn
import pytest

from models.diffusion.ddim_with_logprob import DDIMSchedulerWithLogProb
from train.train_selected_refine_grpo import (
    _execute_trajectories,
    freeze_for_selected_refinement,
    load_cls_grpo_weights_for_refinement,
    normalize_group_advantages,
    recompute_refine_log_probs,
    reconstruct_heading_from_xy,
    refine_grpo_loss,
    rollout_selected_refinement,
    sample_truncated_refine_noise,
)


def test_sample_truncated_refine_noise_shape_and_delta_bound():
    selected_norm = torch.zeros(1, 8, 2)

    noisy = sample_truncated_refine_noise(
        selected_norm,
        num_groups=4,
        noise_std=0.2,
        max_delta_norm=0.05,
        generator=torch.Generator().manual_seed(7),
    )

    assert noisy.shape == (4, 1, 8, 2)
    assert torch.max(torch.abs(noisy - selected_norm.unsqueeze(0))) <= 0.05001


def test_normalize_group_advantages_zero_mean_on_valid_groups():
    rewards = torch.tensor([1.0, 2.0, 4.0, -10.0])
    valid = torch.tensor([True, True, True, False])

    advantages = normalize_group_advantages(rewards, valid)

    assert torch.allclose(advantages[valid].mean(), torch.tensor(0.0), atol=1e-6)
    assert advantages[~valid].abs().sum() == 0


def test_reconstruct_heading_from_xy_returns_trajectory_shape():
    xy = torch.stack([torch.arange(8, dtype=torch.float32), torch.zeros(8)], dim=-1)

    traj = reconstruct_heading_from_xy(xy)

    assert traj.shape == (8, 3)
    assert torch.allclose(traj[:, 2], torch.zeros(8), atol=1e-6)


def test_refine_grpo_loss_includes_bc_penalty():
    new_log_probs = torch.zeros(4, 3, requires_grad=True)
    old_log_probs = torch.zeros(4, 3)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    refined = torch.ones(4, 8, 2, requires_grad=True)
    selected = torch.zeros(8, 2)

    loss, metrics = refine_grpo_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        refined_xy=refined,
        selected_xy=selected,
        bc_weight=0.1,
        kl_weight=0.0,
        clip_range=0.2,
        max_log_ratio=5.0,
    )

    assert loss.requires_grad
    assert metrics["bc_loss"] > 0
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-6
    loss.backward()
    assert refined.grad is not None
    assert refined.grad.abs().sum() > 0


def test_refine_grpo_loss_reports_clipped_ratio():
    new_log_probs = torch.full((4, 3), 2.0, requires_grad=True)
    old_log_probs = torch.zeros(4, 3)
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
    refined = torch.zeros(4, 8, 2, requires_grad=True)
    selected = torch.zeros(8, 2)

    loss, metrics = refine_grpo_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        refined_xy=refined,
        selected_xy=selected,
        bc_weight=0.0,
        kl_weight=0.0,
        clip_range=0.2,
        max_log_ratio=5.0,
    )

    assert loss.requires_grad
    assert metrics["clip_fraction"] > 0.0
    assert metrics["ratio_max"] > 1.2


def test_ddim_scheduler_with_logprob_shapes_and_gradients():
    scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    scheduler.set_timesteps(1000)
    sample = torch.zeros(2, 1, 8, 2)
    model_output = torch.full_like(sample, 0.2, requires_grad=True)

    prev_sample, log_prob, prev_sample_mean = scheduler.step(
        model_output=model_output,
        timestep=torch.tensor(8),
        sample=sample,
        eta=0.1,
    )

    assert prev_sample.shape == sample.shape
    assert prev_sample_mean.shape == sample.shape
    assert log_prob.shape == (2, 1)
    assert log_prob.requires_grad
    (-log_prob.mean()).backward()
    assert model_output.grad is not None
    assert model_output.grad.abs().sum() > 0


def test_ddim_scheduler_with_logprob_uses_fixed_prev_sample():
    scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    scheduler.set_timesteps(1000)
    sample = torch.zeros(2, 1, 8, 2)
    model_output = torch.full_like(sample, 0.2, requires_grad=True)
    fixed_prev = torch.full_like(sample, 0.33)

    prev_sample, log_prob, _ = scheduler.step(
        model_output=model_output,
        timestep=torch.tensor(8),
        sample=sample,
        eta=0.1,
        prev_sample=fixed_prev,
    )

    assert torch.allclose(prev_sample, fixed_prev)
    assert log_prob.shape == (2, 1)


class _IdentityTrajectoryHead(nn.Module):
    def norm_odo(self, traj):
        return traj

    def denorm_odo(self, traj):
        return traj


class _TinyRefinePlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))
        self.model = nn.Module()
        self.model._trajectory_head = _IdentityTrajectoryHead()

    def _device(self):
        return self.bias.device

    def predict_denoised_traj(self, noisy_traj_norm, timestep, context):
        del timestep, context
        return noisy_traj_norm * 0.5 + self.bias


def test_rollout_selected_refinement_returns_ddim_step_log_probs():
    planner = _TinyRefinePlanner()
    scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    selected = torch.zeros(8, 3).numpy()

    rollout = rollout_selected_refinement(
        planner,
        context={},
        selected_traj_np=selected,
        num_groups=3,
        noise_t=8,
        noise_std=0.02,
        max_delta_norm=0.05,
        denoise_steps=4,
        eta=0.1,
        scheduler=scheduler,
    )

    assert rollout["log_probs"].shape == (3, 4)
    assert rollout["old_log_probs"].shape == (3, 4)
    assert rollout["chains_norm"].shape == (3, 5, 1, 8, 2)
    assert rollout["timesteps"].shape == (4,)
    assert rollout["log_probs"].requires_grad
    assert not rollout["old_log_probs"].requires_grad


def test_recompute_refine_log_probs_uses_fixed_chain_with_gradients():
    planner = _TinyRefinePlanner()
    scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    rollout = rollout_selected_refinement(
        planner,
        context={},
        selected_traj_np=torch.zeros(8, 3).numpy(),
        num_groups=3,
        noise_t=8,
        noise_std=0.02,
        max_delta_norm=0.05,
        denoise_steps=4,
        eta=0.1,
        scheduler=scheduler,
    )

    new_log_probs = recompute_refine_log_probs(
        planner,
        context={},
        chains_norm=rollout["chains_norm"],
        timesteps=rollout["timesteps"],
        scheduler=scheduler,
        eta=0.1,
    )

    assert new_log_probs.shape == (3, 4)
    assert new_log_probs.requires_grad
    (-new_log_probs.mean()).backward()
    assert planner.bias.grad is not None
    assert planner.bias.grad.abs().sum() > 0


class _FakeTaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.plan_cls_branch = nn.Linear(2, 1)
        self.hidden_init = nn.Linear(2, 2)
        self.reg_gru_cell = nn.GRUCell(2, 2)
        self.delta_head = nn.Linear(2, 1)


class _FakePlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.model = nn.Module()
        self.model._trajectory_head = nn.Module()
        layer = nn.Module()
        layer.task_decoder = _FakeTaskDecoder()
        self.model._trajectory_head.diff_decoder = nn.Module()
        self.model._trajectory_head.diff_decoder.layers = nn.ModuleList([layer])


def test_freeze_for_selected_refinement_only_unfreezes_reg_decoder_parts():
    planner = _FakePlanner()

    trainable_names = freeze_for_selected_refinement(planner)

    assert trainable_names
    for name, param in planner.named_parameters():
        if any(key in name for key in ("hidden_init", "reg_gru_cell", "delta_head")):
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name


def test_load_cls_grpo_dir_replaces_refinement_planner_cls_branch(tmp_path):
    planner = _FakePlanner()
    cls_branch = planner.model._trajectory_head.diff_decoder.layers[-1].task_decoder.plan_cls_branch
    with torch.no_grad():
        cls_branch.weight.fill_(0.0)
        cls_branch.bias.fill_(0.0)
    expected = {key: torch.full_like(value, 0.37) for key, value in cls_branch.state_dict().items()}
    ckpt_dir = tmp_path / "cls_grpo" / "checkpoints" / "final"
    ckpt_dir.mkdir(parents=True)
    torch.save(expected, ckpt_dir / "plan_cls_branch.pt")

    source = load_cls_grpo_weights_for_refinement(planner, {"cls_grpo_ckpt_dir": str(ckpt_dir)}, torch.device("cpu"))

    assert source == str(ckpt_dir / "plan_cls_branch.pt")
    for key, value in cls_branch.state_dict().items():
        assert torch.allclose(value, expected[key])


def test_load_cls_grpo_rejects_ambiguous_dir_and_full_ckpt():
    planner = _FakePlanner()

    with pytest.raises(ValueError, match="Only one"):
        load_cls_grpo_weights_for_refinement(
            planner,
            {"cls_grpo_ckpt_dir": "/tmp/a", "cls_grpo_full_ckpt": "/tmp/b.ckpt"},
            torch.device("cpu"),
        )


def test_cls_grpo_load_then_freeze_keeps_cls_branch_frozen(tmp_path):
    planner = _FakePlanner()
    cls_branch = planner.model._trajectory_head.diff_decoder.layers[-1].task_decoder.plan_cls_branch
    ckpt_dir = tmp_path / "final"
    ckpt_dir.mkdir()
    torch.save(cls_branch.state_dict(), ckpt_dir / "plan_cls_branch.pt")

    load_cls_grpo_weights_for_refinement(planner, {"cls_grpo_ckpt_dir": str(ckpt_dir)}, torch.device("cpu"))
    trainable_names = freeze_for_selected_refinement(planner)

    assert trainable_names
    for name, param in planner.named_parameters():
        if "plan_cls_branch" in name:
            assert not param.requires_grad, name


def test_execute_trajectories_populates_pending_trajectory_reward_inputs(monkeypatch):
    import train.train_selected_refine_grpo as refine_mod

    monkeypatch.setattr(
        refine_mod,
        "compute_trajectory_control",
        lambda **kwargs: (torch.zeros(2).numpy().astype("float32"), {}),
    )

    class _Vehicle:
        speed_km_h = 0.0

    class _BaseEnv:
        def __init__(self):
            self.agents = {"agent0": _Vehicle(), "agent1": _Vehicle()}
            self._pending_step_trajectories = {}
            self._pending_step_all_candidates = {}
            self._pending_step_mode_valid_masks = {}
            self._pending_step_mode_groups = {}
            self.pending_seen_by_step = None

        def step(self, actions):
            self.pending_seen_by_step = dict(self._pending_step_trajectories)
            return {}, {"agent0": 1.0, "agent1": 1.0}, {"__all__": True}, {"__all__": False}, {}

    class _Env:
        def __init__(self):
            self.base_env = _BaseEnv()
            self._last_raw_obs = {}
            self._last_obs = {"ok": True}

        def _normalize_raw_obs(self, raw_obs, previous_obs=None):
            return raw_obs

        def _refresh_mode_export(self):
            return {"refreshed": True}

    env = _Env()
    trajectories = {
        "agent0": torch.zeros(8, 3).numpy().astype("float32"),
        "agent1": torch.ones(8, 3).numpy().astype("float32"),
    }

    _execute_trajectories(
        env,
        planner=None,
        agent_ids=["agent0", "agent1"],
        trajectories=trajectories,
        config={"lookahead_index": 2, "target_speed_km_h": 30.0, "controller_type": "stabilized"},
    )

    assert set(env.base_env.pending_seen_by_step) == {"agent0", "agent1"}
    for agent_id, trajectory in trajectories.items():
        assert env.base_env.pending_seen_by_step[agent_id].shape == (8, 3)
        assert (env.base_env.pending_seen_by_step[agent_id] == trajectory).all()
