import torch
import torch.nn as nn
import pytest

from models.diffusion.ddim_with_logprob import DDIMSchedulerWithLogProb
from train.train_selected_refine_grpo import (
    CBFBarrierGuidanceConfig,
    apply_cbf_barrier_guidance,
    barrier_noise_bounds_m,
    ddv2_refine_grpo_loss,
    _execute_trajectories,
    agent_best_debug_plot_color,
    agent_debug_plot_color,
    refined_background_alpha,
    build_platoon_gt_mode_group_plot_data,
    _refine_state_dict,
    format_all_mode_debug_matrix,
    save_platoon_gt_mode_group_trajectory_plot,
    save_gt_mode_group_trajectory_plot,
    vehicle_box_corners,
    focal_gt_mode_loss,
    freeze_for_selected_refinement,
    load_cls_grpo_weights_for_refinement,
    normalize_group_advantages,
    normalize_multimodal_advantages,
    resolve_advantage_baseline_reward,
    recompute_refine_log_probs,
    reconstruct_heading_from_xy,
    rollout_selected_refinement,
    slice_gt_mode_refinement,
)


def test_focal_gt_mode_loss_has_gradients_for_logits_only():
    logits = torch.zeros(2, 4, requires_grad=True)
    gt_modes = torch.tensor([1, 3])

    loss = focal_gt_mode_loss(logits, gt_modes)

    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_slice_gt_mode_refinement_keeps_only_target_mode_groups():
    num_groups, num_modes, steps = 3, 4, 2
    chains = torch.arange(num_groups * num_modes * (steps + 1) * 1 * 8 * 2, dtype=torch.float32)
    chains = chains.view(num_groups * num_modes, steps + 1, 1, 8, 2)
    refined = torch.arange(num_groups * num_modes * 8 * 2, dtype=torch.float32).view(num_groups * num_modes, 8, 2)
    ref = refined + 1000.0
    rewards = torch.arange(num_groups * num_modes, dtype=torch.float32).view(num_groups, num_modes)
    valid = torch.ones(num_groups, num_modes, dtype=torch.bool)

    out = slice_gt_mode_refinement(
        chains_norm=chains,
        refined_xy=refined,
        ref_mean_xy=ref,
        rewards=rewards,
        num_groups=num_groups,
        num_modes=num_modes,
        gt_mode=2,
    )

    assert out["chains_norm"].shape == (num_groups, steps + 1, 1, 8, 2)
    assert out["refined_xy"].shape == (num_groups, 8, 2)
    assert out["ref_mean_xy"].shape == (num_groups, 8, 2)
    assert torch.equal(out["rewards"], rewards[:, 2])


def test_normalize_group_advantages_zero_mean_on_valid_groups():
    rewards = torch.tensor([1.0, 2.0, 4.0, -10.0])
    valid = torch.tensor([True, True, True, False])

    advantages = normalize_group_advantages(rewards, valid)

    assert torch.allclose(advantages[valid].mean(), torch.tensor(0.0), atol=1e-6)
    assert advantages[~valid].abs().sum() == 0


def test_normalize_group_advantages_can_use_pretrain_reward_baseline():
    rewards = torch.tensor([1.0, 2.0, 4.0, -10.0])
    valid = torch.tensor([True, True, True, False])

    advantages = normalize_group_advantages(rewards, valid, baseline_reward=torch.tensor(2.5))
    expected = torch.zeros_like(rewards)
    vals = rewards[valid]
    expected[valid] = (vals - 2.5) / (vals.std(unbiased=False) + 1e-6)

    assert torch.allclose(advantages, expected, atol=1e-6)


def test_normalize_multimodal_advantages_can_use_pretrain_reward_baseline():
    rewards = torch.tensor([
        [1.0, 10.0],
        [3.0, 12.0],
        [5.0, 14.0],
    ])

    advantages = normalize_multimodal_advantages(
        rewards.reshape(-1),
        num_groups=3,
        num_modes=2,
        positive_only=False,
        baseline_reward=torch.tensor(2.0),
    ).view(3, 2)

    expected = torch.zeros_like(rewards)
    mode0_vals = rewards[:, 0]
    mode1_vals = rewards[:, 1]
    expected[:, 0] = (mode0_vals - 2.0) / (mode0_vals.std(unbiased=False) + 1e-6)
    expected[:, 1] = (mode1_vals - 2.0) / (mode1_vals.std(unbiased=False) + 1e-6)
    assert torch.allclose(advantages, expected, atol=1e-6)


def test_resolve_advantage_baseline_reward_selects_configured_strategy():
    assert resolve_advantage_baseline_reward("pretrain_reward", 0.75) == pytest.approx(0.75)
    assert resolve_advantage_baseline_reward("group_mean", 0.75) is None
    with pytest.raises(ValueError, match="refine_advantage_baseline"):
        resolve_advantage_baseline_reward("bad_mode", 0.75)


def test_reconstruct_heading_from_xy_returns_trajectory_shape():
    xy = torch.stack([torch.arange(8, dtype=torch.float32), torch.zeros(8)], dim=-1)

    traj = reconstruct_heading_from_xy(xy)

    assert traj.shape == (8, 3)
    assert torch.allclose(traj[:, 2], torch.zeros(8), atol=1e-6)


def test_ddv2_refine_grpo_loss_includes_bc_penalty_and_self_detached_ratio():
    new_log_probs = torch.zeros(4, 3, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    refined = torch.ones(4, 8, 2, requires_grad=True)
    selected = torch.zeros(8, 2)

    loss, metrics = ddv2_refine_grpo_loss(
        new_log_probs=new_log_probs,
        advantages=advantages,
        refined_xy=refined,
        selected_xy=selected,
        bc_weight=0.1,
        kl_weight=0.0,
    )

    assert loss.requires_grad
    assert metrics["bc_loss"] > 0
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-6
    assert metrics["clip_fraction"] == 0.0
    loss.backward()
    assert refined.grad is not None
    assert refined.grad.abs().sum() > 0
    assert new_log_probs.grad is not None
    assert new_log_probs.grad.abs().sum() > 0


def test_ddv2_refine_grpo_loss_counts_positive_advantages():
    new_log_probs = torch.zeros(4, 2, requires_grad=True)
    advantages = torch.tensor([1.0, -2.0, 0.5, -0.5])
    refined = torch.zeros(4, 8, 2, requires_grad=True)
    selected = torch.zeros(8, 2)

    _, metrics = ddv2_refine_grpo_loss(
        new_log_probs=new_log_probs,
        advantages=advantages,
        refined_xy=refined,
        selected_xy=selected,
        bc_weight=0.0,
        kl_weight=0.0,
    )

    assert metrics["active_advantage_count"] == 2
    assert metrics["clip_fraction"] == 0.0


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


def test_ddim_scheduler_with_logprob_can_guide_prev_sample_mean():
    scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    scheduler.set_timesteps(1000)
    sample = torch.zeros(2, 1, 8, 2)
    model_output = torch.full_like(sample, 0.2, requires_grad=True)

    raw_prev, _, raw_mean = scheduler.step(
        model_output=model_output,
        timestep=torch.tensor(8),
        sample=sample,
        eta=0.0,
    )
    guided_prev, guided_log_prob, guided_mean = scheduler.step(
        model_output=model_output,
        timestep=torch.tensor(8),
        sample=sample,
        eta=0.0,
        mean_guidance_fn=lambda mean, _timestep: mean - 0.05,
    )

    assert torch.allclose(guided_mean, raw_mean - 0.05)
    assert torch.allclose(guided_prev, raw_prev - 0.05)
    assert guided_log_prob.requires_grad


def test_barrier_noise_bounds_increase_with_noise_t():
    scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )

    bounds_t2 = barrier_noise_bounds_m(scheduler, noise_t=2, sigma_multiplier=3.0)
    bounds_t8 = barrier_noise_bounds_m(scheduler, noise_t=8, sigma_multiplier=3.0)

    assert bounds_t8[0] > bounds_t2[0]
    assert bounds_t8[1] > bounds_t2[1]


def test_cbf_barrier_guidance_reduces_elliptic_violation_and_stays_finite():
    class _Planner:
        model = nn.Module()

    planner = _Planner()
    planner.model._trajectory_head = _IdentityTrajectoryHead()
    prev_mean = torch.zeros(1, 1, 8, 2)
    prev_mean[..., 0] = 2.0
    anchor = torch.zeros(1, 8, 2)
    config = CBFBarrierGuidanceConfig(
        enabled=True,
        guidance_scale=0.05,
        delta_x_m=1.0,
        delta_y_m=1.0,
        eps=1e-4,
        outside_weight=1.0,
        grad_clip_norm=10.0,
    )

    guided, metrics = apply_cbf_barrier_guidance(
        planner,
        prev_sample_mean_norm=prev_mean,
        anchor_xy=anchor,
        config=config,
    )

    raw_delta = (prev_mean.squeeze(1) - anchor).norm(dim=-1).mean()
    guided_delta = (guided.squeeze(1) - anchor).norm(dim=-1).mean()
    assert torch.isfinite(guided).all()
    assert torch.isfinite(torch.tensor(metrics["barrier_cost"]))
    assert guided_delta < raw_delta
    assert metrics["guidance_grad_norm"] > 0


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
    assert rollout["chains_norm"].shape == (3, 5, 1, 8, 2)
    assert torch.allclose(rollout["all_diffusion_output"], rollout["chains_norm"])
    assert rollout["timesteps"].shape == (4,)
    assert rollout["timesteps"].tolist() == [15, 10, 5, 0]
    assert rollout["log_probs"].requires_grad
    assert "old_log_probs" not in rollout


def test_rollout_selected_refinement_initial_chain_uses_scheduler_add_noise(monkeypatch):
    planner = _TinyRefinePlanner()
    scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    calls = {"count": 0}
    original_add_noise = scheduler.add_noise

    def _spy_add_noise(*args, **kwargs):
        calls["count"] += 1
        return original_add_noise(*args, **kwargs)

    monkeypatch.setattr(scheduler, "add_noise", _spy_add_noise)

    rollout = rollout_selected_refinement(
        planner,
        context={},
        selected_traj_np=torch.zeros(8, 3).numpy(),
        num_groups=3,
        noise_t=8,
        noise_std=0.02,
        max_delta_norm=0.05,
        denoise_steps=2,
        eta=0.1,
        scheduler=scheduler,
    )

    assert calls["count"] == 1
    assert rollout["chains_norm"].shape == (3, 3, 1, 8, 2)


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


def test_freeze_for_selected_refinement_unfreezes_reg_decoder_and_cls_branch():
    planner = _FakePlanner()

    trainable_names = freeze_for_selected_refinement(planner)

    assert trainable_names
    for name, param in planner.named_parameters():
        if any(key in name for key in ("hidden_init", "reg_gru_cell", "delta_head", "plan_cls_branch")):
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


def test_cls_grpo_load_then_freeze_keeps_cls_branch_trainable_for_joint_refinement(tmp_path):
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
            assert param.requires_grad, name


def test_refine_state_dict_includes_cls_branch_and_reg_head():
    planner = _FakePlanner()

    state = _refine_state_dict(planner)

    assert any(key.startswith("plan_cls_branch.") for key in state)
    assert any(key.startswith("hidden_init.") for key in state)
    assert any(key.startswith("reg_gru_cell.") for key in state)
    assert any(key.startswith("delta_head.") for key in state)


def test_format_all_mode_debug_matrix_includes_reward_and_advantage():
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    adv_raw = torch.tensor([[-1.0, 0.5], [1.0, -0.5]])
    adv_clipped = adv_raw.clamp(min=0.0)
    text = format_all_mode_debug_matrix(
        global_step=7,
        episode=2,
        env_step=3,
        agent_id="agent1",
        gt_mode=1,
        gt_group=0,
        gt_reward=2.0,
        on_training_mode=0,
        rewards=rewards,
        adv_raw=adv_raw,
        adv_clipped=adv_clipped,
    )

    assert "global_step=7" in text
    assert "agent=agent1" in text
    assert "on_training_mode=0" in text
    assert "reward[G,M]" in text
    assert "adv_raw[G,M]" in text
    assert "adv_clipped[G,M]" in text
    assert "-1." in text


def test_format_all_mode_debug_matrix_includes_mode_names_when_provided():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    adv_raw = torch.zeros_like(rewards)
    text = format_all_mode_debug_matrix(
        global_step=7,
        episode=2,
        env_step=3,
        agent_id="agent1",
        gt_mode=2,
        gt_group=0,
        gt_reward=3.0,
        on_training_mode=0,
        rewards=rewards,
        adv_raw=adv_raw,
        adv_clipped=adv_raw,
        mode_names=["KEEP_HIGH", "LEFT_LC_HIGH", "STOP"],
    )

    assert "mode_names[M]=" in text
    assert "0:KEEP_HIGH" in text
    assert "1:LEFT_LC_HIGH" in text
    assert "2:STOP" in text


def test_vehicle_box_corners_heading_zero_has_length_along_x_axis():
    corners = vehicle_box_corners([10.0, 5.0, 0.0], length=4.8, width=2.0)

    assert corners.shape == (4, 2)
    assert corners[:, 0].max() - corners[:, 0].min() == pytest.approx(4.8)
    assert corners[:, 1].max() - corners[:, 1].min() == pytest.approx(2.0)


def test_save_gt_mode_group_trajectory_plot_writes_png(tmp_path):
    follower_pose = torch.tensor([0.0, 0.0, 0.0]).numpy()
    leader_pose = torch.tensor([8.0, 0.0, 0.0]).numpy()
    refined_xy = torch.stack([
        torch.stack([torch.linspace(0, 7, 8), torch.zeros(8)], dim=-1),
        torch.stack([torch.linspace(0, 7, 8), torch.ones(8)], dim=-1),
    ])
    rewards = torch.tensor([0.2, 0.9])
    valid = torch.tensor([True, True])
    output_path = tmp_path / "gt_mode_group.png"

    save_gt_mode_group_trajectory_plot(
        output_path=output_path,
        agent_id="agent1",
        prev_agent_id="agent0",
        follower_pose=follower_pose,
        leader_pose=leader_pose,
        refined_xy=refined_xy,
        rewards=rewards,
        gt_mode=3,
        gt_group=1,
        global_step=5,
        episode=0,
        env_step=2,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_platoon_gt_mode_group_plot_data_uses_each_agent_pose():
    agent_ids = ["agent0", "agent1", "agent2"]
    poses = {
        "agent0": torch.tensor([0.0, 1.0, 0.0]).numpy(),
        "agent1": torch.tensor([10.0, 2.0, 0.0]).numpy(),
        "agent2": torch.tensor([20.0, 3.0, 0.0]).numpy(),
    }
    base_xy = torch.stack([torch.arange(8, dtype=torch.float32), torch.zeros(8)], dim=-1)
    trajectories = {
        "agent0": base_xy.unsqueeze(0),
        "agent1": torch.stack([base_xy, base_xy + torch.tensor([0.0, 1.0])]),
        "agent2": torch.stack([base_xy, base_xy + torch.tensor([0.0, -1.0])]),
    }

    plot_data = build_platoon_gt_mode_group_plot_data(agent_ids, poses, trajectories)

    assert list(plot_data["boxes"].keys()) == agent_ids
    assert len(plot_data["trajectories"]["agent0"]) == 1
    assert len(plot_data["trajectories"]["agent1"]) == 2
    assert len(plot_data["trajectories"]["agent2"]) == 2
    for agent_id in agent_ids:
        first_world = plot_data["trajectories"][agent_id][0][0]
        assert first_world == pytest.approx(poses[agent_id][:2])


def test_save_platoon_gt_mode_group_trajectory_plot_writes_all_agents_png(tmp_path):
    agent_ids = ["agent0", "agent1", "agent2"]
    poses = {
        "agent0": torch.tensor([0.0, 0.0, 0.0]).numpy(),
        "agent1": torch.tensor([8.0, 0.0, 0.0]).numpy(),
        "agent2": torch.tensor([16.0, 0.0, 0.0]).numpy(),
    }
    base_xy = torch.stack([torch.linspace(0, 7, 8), torch.zeros(8)], dim=-1)
    trajectories = {
        "agent0": base_xy.unsqueeze(0),
        "agent1": torch.stack([base_xy, base_xy + torch.tensor([0.0, 1.0])]),
        "agent2": torch.stack([base_xy, base_xy + torch.tensor([0.0, -1.0])]),
    }
    output_path = tmp_path / "platoon_gt_mode_group.png"

    save_platoon_gt_mode_group_trajectory_plot(
        output_path=output_path,
        agent_ids=agent_ids,
        poses=poses,
        trajectories_by_agent=trajectories,
        gt_modes_by_agent={"agent0": 0, "agent1": 3, "agent2": 2},
        gt_groups_by_agent={"agent0": 0, "agent1": 1, "agent2": 0},
        rewards_by_agent={"agent1": torch.tensor([0.2, 0.9]), "agent2": torch.tensor([0.7, 0.4])},
        global_step=5,
        episode=0,
        env_step=2,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_agent_debug_plot_color_is_shared_by_refined_trajectories_and_boxes():
    assert agent_debug_plot_color(0) == "tab:blue"
    assert agent_debug_plot_color(1) == "tab:orange"
    assert agent_debug_plot_color(2) == "tab:green"
    assert agent_debug_plot_color(5) == "tab:blue"


def test_best_debug_plot_color_is_darker_and_background_more_transparent():
    assert agent_best_debug_plot_color(0) == "mediumblue"
    assert agent_best_debug_plot_color(1) == "darkorange"
    assert agent_best_debug_plot_color(2) == "forestgreen"
    assert refined_background_alpha() < 0.25


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
