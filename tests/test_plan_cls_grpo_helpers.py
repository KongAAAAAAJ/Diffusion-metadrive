import numpy as np
import torch
import torch.nn.functional as F

from train.train_plan_cls_grpo import grpo_step_loss


def _masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1).detach()


def test_grpo_step_loss_ratio_is_one_when_old_matches_new_policy():
    logits = torch.tensor([0.2, 0.5, -0.1, 1.0], requires_grad=True)
    ref_logits = logits.detach().clone()
    mask = torch.tensor([True, True, False, True])
    rewards = np.asarray([1.0, 2.0, 0.0, 4.0], dtype=np.float32)
    old_log_probs = _masked_log_probs(logits.detach(), mask)

    loss, metrics = grpo_step_loss(
        logits=logits,
        ref_logits=ref_logits,
        mode_valid_mask=mask,
        proxy_rewards=rewards,
        old_log_probs=old_log_probs,
        beta_kl=0.0,
        clip_range=0.2,
        max_log_ratio=5.0,
    )

    assert loss.requires_grad
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-6
    assert metrics["clip_fraction"] == 0.0


def test_grpo_step_loss_reports_clipped_ratio_for_large_policy_shift():
    logits = torch.tensor([8.0, -8.0, 0.0, 7.0], requires_grad=True)
    ref_logits = torch.zeros_like(logits)
    mask = torch.tensor([True, True, False, True])
    rewards = np.asarray([4.0, 1.0, 0.0, 2.0], dtype=np.float32)
    old_log_probs = torch.full_like(logits, -10.0)

    loss, metrics = grpo_step_loss(
        logits=logits,
        ref_logits=ref_logits,
        mode_valid_mask=mask,
        proxy_rewards=rewards,
        old_log_probs=old_log_probs,
        beta_kl=0.0,
        clip_range=0.2,
        max_log_ratio=5.0,
    )

    assert loss.requires_grad
    assert metrics["clip_fraction"] > 0.0
    assert metrics["ratio_max"] > 1.2


def test_grpo_step_loss_ignores_invalid_modes_in_ratio_and_advantage():
    logits = torch.tensor([0.0, 1.0, 100.0, -1.0], requires_grad=True)
    ref_logits = torch.zeros_like(logits)
    mask = torch.tensor([True, True, False, True])
    rewards = np.asarray([1.0, 2.0, 999.0, 3.0], dtype=np.float32)
    old_log_probs = torch.tensor([0.0, 0.0, 1000.0, 0.0])

    loss, metrics = grpo_step_loss(
        logits=logits,
        ref_logits=ref_logits,
        mode_valid_mask=mask,
        proxy_rewards=rewards,
        old_log_probs=old_log_probs,
    )

    assert torch.isfinite(loss)
    assert metrics["n_valid"] == 3
    assert np.isfinite(metrics["ratio_mean"])


def test_grpo_step_loss_returns_complete_zero_metrics_for_too_few_valid_modes():
    logits = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    ref_logits = torch.zeros_like(logits)
    mask = torch.tensor([False, True, False])
    rewards = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    loss, metrics = grpo_step_loss(
        logits=logits,
        ref_logits=ref_logits,
        mode_valid_mask=mask,
        proxy_rewards=rewards,
        old_log_probs=torch.zeros_like(logits),
    )

    assert loss.item() == 0.0
    assert metrics["n_valid"] == 1
    assert metrics["ratio_mean"] == 1.0
    assert metrics["clip_fraction"] == 0.0
