from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ModeClsActorCriticCore(nn.Module):
    """Shared decentralized actor + centralized critic core for mode selection."""

    def __init__(
        self,
        plan_cls_branch: nn.Module,
        num_agents: int,
        num_modes: int,
        cls_feature_dim: int,
        global_state_dim: int = 60,
        value_hidden_dim: int = 256,
        lambda_kl: float = 0.02,
    ):
        super().__init__()
        self.plan_cls_branch = plan_cls_branch
        self.num_agents = int(num_agents)
        self.num_modes = int(num_modes)
        self.cls_feature_dim = int(cls_feature_dim)
        self.lambda_kl = float(lambda_kl)
        self.value_head = nn.Sequential(
            nn.Linear(int(global_state_dim), int(value_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(value_hidden_dim), 1),
        )

    def forward(self, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        cls_feature = obs["agent_cls_features"].float()
        if cls_feature.ndim == 3:
            cls_feature = cls_feature.unsqueeze(0)
        bs, num_agents, num_modes, feature_dim = cls_feature.shape
        logits = self.plan_cls_branch(cls_feature.reshape(bs * num_agents * num_modes, feature_dim))
        logits = logits.reshape(bs, num_agents, num_modes)
        mask = obs.get("agent_mode_masks")
        if mask is not None:
            mask = mask.bool()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        global_state = obs["global_state"].float()
        if global_state.ndim == 1:
            global_state = global_state.unsqueeze(0)
        values = self.value_head(global_state).squeeze(-1)
        pretrained_logits = obs.get("pretrained_logits")
        if pretrained_logits is not None:
            pretrained_logits = pretrained_logits.float()
            if pretrained_logits.ndim == 2:
                pretrained_logits = pretrained_logits.unsqueeze(0)
            if mask is not None:
                pretrained_logits = pretrained_logits.masked_fill(~mask, -1e9)
            logp_new = F.log_softmax(logits, dim=-1)
            p_new = logp_new.exp()
            logp_old = F.log_softmax(pretrained_logits, dim=-1)
            kl = (p_new * (logp_new - logp_old)).sum(dim=-1).mean()
        else:
            kl = logits.new_tensor(0.0)
        entropy = torch.distributions.Categorical(logits=logits.reshape(-1, num_modes)).entropy().mean()
        return logits, values, {"KL_to_pretrained": kl, "mode_entropy": entropy}

    def regularized_loss(self, ppo_loss: torch.Tensor, metrics: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return ppo_loss + self.lambda_kl * metrics.get("KL_to_pretrained", ppo_loss.new_tensor(0.0))


def export_plan_cls_delta(plan_cls_branch: nn.Module, path: str | Path, metadata: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": plan_cls_branch.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )
    return path


def load_plan_cls_delta(plan_cls_branch: nn.Module, path: str | Path) -> Mapping[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    plan_cls_branch.load_state_dict(checkpoint["state_dict"], strict=True)
    return dict(checkpoint.get("metadata", {}))


def require_sb3() -> tuple[Any, Any]:
    try:
        from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "SB3 mode-selection PPO requires stable-baselines3 and sb3-contrib. "
            "Install them in the meta_drive environment before running training."
        ) from exc
    return ModeClsMaskablePPO, ModeClsMaskablePolicy


try:  # pragma: no cover - optional dependency is absent in CI/dev by default
    try:
        import gymnasium.spaces as spaces
    except Exception:  # pragma: no cover
        import gym.spaces as spaces  # type: ignore
    from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO as _MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy as _MaskableBase
    from stable_baselines3.common.utils import explained_variance

    class ModeClsMaskablePolicy(_MaskableBase):
        """SB3 MaskablePPO policy whose actor is the planner plan_cls_branch."""

        def __init__(
            self,
            *args,
            plan_cls_branch: nn.Module,
            num_agents: int,
            num_modes: int,
            cls_feature_dim: int,
            global_state_dim: int = 60,
            lambda_kl: float = 0.02,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.mode_cls_core = ModeClsActorCriticCore(
                plan_cls_branch=plan_cls_branch,
                num_agents=num_agents,
                num_modes=num_modes,
                cls_feature_dim=cls_feature_dim,
                global_state_dim=global_state_dim,
                lambda_kl=lambda_kl,
            )
            self._last_metrics: dict[str, torch.Tensor] = {}

        def _distribution_from_logits(self, logits: torch.Tensor, action_masks=None):
            dist = self.action_dist.proba_distribution(action_logits=logits.reshape(logits.shape[0], -1))
            if action_masks is not None and hasattr(dist, "apply_masking"):
                dist.apply_masking(action_masks)
            return dist

        def forward(self, obs: Mapping[str, torch.Tensor], deterministic: bool = False, action_masks=None):
            logits, values, metrics = self.mode_cls_core(obs)
            self._last_metrics = metrics
            distribution = self._distribution_from_logits(logits, action_masks=action_masks)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return actions, values, log_prob

        def get_distribution(self, obs: Mapping[str, torch.Tensor], action_masks=None):
            logits, _, metrics = self.mode_cls_core(obs)
            self._last_metrics = metrics
            return self._distribution_from_logits(logits, action_masks=action_masks)

        def evaluate_actions(self, obs: Mapping[str, torch.Tensor], actions: torch.Tensor, action_masks=None):
            logits, values, metrics = self.mode_cls_core(obs)
            self._last_metrics = metrics
            distribution = self._distribution_from_logits(logits, action_masks=action_masks)
            return values, distribution.log_prob(actions), distribution.entropy()

        def predict_values(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
            _, values, metrics = self.mode_cls_core(obs)
            self._last_metrics = metrics
            return values

        def custom_loss(self, policy_loss, loss_inputs):
            if isinstance(policy_loss, list):
                return [self.mode_cls_core.regularized_loss(loss, self._last_metrics) for loss in policy_loss]
            return self.mode_cls_core.regularized_loss(policy_loss, self._last_metrics)

        def metrics(self) -> dict[str, torch.Tensor]:
            return {key: value.detach() for key, value in self._last_metrics.items()}


    class ModeClsMaskablePPO(_MaskablePPO):
        """MaskablePPO with an explicit KL-to-pretrained regularizer."""

        def train(self) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []
            pretrained_kl_losses = []
            mode_entropies = []
            continue_training = True

            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        action_masks=rollout_data.action_masks,
                    )

                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                    pg_losses.append(policy_loss.item())

                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)
                    entropy_losses.append(entropy_loss.item())

                    metrics = getattr(self.policy, "metrics", lambda: {})()
                    pretrained_kl = metrics.get("KL_to_pretrained", policy_loss.new_tensor(0.0))
                    mode_entropy = metrics.get("mode_entropy", policy_loss.new_tensor(0.0))
                    pretrained_kl_losses.append(float(pretrained_kl.detach().cpu()))
                    mode_entropies.append(float(mode_entropy.detach().cpu()))

                    lambda_kl = float(getattr(getattr(self.policy, "mode_cls_core", None), "lambda_kl", 0.0))
                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + lambda_kl * pretrained_kl
                    )

                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                if not continue_training:
                    break

            self._n_updates += self.n_epochs
            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            self.logger.record("train/KL_to_pretrained", np.mean(pretrained_kl_losses) if pretrained_kl_losses else 0.0)
            self.logger.record("train/mode_entropy", np.mean(mode_entropies) if mode_entropies else 0.0)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

except Exception:  # pragma: no cover
    ModeClsMaskablePolicy = object  # type: ignore
    ModeClsMaskablePPO = object  # type: ignore
