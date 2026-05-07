from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.platoon.relation_encoder import RelationEncoder


def _as_hidden_dims(value: Sequence[int] | str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(int(item) for item in items) if items else tuple(default)
    return tuple(int(item) for item in value)


def _make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, int(hidden_dim)), nn.ReLU()])
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class ModeSelectionMLPActorCriticCore(nn.Module):
    """External decentralized MLP actor + centralized critic for mode selection."""

    def __init__(
        self,
        *,
        num_agents: int,
        num_modes: int,
        global_state_dim: int = 60,
        relation_input_dim: int = 12,
        relation_dim: int = 32,
        trajectory_embed_dim: int = 128,
        actor_hidden_dims: Sequence[int] | str | None = (256, 128),
        value_hidden_dim: int = 256,
        lambda_kl: float = 0.02,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_modes = int(num_modes)
        self.global_state_dim = int(global_state_dim)
        self.relation_input_dim = int(relation_input_dim)
        self.relation_dim = int(relation_dim)
        self.trajectory_embed_dim = int(trajectory_embed_dim)
        self.lambda_kl = float(lambda_kl)

        self.relation_encoder = RelationEncoder(
            input_dim=self.relation_input_dim,
            hidden_dim=max(64, self.relation_dim * 2),
            output_dim=self.relation_dim,
        )
        self.trajectory_encoder = _make_mlp(8 * 3, (128,), self.trajectory_embed_dim)
        actor_hidden = _as_hidden_dims(actor_hidden_dims, (256, 128))
        # The mask value is included as an explicit feature, but validity is still
        # enforced by hard masking logits before sampling.
        self.actor_head = _make_mlp(self.relation_dim + self.trajectory_embed_dim + 1, actor_hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(self.global_state_dim, int(value_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(value_hidden_dim), 1),
        )

    def _ensure_batch(self, tensor: torch.Tensor, expected_ndim: int) -> torch.Tensor:
        if tensor.ndim == expected_ndim - 1:
            return tensor.unsqueeze(0)
        return tensor

    def forward(self, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        candidates = self._ensure_batch(obs["trajectory_candidates"].float(), 5)
        relation_states = self._ensure_batch(obs["agent_relation_states"].float(), 3)
        mask = self._ensure_batch(obs["agent_mode_masks"].bool(), 3)

        bs, num_agents, num_modes, horizon, traj_dim = candidates.shape
        if horizon != 8 or traj_dim != 3:
            raise ValueError(f"trajectory_candidates must have shape [B,N,M,8,3], got {tuple(candidates.shape)}")
        relation_flat = relation_states.reshape(bs * num_agents, -1)
        relation_emb = self.relation_encoder(relation_flat).reshape(bs, num_agents, 1, self.relation_dim)
        relation_emb = relation_emb.expand(bs, num_agents, num_modes, self.relation_dim)

        traj_flat = candidates.reshape(bs * num_agents * num_modes, horizon * traj_dim)
        traj_emb = self.trajectory_encoder(traj_flat).reshape(bs, num_agents, num_modes, self.trajectory_embed_dim)
        mask_feature = mask.float().unsqueeze(-1)
        actor_input = torch.cat([relation_emb, traj_emb, mask_feature], dim=-1)
        logits = self.actor_head(actor_input.reshape(bs * num_agents * num_modes, -1)).reshape(bs, num_agents, num_modes)
        logits = logits.masked_fill(~mask, -1e9)

        global_state = obs["global_state"].float()
        if global_state.ndim == 1:
            global_state = global_state.unsqueeze(0)
        values = self.value_head(global_state).squeeze(-1)

        pretrained_logits = obs.get("pretrained_logits")
        if pretrained_logits is not None:
            pretrained_logits = self._ensure_batch(pretrained_logits.float(), 3)
            pretrained_logits = pretrained_logits.masked_fill(~mask, -1e9)
            logp_new = F.log_softmax(logits, dim=-1)
            p_new = logp_new.exp()
            logp_old = F.log_softmax(pretrained_logits, dim=-1)
            kl = (p_new * (logp_new - logp_old)).sum(dim=-1).mean()
        else:
            kl = logits.new_tensor(0.0)
        entropy = torch.distributions.Categorical(logits=logits.reshape(-1, num_modes)).entropy().mean()
        return logits, values, {"KL_to_pretrained": kl, "mode_entropy": entropy}


# Backward-compatible alias for tests/imports that need the actor core class name,
# while the implementation is now the external MLP actor.
ModeClsActorCriticCore = ModeSelectionMLPActorCriticCore


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
        """SB3 MaskablePPO policy whose actor is an external MLP selector."""

        def __init__(
            self,
            *args,
            num_agents: int,
            num_modes: int,
            global_state_dim: int = 60,
            relation_input_dim: int = 12,
            relation_dim: int = 32,
            trajectory_embed_dim: int = 128,
            actor_hidden_dims: Sequence[int] | str | None = (256, 128),
            value_hidden_dim: int = 256,
            lambda_kl: float = 0.02,
            **kwargs,
        ):
            lr_schedule = kwargs.get("lr_schedule")
            if lr_schedule is None and len(args) >= 3:
                lr_schedule = args[2]
            super().__init__(*args, **kwargs)
            self.mode_cls_core = ModeSelectionMLPActorCriticCore(
                num_agents=num_agents,
                num_modes=num_modes,
                global_state_dim=global_state_dim,
                relation_input_dim=relation_input_dim,
                relation_dim=relation_dim,
                trajectory_embed_dim=trajectory_embed_dim,
                actor_hidden_dims=actor_hidden_dims,
                value_hidden_dim=value_hidden_dim,
                lambda_kl=lambda_kl,
            )
            self._last_metrics: dict[str, torch.Tensor] = {}
            self._rebuild_mode_cls_optimizer(lr_schedule)

        def _rebuild_mode_cls_optimizer(self, lr_schedule) -> None:
            if lr_schedule is None:
                raise RuntimeError("ModeClsMaskablePolicy could not resolve SB3 lr_schedule.")
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("mode_cls_core."))
            trainable_parameters = [p for p in self.mode_cls_core.parameters() if p.requires_grad]
            if not trainable_parameters:
                raise RuntimeError("ModeClsMaskablePolicy has no trainable external actor parameters.")
            self.optimizer = self.optimizer_class(
                trainable_parameters,
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

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
