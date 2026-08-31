"""Variant-A joint GRPO core for the BEV-only diffusion planner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import weakref
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from torch import Tensor, nn
from torch.optim import AdamW

from models.bev_planner.bev_only_diffusion_planner import (
    BEVOnlyDiffusionPlanner,
    BEVPlannerContext,
    NUM_PLATOON_ROLES,
    TRAJECTORY_DIM,
)
from models.bev_planner.mode_contract import NUM_MODES, TRAJECTORY_STEPS


class JointGRPOError(RuntimeError):
    """Raised when the strict joint GRPO contract is violated."""


@dataclass(frozen=True)
class JointGRPOConfig:
    group_size: int = 4
    initial_noise_timestep: int = 8
    denoise_steps: int = 4
    eta: float = 1.0
    mode_pg_weight: float = 1.0
    trajectory_pg_weight: float = 1.0
    bc_weight: float = 0.1
    reference_kl_weight: float = 0.02
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    advantage_eps: float = 1e-6
    xy_beta_m: float = 1.0
    heading_beta_rad: float = 0.1
    heading_bc_weight: float = 0.2

    def __post_init__(self) -> None:
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 2
        ):
            raise JointGRPOError(
                "group_size must be an integer greater than or equal to 2"
            )
        if self.initial_noise_timestep != 8:
            raise JointGRPOError("initial_noise_timestep is frozen to 8")
        if self.denoise_steps != 4:
            raise JointGRPOError("denoise_steps is frozen to 4")
        if not math.isclose(float(self.eta), 1.0):
            raise JointGRPOError("eta is frozen to 1.0")
        positive = (
            "learning_rate",
            "max_grad_norm",
            "advantage_eps",
            "xy_beta_m",
            "heading_beta_rad",
        )
        non_negative = (
            "mode_pg_weight",
            "trajectory_pg_weight",
            "bc_weight",
            "reference_kl_weight",
            "weight_decay",
            "heading_bc_weight",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise JointGRPOError(f"{name} must be positive and finite")
        for name in non_negative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise JointGRPOError(f"{name} must be non-negative and finite")

    @property
    def roll_timesteps(self) -> tuple[int, ...]:
        return BEVOnlyDiffusionPlanner.inference_roll_timesteps(self.denoise_steps)

    @property
    def stochastic_timesteps(self) -> tuple[int, ...]:
        return tuple(value for value in self.roll_timesteps if value > 0)


@dataclass(frozen=True)
class JointGRPOPolicyUpdateConfig:
    """PPO-style update semantics applied to one frozen joint rollout."""

    update_epochs: int = 4
    clip_epsilon: float = 0.2

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_epochs, bool)
            or not isinstance(self.update_epochs, int)
            or self.update_epochs < 1
        ):
            raise JointGRPOError("update_epochs must be a positive integer")
        clip_epsilon = float(self.clip_epsilon)
        if (
            not math.isfinite(clip_epsilon)
            or clip_epsilon <= 0.0
            or clip_epsilon >= 1.0
        ):
            raise JointGRPOError("clip_epsilon must be finite and in (0,1)")


def joint_grpo_optimizer_contract(
    config: JointGRPOPolicyUpdateConfig | None = None,
) -> dict[str, object]:
    """Return the machine-readable Stage-2 clipped-GRPO optimizer contract."""

    policy_update = config if config is not None else JointGRPOPolicyUpdateConfig()
    return {
        "version": "stage2_joint_grpo_optimizer_v2",
        "behavior_policy_snapshot": (
            "detached sampled mode and stochastic DDIM-transition log "
            "probabilities captured before any policy update"
        ),
        "update_epochs": int(policy_update.update_epochs),
        "clip_epsilon": float(policy_update.clip_epsilon),
        "mode_ratio_factorization": (
            "one ratio per joint sampled mode group entry [B,G]"
        ),
        "trajectory_ratio_factorization": (
            "one ratio per stochastic DDIM transition [B,G,S]"
        ),
        "clipped_surrogate": (
            "negative mean of min(ratio*advantage, "
            "clip(ratio,1-epsilon,1+epsilon)*advantage)"
        ),
        "advantage_reuse": (
            "group-normalize once, detach, and freeze across all epochs of "
            "the rollout"
        ),
        "rollout_consumption": (
            "one external update call consumes one live rollout and performs "
            "exactly update_epochs serial optimizer steps when informative"
        ),
        "minibatch_semantics": "none; reuse the complete joint group each epoch",
        "reference_regularization": (
            "recompute behavior-cloning and frozen-Stage1 reference KL every epoch"
        ),
        "kl_early_stop": False,
        "budget_unit": "fresh_rollout_group",
        "checkpoint_boundary": (
            "after a complete rollout and all of its optimizer epochs"
        ),
        "trainable_modules": ["diffusion_decoder", "mode_head"],
    }


def joint_grpo_optimizer_contract_sha256(
    config: JointGRPOPolicyUpdateConfig | None = None,
) -> str:
    """Return the canonical digest for a concrete optimizer contract."""

    encoded = json.dumps(
        joint_grpo_optimizer_contract(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GaussianDDIMStep:
    prev_sample: Tensor
    mean: Tensor
    std: Tensor
    log_prob: Tensor | None


class StandardGaussianDDIM:
    """DDIM transition with additive Gaussian noise and exact log-probability."""

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.scheduler.set_timesteps(num_train_timesteps)

    def add_noise(
        self, original: Tensor, noise: Tensor, timesteps: Tensor
    ) -> Tensor:
        return self.scheduler.add_noise(original, noise, timesteps)

    def step(
        self,
        *,
        model_output: Tensor,
        timestep: int,
        previous_timestep: int,
        sample: Tensor,
        eta: float,
        generator: torch.Generator | None = None,
        prev_sample: Tensor | None = None,
    ) -> GaussianDDIMStep:
        if isinstance(timestep, bool) or not isinstance(timestep, int):
            raise JointGRPOError("DDIM timestep must be an integer")
        if timestep < 0 or timestep >= self.scheduler.config.num_train_timesteps:
            raise JointGRPOError("DDIM timestep is outside the scheduler")
        if (
            isinstance(previous_timestep, bool)
            or not isinstance(previous_timestep, int)
            or previous_timestep < -1
            or previous_timestep >= timestep
        ):
            raise JointGRPOError(
                "DDIM previous_timestep must be an integer in [-1,timestep)"
            )
        if timestep == 0 and previous_timestep != -1:
            raise JointGRPOError("the deterministic t=0 step must terminate at -1")
        if timestep > 0 and previous_timestep < 0:
            raise JointGRPOError("only t=0 may transition to -1")
        if sample.shape != model_output.shape:
            raise JointGRPOError("DDIM sample and model_output shapes must match")
        if sample.dtype != torch.float32 or model_output.dtype != torch.float32:
            raise JointGRPOError("DDIM probability tensors must use float32")
        if sample.device != model_output.device:
            raise JointGRPOError("DDIM sample and model_output devices must match")
        if not bool(torch.isfinite(sample).all()) or not bool(
            torch.isfinite(model_output).all()
        ):
            raise JointGRPOError("DDIM inputs must be finite")
        if not math.isfinite(float(eta)) or float(eta) < 0.0:
            raise JointGRPOError("DDIM eta must be non-negative and finite")

        device = sample.device
        dtype = sample.dtype
        alpha_t = self.scheduler.alphas_cumprod[timestep].to(device, dtype)
        alpha_previous = (
            self.scheduler.alphas_cumprod[previous_timestep].to(device, dtype)
            if previous_timestep >= 0
            else self.scheduler.final_alpha_cumprod.to(device, dtype)
        )
        beta_t = 1.0 - alpha_t
        prediction = model_output.clamp(
            -float(self.scheduler.config.clip_sample_range),
            float(self.scheduler.config.clip_sample_range),
        )
        epsilon = (sample - alpha_t.sqrt() * prediction) / beta_t.sqrt().clamp_min(
            torch.finfo(dtype).eps
        )
        variance = (
            (1.0 - alpha_previous)
            / (1.0 - alpha_t).clamp_min(torch.finfo(dtype).eps)
            * (1.0 - alpha_t / alpha_previous)
        ).clamp_min(0.0)
        std = float(eta) * variance.sqrt()
        direction_scale = (1.0 - alpha_previous - std.square()).clamp_min(0.0)
        mean = alpha_previous.sqrt() * prediction + direction_scale.sqrt() * epsilon

        stochastic = bool(float(std.detach().cpu()) > 0.0)
        if prev_sample is None:
            if stochastic:
                noise = torch.randn(
                    sample.shape,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                sampled = mean + std * noise
            else:
                sampled = mean
        else:
            if prev_sample.shape != sample.shape or prev_sample.dtype != dtype:
                raise JointGRPOError(
                    "replayed DDIM prev_sample must match sample shape and dtype"
                )
            if prev_sample.device != device or not bool(
                torch.isfinite(prev_sample).all()
            ):
                raise JointGRPOError(
                    "replayed DDIM prev_sample must be finite and colocated"
                )
            sampled = prev_sample

        log_prob = None
        if stochastic:
            elementwise = (
                -0.5 * ((sampled.detach() - mean) / std).square()
                - torch.log(std)
                - 0.5 * math.log(2.0 * math.pi)
            )
            log_prob = elementwise.sum(dim=(-2, -1))
        return GaussianDDIMStep(
            prev_sample=sampled,
            mean=mean,
            std=std,
            log_prob=log_prob,
        )


@dataclass(frozen=True)
class JointGRPORollout:
    context: BEVPlannerContext
    coarse_trajectories: Tensor
    mode_valid_mask: Tensor
    chains_normalized: Tensor
    sampled_modes: Tensor
    selected_trajectories: Tensor
    old_mode_log_prob: Tensor
    old_trajectory_log_prob: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.sampled_modes.shape[0])

    @property
    def group_size(self) -> int:
        return int(self.sampled_modes.shape[1])


@dataclass(frozen=True)
class JointGRPORolloutB(JointGRPORollout):
    predecessor_action_history_normalized: Tensor


@dataclass(frozen=True)
class JointGRPOLossResult:
    total: Tensor
    mode_pg: Tensor
    trajectory_pg: Tensor
    behavior_cloning: Tensor
    mode_reference_kl: Tensor
    trajectory_reference_kl: Tensor
    reference_kl: Tensor
    advantages: Tensor
    new_mode_log_prob: Tensor
    new_trajectory_log_prob: Tensor
    mode_importance_ratio: Tensor
    trajectory_importance_ratio: Tensor
    mode_clip_fraction_low: Tensor
    mode_clip_fraction_high: Tensor
    trajectory_clip_fraction_low: Tensor
    trajectory_clip_fraction_high: Tensor
    mode_old_policy_approx_kl: Tensor
    trajectory_old_policy_approx_kl: Tensor

    def scalar_metrics(self) -> dict[str, float]:
        values = {
            "loss/total": self.total,
            "loss/mode_pg": self.mode_pg,
            "loss/trajectory_pg": self.trajectory_pg,
            "loss/behavior_cloning": self.behavior_cloning,
            "loss/mode_reference_kl": self.mode_reference_kl,
            "loss/trajectory_reference_kl": self.trajectory_reference_kl,
            "loss/reference_kl": self.reference_kl,
            "advantage/mean": self.advantages.mean(),
            "advantage/std": self.advantages.std(unbiased=False),
            "advantage/min": self.advantages.min(),
            "advantage/max": self.advantages.max(),
            "policy/mode_ratio_mean": self.mode_importance_ratio.mean(),
            "policy/mode_ratio_min": self.mode_importance_ratio.min(),
            "policy/mode_ratio_max": self.mode_importance_ratio.max(),
            "policy/trajectory_ratio_mean": (
                self.trajectory_importance_ratio.mean()
            ),
            "policy/trajectory_ratio_min": (
                self.trajectory_importance_ratio.min()
            ),
            "policy/trajectory_ratio_max": (
                self.trajectory_importance_ratio.max()
            ),
            "policy/mode_clip_fraction_low": self.mode_clip_fraction_low,
            "policy/mode_clip_fraction_high": self.mode_clip_fraction_high,
            "policy/mode_clip_fraction": (
                self.mode_clip_fraction_low + self.mode_clip_fraction_high
            ),
            "policy/trajectory_clip_fraction_low": (
                self.trajectory_clip_fraction_low
            ),
            "policy/trajectory_clip_fraction_high": (
                self.trajectory_clip_fraction_high
            ),
            "policy/trajectory_clip_fraction": (
                self.trajectory_clip_fraction_low
                + self.trajectory_clip_fraction_high
            ),
            "policy/mode_old_policy_approx_kl": (
                self.mode_old_policy_approx_kl
            ),
            "policy/trajectory_old_policy_approx_kl": (
                self.trajectory_old_policy_approx_kl
            ),
        }
        return {name: float(value.detach().cpu()) for name, value in values.items()}

    def detached(self) -> JointGRPOLossResult:
        """Return a graph-free result safe to retain across optimizer epochs."""

        return JointGRPOLossResult(
            total=self.total.detach(),
            mode_pg=self.mode_pg.detach(),
            trajectory_pg=self.trajectory_pg.detach(),
            behavior_cloning=self.behavior_cloning.detach(),
            mode_reference_kl=self.mode_reference_kl.detach(),
            trajectory_reference_kl=self.trajectory_reference_kl.detach(),
            reference_kl=self.reference_kl.detach(),
            advantages=self.advantages.detach(),
            new_mode_log_prob=self.new_mode_log_prob.detach(),
            new_trajectory_log_prob=self.new_trajectory_log_prob.detach(),
            mode_importance_ratio=self.mode_importance_ratio.detach(),
            trajectory_importance_ratio=(
                self.trajectory_importance_ratio.detach()
            ),
            mode_clip_fraction_low=self.mode_clip_fraction_low.detach(),
            mode_clip_fraction_high=self.mode_clip_fraction_high.detach(),
            trajectory_clip_fraction_low=(
                self.trajectory_clip_fraction_low.detach()
            ),
            trajectory_clip_fraction_high=(
                self.trajectory_clip_fraction_high.detach()
            ),
            mode_old_policy_approx_kl=(
                self.mode_old_policy_approx_kl.detach()
            ),
            trajectory_old_policy_approx_kl=(
                self.trajectory_old_policy_approx_kl.detach()
            ),
        )


@dataclass(frozen=True)
class JointGRPOEpochUpdateResult:
    epoch_in_rollout: int
    loss: JointGRPOLossResult
    gradient_norms: Mapping[str, float]
    total_gradient_norm: float
    optimizer_step: int


@dataclass(frozen=True)
class JointGRPOUpdateResult:
    loss: JointGRPOLossResult
    gradient_norms: Mapping[str, float]
    total_gradient_norm: float
    optimizer_step: int
    epoch_results: tuple[JointGRPOEpochUpdateResult, ...]


class FrozenGRPOReference(nn.Module):
    """Frozen decoder and mode head copied from the initial Stage 1 policy."""

    def __init__(self, planner: BEVOnlyDiffusionPlanner) -> None:
        super().__init__()
        self.diffusion_decoder = copy.deepcopy(planner.diffusion_decoder)
        self.mode_head = copy.deepcopy(planner.mode_head)
        self.requires_grad_(False)
        self.eval()

    def predict(
        self,
        planner: BEVOnlyDiffusionPlanner,
        noisy_xy_normalized: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse_trajectories: Tensor,
    ) -> tuple[Tensor, Tensor]:
        noisy_metric = planner._denormalize_xy(
            noisy_xy_normalized.clamp(-1.0, 1.0)
        )
        candidates, features = self.diffusion_decoder(
            noisy_metric,
            coarse_trajectories,
            timesteps,
            context,
        )
        return candidates, self.mode_head(features).squeeze(-1)


# Preserve the Round-11 public name while using a variant-neutral implementation.
FrozenVariantAReference = FrozenGRPOReference


def normalize_signed_advantages(
    rewards: Tensor, *, group_size: int = 4, eps: float = 1e-6
) -> Tensor:
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 2
    ):
        raise JointGRPOError(
            "group_size must be an integer greater than or equal to 2"
        )
    if not isinstance(rewards, Tensor):
        raise JointGRPOError("rewards must be a torch.Tensor")
    if rewards.dtype != torch.float32:
        raise JointGRPOError("rewards must use dtype torch.float32")
    if rewards.ndim != 2 or int(rewards.shape[1]) != int(group_size):
        raise JointGRPOError(f"rewards must have shape [B,{group_size}]")
    if int(rewards.shape[0]) <= 0 or not bool(torch.isfinite(rewards).all()):
        raise JointGRPOError("rewards must contain a non-empty finite batch")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, unbiased=False, keepdim=True)
    return centered / (scale + float(eps))


def _clipped_grpo_surrogate(
    importance_ratio: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
) -> Tensor:
    """Return the negative PPO clipped surrogate for aligned components."""

    unclipped = importance_ratio * advantages
    clipped = importance_ratio.clamp(
        1.0 - float(clip_epsilon),
        1.0 + float(clip_epsilon),
    ) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def _importance_ratio_diagnostics(
    log_ratio: Tensor,
    importance_ratio: Tensor,
    *,
    clip_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    detached_log_ratio = log_ratio.detach()
    detached_ratio = importance_ratio.detach()
    low = (detached_ratio < 1.0 - float(clip_epsilon)).float().mean()
    high = (detached_ratio > 1.0 + float(clip_epsilon)).float().mean()
    approximate_kl = (
        detached_ratio - 1.0 - detached_log_ratio
    ).mean().clamp_min(0.0)
    return low, high, approximate_kl


def _repeat_context(context: BEVPlannerContext, groups: int) -> BEVPlannerContext:
    batch_size = context.batch_size
    bev = (
        context.bev_feature.unsqueeze(1)
        .expand(-1, groups, -1, -1, -1, -1)
        .reshape(batch_size * groups, *context.bev_feature.shape[1:])
    )
    roles = (
        context.role_tokens.unsqueeze(1)
        .expand(-1, groups, -1, -1)
        .reshape(batch_size * groups, *context.role_tokens.shape[1:])
    )
    return BEVPlannerContext(bev_feature=bev, role_tokens=roles)


def _repeat_groups(value: Tensor, groups: int) -> Tensor:
    return (
        value.unsqueeze(1)
        .expand(-1, groups, *([-1] * (value.ndim - 1)))
        .reshape(value.shape[0] * groups, *value.shape[1:])
    )


def _gather_modes(value: Tensor, modes: Tensor) -> Tensor:
    """Gather the mode axis from [N,3,K,...] with modes [N,3]."""

    tail = value.shape[3:]
    index = modes[..., None]
    for _ in tail:
        index = index.unsqueeze(-1)
    index = index.expand(-1, -1, 1, *tail)
    return value.gather(2, index).squeeze(2)


def _sample_masked_modes(
    logits: Tensor,
    valid_mask: Tensor,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    masked = logits.float().masked_fill(~valid_mask, float("-inf"))
    probabilities = torch.softmax(masked, dim=-1)
    flat = probabilities.reshape(-1, NUM_MODES)
    modes = torch.multinomial(flat, 1, replacement=True, generator=generator).reshape(
        logits.shape[0], NUM_PLATOON_ROLES
    )
    role_log_prob = torch.log_softmax(masked, dim=-1).gather(
        -1, modes.unsqueeze(-1)
    ).squeeze(-1)
    return modes, role_log_prob.sum(dim=-1)


def _masked_categorical_kl(
    current_logits: Tensor, reference_logits: Tensor, valid_mask: Tensor
) -> Tensor:
    current = current_logits.float().masked_fill(~valid_mask, float("-inf"))
    reference = reference_logits.float().masked_fill(~valid_mask, float("-inf"))
    current_log = torch.log_softmax(current, dim=-1)
    reference_log = torch.log_softmax(reference, dim=-1)
    probability = torch.softmax(current, dim=-1)
    # Avoid creating ``-inf - -inf`` on invalid modes.  Masking the final
    # expression with torch.where is insufficient because autograd can still
    # encounter the NaN intermediate during backward.
    current_log = current_log.masked_fill(~valid_mask, 0.0)
    reference_log = reference_log.masked_fill(~valid_mask, 0.0)
    probability = probability.masked_fill(~valid_mask, 0.0)
    terms = probability * (current_log - reference_log)
    # Analytic KL is non-negative.  Float32 reduction can produce a tiny
    # negative value (around 1e-8) for identical GPU policies.
    return terms.sum(dim=-1).mean().clamp_min(0.0)


def _trajectory_bc(current: Tensor, reference: Tensor, config: JointGRPOConfig) -> Tensor:
    xy = F.smooth_l1_loss(
        current[..., :2],
        reference[..., :2],
        beta=float(config.xy_beta_m),
    )
    heading_error = torch.atan2(
        torch.sin(current[..., 2] - reference[..., 2]),
        torch.cos(current[..., 2] - reference[..., 2]),
    )
    heading = F.smooth_l1_loss(
        heading_error,
        torch.zeros_like(heading_error),
        beta=float(config.heading_beta_rad),
    )
    return xy + float(config.heading_bc_weight) * heading


class _JointGRPOTrainerBase:
    """Variant-neutral clipped joint GRPO implementation."""

    variant = ""
    predecessor_condition = ""

    required_model_inputs = (
        "bev",
        "ego_state",
        "formation_relation_state",
        "relation_valid_mask",
        "agent_role",
        "coarse_trajectories",
        "mode_valid_mask",
    )

    def __init__(
        self,
        planner: BEVOnlyDiffusionPlanner,
        config: JointGRPOConfig | None = None,
    ) -> None:
        if planner.config.predecessor_condition != self.predecessor_condition:
            raise JointGRPOError(
                f"JointGRPOTrainer{self.variant} requires variant "
                f"{self.variant} / {self.predecessor_condition}"
            )
        self.planner = planner
        self.config = config or JointGRPOConfig()
        self.transition = StandardGaussianDDIM(planner.config.num_train_timesteps)
        for parameter in planner.parameters():
            parameter.requires_grad_(False)
        for parameter in planner.diffusion_decoder.parameters():
            parameter.requires_grad_(True)
        for parameter in planner.mode_head.parameters():
            parameter.requires_grad_(True)
        self.reference = FrozenGRPOReference(planner).to(
            next(planner.parameters()).device
        )
        self.planner.eval()
        self.optimizer = AdamW(
            [
                *self.planner.diffusion_decoder.parameters(),
                *self.planner.mode_head.parameters(),
            ],
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        self.optimizer_step = 0
        self._consumed_rollouts: weakref.WeakValueDictionary[
            int, JointGRPORollout
        ] = weakref.WeakValueDictionary()

    def _context_from_inputs(self, model_inputs: Mapping[str, Tensor]) -> BEVPlannerContext:
        missing = [name for name in self.required_model_inputs if name not in model_inputs]
        if missing:
            raise JointGRPOError(f"model_inputs are missing fields: {missing}")
        v2_inputs = {
            name: model_inputs[name]
            for name in (
                "background_actor_state",
                "background_actor_valid_mask",
                "scenario_code",
                "rule_formation_state",
                "rule_action_condition",
            )
            if name in model_inputs
        }
        with torch.no_grad():
            return self.planner.encode_context(
                model_inputs["bev"],
                model_inputs["ego_state"],
                model_inputs["formation_relation_state"],
                model_inputs["relation_valid_mask"],
                model_inputs["agent_role"],
                **v2_inputs,
            )

    def _rollout_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        candidates, logits = self.planner.predict_denoised_candidates(
            sample,
            timesteps,
            context,
            coarse,
            valid_mask,
        )
        return candidates, logits, None

    def _frozen_reference_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del valid_mask
        return self.reference.predict(
            self.planner,
            sample,
            timesteps,
            context,
            coarse,
        )

    @torch.inference_mode()
    def infer_frozen_pretrain(
        self,
        rollout: JointGRPORollout,
    ) -> dict[str, Tensor]:
        """Run one deterministic Stage-1 policy inference on a rollout state."""

        if not isinstance(rollout, JointGRPORollout):
            raise JointGRPOError("frozen pretrain inference requires a joint rollout")
        context = rollout.context
        coarse = rollout.coarse_trajectories
        valid_mask = rollout.mode_valid_mask
        self.planner._validate_trajectory_inputs(
            coarse,
            valid_mask,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )

        batch_size = context.batch_size
        device = coarse.device
        anchor_normalized = self.planner._normalize_xy(coarse[..., :2])
        noise = self.planner._inference_noise(
            anchor_normalized.shape,
            device=device,
        )
        noise_timesteps = torch.full(
            (batch_size * NUM_PLATOON_ROLES,),
            self.planner.config.inference_noise_timestep,
            device=device,
            dtype=torch.int64,
        )
        flat_shape = (
            batch_size * NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        sample = self.planner.diffusion_scheduler.add_noise(
            anchor_normalized.reshape(flat_shape),
            noise.reshape(flat_shape),
            noise_timesteps,
        ).reshape_as(anchor_normalized)
        self.planner.diffusion_scheduler.set_timesteps(
            self.planner.config.num_train_timesteps,
            device=device,
        )

        candidates: Tensor | None = None
        raw_logits: Tensor | None = None
        for timestep in self.planner.inference_roll_timesteps(
            self.planner.config.inference_denoise_steps
        ):
            batch_timesteps = torch.full(
                (batch_size, NUM_PLATOON_ROLES),
                timestep,
                device=device,
                dtype=torch.int64,
            )
            candidates, raw_logits = self._frozen_reference_prediction(
                sample.clamp(-1.0, 1.0),
                batch_timesteps,
                context,
                coarse,
                valid_mask,
            )
            predicted_normalized = self.planner._normalize_xy(
                candidates[..., :2]
            )
            sample = self.planner.diffusion_scheduler.step(
                model_output=predicted_normalized.reshape(flat_shape),
                timestep=timestep,
                sample=sample.reshape(flat_shape),
            ).prev_sample.reshape_as(sample)
        if candidates is None or raw_logits is None:
            raise JointGRPOError(
                "frozen pretrain inference produced no denoising steps"
            )
        selected = self.planner._select(candidates, raw_logits, valid_mask)
        return {
            "selected_trajectory": selected["selected_trajectory"],
            "selected_mode": selected["selected_mode"],
        }

    def _make_rollout(
        self,
        *,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        chains: Tensor,
        modes: Tensor,
        selected: Tensor,
        mode_log_prob: Tensor,
        trajectory_log_prob: Tensor,
        step_histories: list[Tensor],
    ) -> JointGRPORollout:
        if step_histories:
            raise JointGRPOError("variant A rollout must not contain action history")
        return JointGRPORollout(
            context=BEVPlannerContext(
                bev_feature=context.bev_feature.detach(),
                role_tokens=context.role_tokens.detach(),
            ),
            coarse_trajectories=coarse.detach(),
            mode_valid_mask=valid_mask.detach(),
            chains_normalized=chains.detach(),
            sampled_modes=modes.detach(),
            selected_trajectories=selected.detach(),
            old_mode_log_prob=mode_log_prob.detach(),
            old_trajectory_log_prob=trajectory_log_prob.detach(),
        )

    @torch.no_grad()
    def sample_groups(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        generator: torch.Generator,
    ) -> JointGRPORollout:
        if not isinstance(generator, torch.Generator):
            raise JointGRPOError("sample_groups requires an explicit torch.Generator")
        context = self._context_from_inputs(model_inputs)
        coarse = model_inputs["coarse_trajectories"]
        valid_mask = model_inputs["mode_valid_mask"]
        self.planner._validate_trajectory_inputs(
            coarse,
            valid_mask,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )
        groups = self.config.group_size
        repeated_context = _repeat_context(context, groups)
        repeated_coarse = _repeat_groups(coarse, groups)
        repeated_mask = _repeat_groups(valid_mask, groups)
        anchor = self.planner._normalize_xy(repeated_coarse[..., :2])
        noise = torch.randn(
            anchor.shape,
            device=anchor.device,
            dtype=torch.float32,
            generator=generator,
        )
        noise_timestep = torch.full(
            (anchor.shape[0] * NUM_PLATOON_ROLES,),
            self.config.initial_noise_timestep,
            device=anchor.device,
            dtype=torch.int64,
        )
        flat_shape = (
            anchor.shape[0] * NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        sample = self.transition.add_noise(
            anchor.reshape(flat_shape),
            noise.reshape(flat_shape),
            noise_timestep,
        ).reshape_as(anchor).clamp(-1.0, 1.0)
        chains = [sample]
        stochastic_log_probs = []
        step_histories: list[Tensor] = []
        final_candidates = final_logits = None
        destinations = (*self.config.roll_timesteps[1:], -1)
        for timestep, previous_timestep in zip(
            self.config.roll_timesteps, destinations
        ):
            batch_timestep = torch.full(
                (anchor.shape[0], NUM_PLATOON_ROLES),
                timestep,
                device=anchor.device,
                dtype=torch.int64,
            )
            candidates, logits, step_history = self._rollout_prediction(
                sample,
                batch_timestep,
                repeated_context,
                repeated_coarse,
                repeated_mask,
            )
            if step_history is not None:
                step_histories.append(step_history)
            model_output = self.planner._normalize_xy(candidates[..., :2]).float()
            transition = self.transition.step(
                model_output=model_output,
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample.float(),
                eta=self.config.eta,
                generator=generator,
            )
            if transition.log_prob is not None:
                stochastic_log_probs.append(transition.log_prob)
            sample = transition.prev_sample.detach()
            chains.append(sample)
            final_candidates, final_logits = candidates, logits
        if final_candidates is None or final_logits is None:
            raise JointGRPOError("joint rollout produced no decoder output")
        if len(stochastic_log_probs) != len(self.config.stochastic_timesteps):
            raise JointGRPOError("joint rollout stochastic transition count mismatch")
        modes, mode_log_prob = _sample_masked_modes(
            final_logits,
            repeated_mask,
            generator=generator,
        )
        trajectory_log_prob = torch.stack(
            [
                _gather_modes(value.unsqueeze(-1), modes).squeeze(-1).sum(dim=-1)
                for value in stochastic_log_probs
            ],
            dim=-1,
        )
        selected = _gather_modes(final_candidates, modes)
        batch_size = context.batch_size
        chain_tensor = torch.stack(chains, dim=1).reshape(
            batch_size,
            groups,
            len(chains),
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        return self._make_rollout(
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            chains=chain_tensor,
            modes=modes.reshape(batch_size, groups, NUM_PLATOON_ROLES),
            selected=selected.reshape(
                batch_size,
                groups,
                NUM_PLATOON_ROLES,
                TRAJECTORY_STEPS,
                TRAJECTORY_DIM,
            ),
            mode_log_prob=mode_log_prob.reshape(batch_size, groups),
            trajectory_log_prob=trajectory_log_prob.reshape(
                batch_size, groups, len(self.config.stochastic_timesteps)
            ),
            step_histories=step_histories,
        )

    def _replay_history(
        self,
        rollout: JointGRPORollout,
        *,
        step_index: int,
        flat_count: int,
    ) -> Tensor | None:
        del rollout, step_index, flat_count
        return None

    def _replay_predictions(
        self,
        *,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        predecessor_history: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if predecessor_history is not None:
            raise JointGRPOError("variant A replay must not receive action history")
        current_candidates, current_logits = (
            self.planner.predict_denoised_candidates(
                sample,
                timesteps,
                context,
                coarse,
                valid_mask,
            )
        )
        with torch.no_grad():
            reference_candidates, reference_logits = self.reference.predict(
                self.planner,
                sample,
                timesteps,
                context,
                coarse,
            )
        return (
            current_candidates,
            current_logits,
            reference_candidates,
            reference_logits,
        )

    def compute_loss(
        self,
        rollout: JointGRPORollout,
        rewards: Tensor,
        *,
        clip_epsilon: float = 0.2,
    ) -> JointGRPOLossResult:
        policy_update = JointGRPOPolicyUpdateConfig(
            update_epochs=1,
            clip_epsilon=clip_epsilon,
        )
        advantages = normalize_signed_advantages(
            rewards,
            group_size=self.config.group_size,
            eps=self.config.advantage_eps,
        ).to(device=rollout.old_mode_log_prob.device).detach()
        return self._compute_loss_from_advantages(
            rollout,
            advantages,
            clip_epsilon=policy_update.clip_epsilon,
        )

    def _compute_loss_from_advantages(
        self,
        rollout: JointGRPORollout,
        advantages: Tensor,
        *,
        clip_epsilon: float,
    ) -> JointGRPOLossResult:
        if tuple(advantages.shape) != (
            rollout.batch_size,
            rollout.group_size,
        ):
            raise JointGRPOError("reward batch does not match rollout")
        batch_size = rollout.batch_size
        groups = rollout.group_size
        flat_count = batch_size * groups
        context = _repeat_context(rollout.context, groups)
        coarse = _repeat_groups(rollout.coarse_trajectories, groups)
        valid_mask = _repeat_groups(rollout.mode_valid_mask, groups)
        modes = rollout.sampled_modes.reshape(flat_count, NUM_PLATOON_ROLES)
        chains = rollout.chains_normalized.reshape(
            flat_count,
            len(self.config.roll_timesteps) + 1,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )

        current_log_probs = []
        trajectory_kls = []
        final_current_candidates = final_reference_candidates = None
        final_current_logits = final_reference_logits = None
        destinations = (*self.config.roll_timesteps[1:], -1)
        for step_index, (timestep, previous_timestep) in enumerate(
            zip(self.config.roll_timesteps, destinations)
        ):
            sample = chains[:, step_index]
            next_sample = chains[:, step_index + 1]
            batch_timestep = torch.full(
                (flat_count, NUM_PLATOON_ROLES),
                timestep,
                device=sample.device,
                dtype=torch.int64,
            )
            predecessor_history = self._replay_history(
                rollout,
                step_index=step_index,
                flat_count=flat_count,
            )
            (
                current_candidates,
                current_logits,
                reference_candidates,
                reference_logits,
            ) = self._replay_predictions(
                sample=sample,
                timesteps=batch_timestep,
                context=context,
                coarse=coarse,
                valid_mask=valid_mask,
                predecessor_history=predecessor_history,
            )
            current_output = self.planner._normalize_xy(
                current_candidates[..., :2]
            ).float()
            reference_output = self.planner._normalize_xy(
                reference_candidates[..., :2]
            ).float()
            current_transition = self.transition.step(
                model_output=current_output,
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample.float(),
                eta=self.config.eta,
                prev_sample=next_sample.float(),
            )
            with torch.no_grad():
                reference_transition = self.transition.step(
                    model_output=reference_output,
                    timestep=timestep,
                    previous_timestep=previous_timestep,
                    sample=sample.float(),
                    eta=self.config.eta,
                    prev_sample=next_sample.float(),
                )
            if current_transition.log_prob is not None:
                selected_log_prob = _gather_modes(
                    current_transition.log_prob.unsqueeze(-1), modes
                ).squeeze(-1)
                current_log_probs.append(selected_log_prob.sum(dim=-1))
                selected_current_mean = _gather_modes(
                    current_transition.mean, modes
                )
                selected_reference_mean = _gather_modes(
                    reference_transition.mean, modes
                )
                sigma_squared = current_transition.std.square()
                trajectory_kls.append(
                    (
                        (selected_current_mean - selected_reference_mean).square()
                        / (2.0 * sigma_squared)
                    ).mean()
                )
            final_current_candidates = current_candidates
            final_reference_candidates = reference_candidates
            final_current_logits = current_logits
            final_reference_logits = reference_logits

        if (
            final_current_candidates is None
            or final_reference_candidates is None
            or final_current_logits is None
            or final_reference_logits is None
        ):
            raise JointGRPOError("joint replay produced no decoder output")
        new_trajectory_log_prob = torch.stack(current_log_probs, dim=-1).reshape(
            batch_size, groups, -1
        )
        masked_current = final_current_logits.float().masked_fill(
            ~valid_mask, float("-inf")
        )
        role_mode_log_prob = torch.log_softmax(masked_current, dim=-1).gather(
            -1, modes.unsqueeze(-1)
        ).squeeze(-1)
        new_mode_log_prob = role_mode_log_prob.sum(dim=-1).reshape(
            batch_size, groups
        )
        mode_log_ratio = new_mode_log_prob - rollout.old_mode_log_prob
        trajectory_log_ratio = (
            new_trajectory_log_prob - rollout.old_trajectory_log_prob
        )
        mode_ratio = torch.exp(mode_log_ratio)
        trajectory_ratio = torch.exp(trajectory_log_ratio)
        mode_pg = _clipped_grpo_surrogate(
            mode_ratio,
            advantages,
            clip_epsilon=clip_epsilon,
        )
        trajectory_pg = _clipped_grpo_surrogate(
            trajectory_ratio,
            advantages.unsqueeze(-1).expand_as(trajectory_ratio),
            clip_epsilon=clip_epsilon,
        )
        (
            mode_clip_fraction_low,
            mode_clip_fraction_high,
            mode_old_policy_approx_kl,
        ) = _importance_ratio_diagnostics(
            mode_log_ratio,
            mode_ratio,
            clip_epsilon=clip_epsilon,
        )
        (
            trajectory_clip_fraction_low,
            trajectory_clip_fraction_high,
            trajectory_old_policy_approx_kl,
        ) = _importance_ratio_diagnostics(
            trajectory_log_ratio,
            trajectory_ratio,
            clip_epsilon=clip_epsilon,
        )

        current_selected = _gather_modes(final_current_candidates, modes)
        reference_selected = _gather_modes(final_reference_candidates, modes)
        behavior_cloning = _trajectory_bc(
            current_selected,
            reference_selected,
            self.config,
        )
        mode_reference_kl = _masked_categorical_kl(
            final_current_logits,
            final_reference_logits,
            valid_mask,
        )
        trajectory_reference_kl = torch.stack(trajectory_kls).mean()
        reference_kl = mode_reference_kl + trajectory_reference_kl
        total = (
            float(self.config.mode_pg_weight) * mode_pg
            + float(self.config.trajectory_pg_weight) * trajectory_pg
            + float(self.config.bc_weight) * behavior_cloning
            + float(self.config.reference_kl_weight) * reference_kl
        )
        tensors = (
            total,
            mode_pg,
            trajectory_pg,
            behavior_cloning,
            mode_reference_kl,
            trajectory_reference_kl,
            reference_kl,
            new_mode_log_prob,
            new_trajectory_log_prob,
            mode_ratio,
            trajectory_ratio,
            mode_old_policy_approx_kl,
            trajectory_old_policy_approx_kl,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise JointGRPOError("joint GRPO loss contains non-finite values")
        if bool((mode_reference_kl < -1e-7).item()) or bool(
            (trajectory_reference_kl < -1e-7).item()
        ):
            raise JointGRPOError("reference KL must be non-negative")
        return JointGRPOLossResult(
            total=total,
            mode_pg=mode_pg,
            trajectory_pg=trajectory_pg,
            behavior_cloning=behavior_cloning,
            mode_reference_kl=mode_reference_kl,
            trajectory_reference_kl=trajectory_reference_kl,
            reference_kl=reference_kl,
            advantages=advantages,
            new_mode_log_prob=new_mode_log_prob,
            new_trajectory_log_prob=new_trajectory_log_prob,
            mode_importance_ratio=mode_ratio,
            trajectory_importance_ratio=trajectory_ratio,
            mode_clip_fraction_low=mode_clip_fraction_low,
            mode_clip_fraction_high=mode_clip_fraction_high,
            trajectory_clip_fraction_low=trajectory_clip_fraction_low,
            trajectory_clip_fraction_high=trajectory_clip_fraction_high,
            mode_old_policy_approx_kl=mode_old_policy_approx_kl,
            trajectory_old_policy_approx_kl=trajectory_old_policy_approx_kl,
        )

    @staticmethod
    def _module_gradient_norm(module: nn.Module) -> float:
        values = []
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all()):
                raise JointGRPOError("joint GRPO gradient is non-finite")
            values.append(gradient.square().sum())
        if not values:
            return 0.0
        return float(torch.stack(values).sum().sqrt().cpu())

    @staticmethod
    def _parameter_gradient_norm(parameter: Tensor) -> float:
        if parameter.grad is None:
            return 0.0
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise JointGRPOError("joint GRPO gradient is non-finite")
        return float(torch.linalg.vector_norm(gradient).cpu())

    def _extra_gradient_norms(self) -> dict[str, float]:
        return {}

    def update(
        self,
        rollout: JointGRPORollout,
        rewards: Tensor,
        *,
        policy_update: JointGRPOPolicyUpdateConfig | None = None,
    ) -> JointGRPOUpdateResult:
        update_config = (
            policy_update
            if policy_update is not None
            else JointGRPOPolicyUpdateConfig(update_epochs=1)
        )
        if not isinstance(update_config, JointGRPOPolicyUpdateConfig):
            raise JointGRPOError(
                "policy_update must be a JointGRPOPolicyUpdateConfig"
            )
        rollout_id = id(rollout)
        if self._consumed_rollouts.get(rollout_id) is rollout:
            raise JointGRPOError(
                "each live joint rollout may enter update exactly once"
            )
        advantages = normalize_signed_advantages(
            rewards,
            group_size=self.config.group_size,
            eps=self.config.advantage_eps,
        ).to(device=rollout.old_mode_log_prob.device).detach()
        self.optimizer.zero_grad(set_to_none=True)
        first_loss = self._compute_loss_from_advantages(
            rollout,
            advantages,
            clip_epsilon=update_config.clip_epsilon,
        )
        # Register consumption before the first optimizer mutation.  A failed
        # later epoch must never make a partially-updated rollout reusable.
        self._consumed_rollouts[rollout_id] = rollout
        trainable = [
            *self.planner.diffusion_decoder.parameters(),
            *self.planner.mode_head.parameters(),
        ]
        epoch_results = []
        for epoch_index in range(update_config.update_epochs):
            if epoch_index == 0:
                loss = first_loss
                del first_loss
            else:
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._compute_loss_from_advantages(
                    rollout,
                    advantages,
                    clip_epsilon=update_config.clip_epsilon,
                )
            loss.total.backward()
            gradient_norms = {
                "diffusion_decoder": self._module_gradient_norm(
                    self.planner.diffusion_decoder
                ),
                "mode_head": self._module_gradient_norm(
                    self.planner.mode_head
                ),
            }
            gradient_norms.update(self._extra_gradient_norms())
            if any(
                not math.isfinite(value) for value in gradient_norms.values()
            ):
                raise JointGRPOError("joint GRPO gradients must be finite")
            if any(
                gradient_norms[name] <= 0.0
                for name in ("diffusion_decoder", "mode_head")
            ):
                raise JointGRPOError(
                    "mode head and diffusion decoder require finite non-zero "
                    "gradients"
                )
            total_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(self.config.max_grad_norm)
            )
            if not bool(torch.isfinite(total_norm)):
                raise JointGRPOError(
                    "joint GRPO total gradient norm is non-finite"
                )
            self.optimizer.step()
            self.optimizer_step += 1
            epoch_results.append(
                JointGRPOEpochUpdateResult(
                    epoch_in_rollout=epoch_index + 1,
                    loss=loss.detached(),
                    gradient_norms=gradient_norms,
                    total_gradient_norm=float(total_norm.detach().cpu()),
                    optimizer_step=self.optimizer_step,
                )
            )
            del loss
        frozen_epoch_results = tuple(epoch_results)
        final_epoch = frozen_epoch_results[-1]
        return JointGRPOUpdateResult(
            loss=final_epoch.loss,
            gradient_norms=final_epoch.gradient_norms,
            total_gradient_norm=final_epoch.total_gradient_norm,
            optimizer_step=final_epoch.optimizer_step,
            epoch_results=frozen_epoch_results,
        )


class JointGRPOTrainerA(_JointGRPOTrainerBase):
    """Variant-A joint GRPO with vectorized independent role decoding."""

    variant = "A"
    predecessor_condition = "none"


class JointGRPOTrainerB(_JointGRPOTrainerBase):
    """Variant-B joint GRPO with fixed predicted-detached action history."""

    variant = "B"
    predecessor_condition = "predicted_detached"

    def _decode_roles(
        self,
        *,
        decoder: nn.Module,
        mode_head: nn.Module,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        fixed_history: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = context.batch_size
        if fixed_history is not None:
            expected = (
                batch_size,
                NUM_PLATOON_ROLES - 1,
                TRAJECTORY_STEPS,
                TRAJECTORY_DIM,
            )
            if (
                fixed_history.dtype != torch.float32
                or tuple(fixed_history.shape) != expected
                or fixed_history.device != context.role_tokens.device
                or not bool(torch.isfinite(fixed_history).all())
                or fixed_history.requires_grad
            ):
                raise JointGRPOError(
                    "fixed predecessor history must be detached float32 "
                    "with shape [N,2,8,3]"
                )
        noisy_metric = self.planner._denormalize_xy(
            sample.clamp(-1.0, 1.0)
        )
        role_candidates = []
        role_logits = []
        generated_history = []
        predecessor_action: Tensor | None = None
        for role_index in range(NUM_PLATOON_ROLES):
            if role_index > 0 and fixed_history is not None:
                predecessor_action = fixed_history[:, role_index - 1]
            candidates, mode_features = decoder.forward_role(
                noisy_metric,
                coarse,
                timesteps,
                context,
                role_index=role_index,
                predecessor_action=predecessor_action,
            )
            logits = mode_head(mode_features).squeeze(-1)
            role_candidates.append(candidates)
            role_logits.append(logits)
            if role_index + 1 < NUM_PLATOON_ROLES and fixed_history is None:
                selected = self.planner._select_predecessor_trajectory(
                    candidates,
                    logits,
                    valid_mask[:, role_index],
                )
                predecessor_action = (
                    self.planner._normalize_predecessor_action(selected)
                ).float()
                generated_history.append(predecessor_action)
        history = (
            fixed_history
            if fixed_history is not None
            else torch.stack(generated_history, dim=1).detach()
        )
        return (
            torch.stack(role_candidates, dim=1),
            torch.stack(role_logits, dim=1),
            history,
        )

    def _rollout_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self._decode_roles(
            decoder=self.planner.diffusion_decoder,
            mode_head=self.planner.mode_head,
            sample=sample,
            timesteps=timesteps,
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            fixed_history=None,
        )

    def _frozen_reference_prediction(
        self,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidates, logits, _ = self._decode_roles(
            decoder=self.reference.diffusion_decoder,
            mode_head=self.reference.mode_head,
            sample=sample,
            timesteps=timesteps,
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            fixed_history=None,
        )
        return candidates, logits

    def _make_rollout(
        self,
        *,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        chains: Tensor,
        modes: Tensor,
        selected: Tensor,
        mode_log_prob: Tensor,
        trajectory_log_prob: Tensor,
        step_histories: list[Tensor],
    ) -> JointGRPORolloutB:
        if len(step_histories) != len(self.config.roll_timesteps):
            raise JointGRPOError("variant B rollout action history count mismatch")
        batch_size, groups = modes.shape[:2]
        history = torch.stack(step_histories, dim=1).reshape(
            batch_size,
            groups,
            len(self.config.roll_timesteps),
            NUM_PLATOON_ROLES - 1,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        )
        if history.dtype != torch.float32 or not bool(
            torch.isfinite(history).all()
        ):
            raise JointGRPOError(
                "variant B rollout action history must be finite float32"
            )
        return JointGRPORolloutB(
            context=BEVPlannerContext(
                bev_feature=context.bev_feature.detach(),
                role_tokens=context.role_tokens.detach(),
            ),
            coarse_trajectories=coarse.detach(),
            mode_valid_mask=valid_mask.detach(),
            chains_normalized=chains.detach(),
            sampled_modes=modes.detach(),
            selected_trajectories=selected.detach(),
            old_mode_log_prob=mode_log_prob.detach(),
            old_trajectory_log_prob=trajectory_log_prob.detach(),
            predecessor_action_history_normalized=history.detach(),
        )

    def _replay_history(
        self,
        rollout: JointGRPORollout,
        *,
        step_index: int,
        flat_count: int,
    ) -> Tensor:
        if not isinstance(rollout, JointGRPORolloutB):
            raise JointGRPOError("variant B requires JointGRPORolloutB")
        expected = (
            rollout.batch_size,
            rollout.group_size,
            len(self.config.roll_timesteps),
            NUM_PLATOON_ROLES - 1,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        )
        history = rollout.predecessor_action_history_normalized
        if (
            history.dtype != torch.float32
            or tuple(history.shape) != expected
            or not bool(torch.isfinite(history).all())
            or history.requires_grad
        ):
            raise JointGRPOError("variant B rollout action history is invalid")
        return history.reshape(
            flat_count,
            len(self.config.roll_timesteps),
            NUM_PLATOON_ROLES - 1,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        )[:, step_index]

    def _replay_predictions(
        self,
        *,
        sample: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        predecessor_history: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if predecessor_history is None:
            raise JointGRPOError("variant B replay requires action history")
        current_candidates, current_logits, _ = self._decode_roles(
            decoder=self.planner.diffusion_decoder,
            mode_head=self.planner.mode_head,
            sample=sample,
            timesteps=timesteps,
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            fixed_history=predecessor_history,
        )
        with torch.no_grad():
            reference_candidates, reference_logits, _ = self._decode_roles(
                decoder=self.reference.diffusion_decoder,
                mode_head=self.reference.mode_head,
                sample=sample,
                timesteps=timesteps,
                context=context,
                coarse=coarse,
                valid_mask=valid_mask,
                fixed_history=predecessor_history,
            )
        return (
            current_candidates,
            current_logits,
            reference_candidates,
            reference_logits,
        )

    def _extra_gradient_norms(self) -> dict[str, float]:
        encoder = self.planner.diffusion_decoder.predecessor_action_encoder
        gate = self.planner.diffusion_decoder.predecessor_residual_gate
        if encoder is None or gate is None:
            raise JointGRPOError("variant B condition modules are missing")
        return {
            "predecessor_action_encoder": self._module_gradient_norm(encoder),
            "predecessor_residual_gate": self._parameter_gradient_norm(gate),
        }


__all__ = [
    "FrozenGRPOReference",
    "FrozenVariantAReference",
    "GaussianDDIMStep",
    "JointGRPOConfig",
    "JointGRPOEpochUpdateResult",
    "JointGRPOError",
    "JointGRPOLossResult",
    "JointGRPOPolicyUpdateConfig",
    "JointGRPORollout",
    "JointGRPORolloutB",
    "JointGRPOTrainerA",
    "JointGRPOTrainerB",
    "JointGRPOUpdateResult",
    "StandardGaussianDDIM",
    "joint_grpo_optimizer_contract",
    "joint_grpo_optimizer_contract_sha256",
    "normalize_signed_advantages",
]
