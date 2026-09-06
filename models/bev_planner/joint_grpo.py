"""Per-vehicle, same-mode GRPO core for the BEV-only diffusion planner."""

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
from torch import Tensor, nn
from torch.optim import AdamW

from models.bev_planner.bev_only_diffusion_planner import (
    BEVOnlyDiffusionPlanner,
    BEVPlannerContext,
    NUM_PLATOON_ROLES,
    TRAJECTORY_DIM,
)
from models.bev_planner.mode_contract import NUM_MODES, TRAJECTORY_STEPS
from models.bev_planner.ddim_transition import (
    DDIMNoiseBundle,
    DEFAULT_DDIM_PATH,
    StandardGaussianDDIM,
)


class JointGRPOError(RuntimeError):
    """Raised when the strict joint GRPO contract is violated."""


@dataclass(frozen=True)
class JointGRPOConfig:
    trajectories_per_mode: int = 48
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
            isinstance(self.trajectories_per_mode, bool)
            or not isinstance(self.trajectories_per_mode, int)
            or self.trajectories_per_mode < 2
        ):
            raise JointGRPOError(
                "trajectories_per_mode must be an integer greater than or equal to 2"
            )
        positive = (
            "learning_rate",
            "max_grad_norm",
            "advantage_eps",
            "xy_beta_m",
            "heading_beta_rad",
        )
        non_negative = (
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
        return DEFAULT_DDIM_PATH.timesteps

    @property
    def stochastic_timesteps(self) -> tuple[int, ...]:
        return self.roll_timesteps[:-1]


@dataclass(frozen=True)
class JointGRPOPolicyUpdateConfig:
    """PPO-style update semantics applied to one frozen joint rollout."""

    update_epochs: int = 10
    clip_epsilon_low: float = 0.1
    clip_epsilon_high: float = 0.2

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_epochs, bool)
            or not isinstance(self.update_epochs, int)
            or self.update_epochs < 1
        ):
            raise JointGRPOError("update_epochs must be a positive integer")
        for name in ("clip_epsilon_low", "clip_epsilon_high"):
            clip_epsilon = float(getattr(self, name))
            if (
                not math.isfinite(clip_epsilon)
                or clip_epsilon <= 0.0
                or clip_epsilon >= 1.0
            ):
                raise JointGRPOError(f"{name} must be finite and in (0,1)")


def joint_grpo_optimizer_contract(
    config: JointGRPOPolicyUpdateConfig | None = None,
) -> dict[str, object]:
    """Return the machine-readable Stage-2 clipped-GRPO optimizer contract."""

    policy_update = config if config is not None else JointGRPOPolicyUpdateConfig()
    return {
        "version": "stage2_joint_grpo_optimizer_v7",
        "ddim_path": DEFAULT_DDIM_PATH.as_dict(),
        "behavior_policy_snapshot": (
            "detached per-vehicle, per-mode stochastic DDIM-transition log "
            "probabilities captured before any policy update"
        ),
        "update_epochs": int(policy_update.update_epochs),
        "clip_epsilon_low": float(policy_update.clip_epsilon_low),
        "clip_epsilon_high": float(policy_update.clip_epsilon_high),
        "trajectory_ratio_factorization": (
            "one ratio per vehicle, mode, trajectory and stochastic DDIM "
            "transition [B,3,10,N,S]"
        ),
        "clipped_surrogate": (
            "negative mean of min(ratio*advantage, "
            "clip(ratio,1-epsilon_low,1+epsilon_high)*advantage)"
        ),
        "activation_gate": (
            "active = hard_valid_and_executable AND "
            "mean_j(reward[b,r,k,j]) >= frozen_pretrain_reward[b,r,k]"
        ),
        "advantage_normalization": (
            "independently for every [B,vehicle,mode] block: subtract the "
            "trajectory mean and divide by sqrt(population_variance+epsilon)"
        ),
        "advantage_reuse": (
            "same-mode normalize once, detach, and freeze across "
            "all epochs of the accepted rollout"
        ),
        "rollout_consumption": (
            "one external update call consumes one accepted live rollout and "
            "performs exactly update_epochs serial optimizer steps"
        ),
        "loss_reduction": (
            "mean over trajectories*DDIM steps, then active modes per vehicle, "
            "then vehicles, then batch"
        ),
        "minibatch_semantics": "none; reuse the complete all-mode rollout each epoch",
        "reference_regularization": (
            "recompute trajectory behavior-cloning and frozen-Stage1 trajectory "
            "KL for every hard-valid optimizer-executable mode each epoch; no mode KL"
        ),
        "kl_early_stop": False,
        "budget_unit": "accepted_update_state",
        "checkpoint_boundary": (
            "after a complete rollout and all of its optimizer epochs"
        ),
        "trainable_modules": ["diffusion_decoder.trajectory_head"],
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
class JointGRPORollout:
    context: BEVPlannerContext
    coarse_trajectories: Tensor
    mode_valid_mask: Tensor
    chains_normalized: Tensor
    candidate_trajectories: Tensor
    old_trajectory_log_prob: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.candidate_trajectories.shape[0])

    @property
    def trajectories_per_mode(self) -> int:
        return int(self.candidate_trajectories.shape[3])


@dataclass(frozen=True)
class JointGRPORolloutB(JointGRPORollout):
    predecessor_action_history_normalized: Tensor


@dataclass(frozen=True)
class JointGRPOLossResult:
    total: Tensor
    trajectory_pg: Tensor
    behavior_cloning: Tensor
    trajectory_reference_kl: Tensor
    reference_kl: Tensor
    advantages: Tensor
    active_mode_mask: Tensor
    new_trajectory_log_prob: Tensor
    trajectory_importance_ratio: Tensor
    trajectory_clip_fraction_low: Tensor
    trajectory_clip_fraction_high: Tensor
    trajectory_old_policy_approx_kl: Tensor

    def scalar_metrics(self) -> dict[str, float]:
        values = {
            "loss/total": self.total,
            "loss/trajectory_pg": self.trajectory_pg,
            "loss/behavior_cloning": self.behavior_cloning,
            "loss/trajectory_reference_kl": self.trajectory_reference_kl,
            "loss/reference_kl": self.reference_kl,
            "advantage/mean": _active_tensor_mean(
                self.advantages, self.active_mode_mask
            ),
            "advantage/rms": torch.sqrt(
                _active_tensor_mean(self.advantages.square(), self.active_mode_mask)
            ),
            "active_mode/count": self.active_mode_mask.float().sum(),
            "policy/trajectory_ratio_mean": (
                _active_tensor_mean(
                    self.trajectory_importance_ratio, self.active_mode_mask
                )
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
            "policy/trajectory_old_policy_approx_kl": (
                self.trajectory_old_policy_approx_kl
            ),
        }
        return {name: float(value.detach().cpu()) for name, value in values.items()}

    def detached(self) -> JointGRPOLossResult:
        """Return a graph-free result safe to retain across optimizer epochs."""

        return JointGRPOLossResult(
            total=self.total.detach(),
            trajectory_pg=self.trajectory_pg.detach(),
            behavior_cloning=self.behavior_cloning.detach(),
            trajectory_reference_kl=self.trajectory_reference_kl.detach(),
            reference_kl=self.reference_kl.detach(),
            advantages=self.advantages.detach(),
            active_mode_mask=self.active_mode_mask.detach(),
            new_trajectory_log_prob=self.new_trajectory_log_prob.detach(),
            trajectory_importance_ratio=(
                self.trajectory_importance_ratio.detach()
            ),
            trajectory_clip_fraction_low=(
                self.trajectory_clip_fraction_low.detach()
            ),
            trajectory_clip_fraction_high=(
                self.trajectory_clip_fraction_high.detach()
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
    clipped_gradient_norms: Mapping[str, float]
    total_gradient_norm: float
    trajectory_head_relative_drift: float
    optimizer_step: int
    zero_signal_epoch: bool


@dataclass(frozen=True)
class JointGRPOUpdateResult:
    loss: JointGRPOLossResult
    gradient_norms: Mapping[str, float]
    clipped_gradient_norms: Mapping[str, float]
    total_gradient_norm: float
    trajectory_head_relative_drift: float
    optimizer_step: int
    epoch_results: tuple[JointGRPOEpochUpdateResult, ...]
    zero_signal_epochs: int


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


def same_mode_active_mask(
    rewards: Tensor,
    pretrain_rewards: Tensor,
    *,
    valid_mode_mask: Tensor,
) -> Tensor:
    """Apply the strict same-mode frozen-pretrain acceptance gate."""

    if not isinstance(rewards, Tensor) or rewards.dtype != torch.float32:
        raise JointGRPOError("rewards must be a float32 torch.Tensor")
    if rewards.ndim != 4 or tuple(rewards.shape[1:3]) != (
        NUM_PLATOON_ROLES,
        NUM_MODES,
    ):
        raise JointGRPOError("rewards must have shape [B,3,10,N]")
    if int(rewards.shape[0]) <= 0 or int(rewards.shape[3]) < 2:
        raise JointGRPOError("rewards must contain a non-empty trajectory group")
    if not bool(torch.isfinite(rewards).all()):
        raise JointGRPOError("rewards must contain finite values")
    expected = tuple(rewards.shape[:3])
    if (
        not isinstance(pretrain_rewards, Tensor)
        or pretrain_rewards.dtype != torch.float32
        or tuple(pretrain_rewards.shape) != expected
    ):
        raise JointGRPOError("pretrain_rewards must be float32 with shape [B,3,10]")
    if not bool(torch.isfinite(pretrain_rewards).all()):
        raise JointGRPOError("pretrain_rewards must contain finite values")
    if (
        not isinstance(valid_mode_mask, Tensor)
        or valid_mode_mask.dtype != torch.bool
        or tuple(valid_mode_mask.shape) != expected
    ):
        raise JointGRPOError("valid_mode_mask must be bool with shape [B,3,10]")
    if not (
        rewards.device == pretrain_rewards.device == valid_mode_mask.device
    ):
        raise JointGRPOError("gate tensors must use the same device")
    return valid_mode_mask & (rewards.mean(dim=-1) >= pretrain_rewards)


def normalize_same_mode_advantages(
    rewards: Tensor,
    active_mode_mask: Tensor,
    *,
    trajectories_per_mode: int | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Center and population-standardize each vehicle-mode trajectory group."""

    if not isinstance(rewards, Tensor) or rewards.dtype != torch.float32:
        raise JointGRPOError("rewards must be a float32 torch.Tensor")
    if rewards.ndim != 4 or tuple(rewards.shape[1:3]) != (
        NUM_PLATOON_ROLES,
        NUM_MODES,
    ):
        raise JointGRPOError("rewards must have shape [B,3,10,N]")
    count = int(rewards.shape[-1])
    if int(rewards.shape[0]) <= 0 or count < 2:
        raise JointGRPOError("rewards must contain a non-empty trajectory group")
    if trajectories_per_mode is not None and (
        isinstance(trajectories_per_mode, bool)
        or not isinstance(trajectories_per_mode, int)
        or trajectories_per_mode != count
    ):
        raise JointGRPOError(
            "trajectories_per_mode must match the rewards trajectory axis"
        )
    if not bool(torch.isfinite(rewards).all()):
        raise JointGRPOError("rewards must contain finite values")
    if (
        not isinstance(active_mode_mask, Tensor)
        or active_mode_mask.dtype != torch.bool
        or tuple(active_mode_mask.shape) != tuple(rewards.shape[:3])
    ):
        raise JointGRPOError(
            "active_mode_mask must be bool with shape [B,3,10]"
        )
    if active_mode_mask.device != rewards.device:
        raise JointGRPOError("rewards and active_mode_mask must use the same device")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise JointGRPOError("advantage eps must be positive and finite")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + epsilon)
    normalized = centered / scale
    return normalized * active_mode_mask.unsqueeze(-1).to(normalized.dtype)


def _active_tensor_mean(value: Tensor, active_mode_mask: Tensor) -> Tensor:
    """Mean all trailing entries belonging to active [B,R,K] blocks."""

    expanded = active_mode_mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    weights = expanded.expand_as(value).to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _hierarchical_active_mean(value: Tensor, active_mode_mask: Tensor) -> Tensor:
    """Reduce trailing axes, active modes, roles and batch in that order."""

    if tuple(value.shape[:3]) != tuple(active_mode_mask.shape):
        raise JointGRPOError("loss blocks and active_mode_mask do not align")
    block_mean = value.reshape(*value.shape[:3], -1).mean(dim=-1)
    weights = active_mode_mask.to(block_mean.dtype)
    per_role = (block_mean * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    return per_role.mean(dim=1).mean(dim=0)


def _clipped_grpo_surrogate(
    importance_ratio: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
) -> Tensor:
    """Return the negative PPO clipped surrogate for aligned components."""

    return -torch.minimum(
        importance_ratio * advantages,
        importance_ratio.clamp(
            1.0 - float(clip_epsilon_low),
            1.0 + float(clip_epsilon_high),
        )
        * advantages,
    ).mean()


def _clipped_grpo_terms(
    importance_ratio: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
) -> Tensor:
    unclipped = importance_ratio * advantages
    clipped = importance_ratio.clamp(
        1.0 - float(clip_epsilon_low),
        1.0 + float(clip_epsilon_high),
    ) * advantages
    return -torch.minimum(unclipped, clipped)


def _importance_ratio_diagnostics(
    log_ratio: Tensor,
    importance_ratio: Tensor,
    *,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
) -> tuple[Tensor, Tensor, Tensor]:
    detached_log_ratio = log_ratio.detach()
    detached_ratio = importance_ratio.detach()
    low = (detached_ratio < 1.0 - float(clip_epsilon_low)).float().mean()
    high = (detached_ratio > 1.0 + float(clip_epsilon_high)).float().mean()
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


def _trajectory_bc(current: Tensor, reference: Tensor, config: JointGRPOConfig) -> Tensor:
    return _trajectory_bc_blocks(current, reference, config).mean()


def _trajectory_bc_blocks(
    current: Tensor, reference: Tensor, config: JointGRPOConfig
) -> Tensor:
    """Return one BC value for every leading trajectory block."""

    xy = F.smooth_l1_loss(
        current[..., :2],
        reference[..., :2],
        beta=float(config.xy_beta_m),
        reduction="none",
    ).mean(dim=(-2, -1))
    heading_error = torch.atan2(
        torch.sin(current[..., 2] - reference[..., 2]),
        torch.cos(current[..., 2] - reference[..., 2]),
    )
    heading = F.smooth_l1_loss(
        heading_error,
        torch.zeros_like(heading_error),
        beta=float(config.heading_beta_rad),
        reduction="none",
    ).mean(dim=-1)
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
        for parameter in planner.diffusion_decoder.trajectory_head.parameters():
            parameter.requires_grad_(True)
        self.reference = FrozenGRPOReference(planner).to(
            next(planner.parameters()).device
        )
        self.planner.eval()
        self.optimizer = AdamW(
            self.planner.diffusion_decoder.trajectory_head.parameters(),
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
        *,
        noise_bundle: DDIMNoiseBundle | None = None,
    ) -> dict[str, Tensor]:
        """Run one deterministic Stage-1 policy inference on a rollout state."""

        if not isinstance(rollout, JointGRPORollout):
            raise JointGRPOError("frozen pretrain inference requires a joint rollout")
        return self._infer_frozen_pretrain_context(
            rollout.context,
            rollout.coarse_trajectories,
            rollout.mode_valid_mask,
            noise_bundle=noise_bundle,
        )

    @torch.inference_mode()
    def infer_frozen_pretrain_from_inputs(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        noise_bundle: DDIMNoiseBundle | None = None,
    ) -> dict[str, Tensor]:
        """Run deterministic Stage-1 inference without allocating an N-rollout."""

        context = self._context_from_inputs(model_inputs)
        return self._infer_frozen_pretrain_context(
            context,
            model_inputs["coarse_trajectories"],
            model_inputs["mode_valid_mask"],
            noise_bundle=noise_bundle,
        )

    def _infer_frozen_pretrain_context(
        self,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        *,
        noise_bundle: DDIMNoiseBundle | None,
    ) -> dict[str, Tensor]:
        self.planner._validate_trajectory_inputs(
            coarse,
            valid_mask,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )

        batch_size = context.batch_size
        device = coarse.device
        anchor_normalized = self.planner._normalize_xy(coarse[..., :2])
        bundle = (
            self.planner._inference_noise_bundle(
                anchor_normalized.shape,
                device=device,
            )
            if noise_bundle is None
            else noise_bundle
        )
        bundle.validate(anchor_normalized.shape, device=device)
        noise_timesteps = torch.full(
            (batch_size * NUM_PLATOON_ROLES,),
            DEFAULT_DDIM_PATH.initial_timestep,
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
            bundle.initial_noise.reshape(flat_shape),
            noise_timesteps,
        ).reshape_as(anchor_normalized)

        candidates: Tensor | None = None
        raw_logits: Tensor | None = None
        transition_noise_index = 0
        for timestep, previous_timestep in DEFAULT_DDIM_PATH.transitions():
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
            step_noise = None
            if previous_timestep >= 0:
                step_noise = bundle.transition_noises[transition_noise_index]
                transition_noise_index += 1
            transition = self.transition.step(
                model_output=predicted_normalized.reshape(flat_shape),
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=sample.reshape(flat_shape),
                eta=DEFAULT_DDIM_PATH.eta,
                noise=(
                    step_noise.reshape(flat_shape)
                    if step_noise is not None
                    else None
                ),
            )
            sample = transition.prev_sample.reshape_as(sample)
        if candidates is None or raw_logits is None:
            raise JointGRPOError(
                "frozen pretrain inference produced no denoising steps"
            )
        selected = self.planner._select(candidates, raw_logits, valid_mask)
        return {
            "selected_trajectory": selected["selected_trajectory"],
            "selected_mode": selected["selected_mode"],
            "all_mode_trajectories": candidates,
            "mode_logits": raw_logits,
        }

    def _make_rollout(
        self,
        *,
        context: BEVPlannerContext,
        coarse: Tensor,
        valid_mask: Tensor,
        chains: Tensor,
        candidates: Tensor,
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
            candidate_trajectories=candidates.detach(),
            old_trajectory_log_prob=trajectory_log_prob.detach(),
        )

    @torch.no_grad()
    def sample_groups(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        generator: torch.Generator,
        transition_generator: torch.Generator | None = None,
    ) -> JointGRPORollout:
        if not isinstance(generator, torch.Generator):
            raise JointGRPOError("sample_groups requires an explicit torch.Generator")
        if transition_generator is None:
            transition_generator = generator
        if not isinstance(transition_generator, torch.Generator):
            raise JointGRPOError(
                "sample_groups transition_generator must be a torch.Generator"
            )
        context = self._context_from_inputs(model_inputs)
        coarse = model_inputs["coarse_trajectories"]
        valid_mask = model_inputs["mode_valid_mask"]
        self.planner._validate_trajectory_inputs(
            coarse,
            valid_mask,
            batch_size=context.batch_size,
            device=context.role_tokens.device,
        )
        trajectories = self.config.trajectories_per_mode
        repeated_context = _repeat_context(context, trajectories)
        repeated_coarse = _repeat_groups(coarse, trajectories)
        repeated_mask = _repeat_groups(valid_mask, trajectories)
        anchor = self.planner._normalize_xy(repeated_coarse[..., :2])
        noise = torch.randn(
            anchor.shape,
            device=anchor.device,
            dtype=torch.float32,
            generator=generator,
        )
        noise_timestep = torch.full(
            (anchor.shape[0] * NUM_PLATOON_ROLES,),
            DEFAULT_DDIM_PATH.initial_timestep,
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
        for timestep, previous_timestep in DEFAULT_DDIM_PATH.transitions():
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
                eta=DEFAULT_DDIM_PATH.eta,
                generator=transition_generator,
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
        batch_size = context.batch_size
        chain_tensor = torch.stack(chains, dim=1).reshape(
            batch_size,
            trajectories,
            len(chains),
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        candidate_tensor = final_candidates.reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ).permute(0, 2, 3, 1, 4, 5).contiguous()
        trajectory_log_prob = torch.stack(
            stochastic_log_probs,
            dim=-1,
        ).reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            len(self.config.stochastic_timesteps),
        ).permute(0, 2, 3, 1, 4).contiguous()
        return self._make_rollout(
            context=context,
            coarse=coarse,
            valid_mask=valid_mask,
            chains=chain_tensor,
            candidates=candidate_tensor,
            trajectory_log_prob=trajectory_log_prob,
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
        active_mode_mask: Tensor,
        *,
        clip_epsilon_low: float = 0.1,
        clip_epsilon_high: float = 0.2,
    ) -> JointGRPOLossResult:
        policy_update = JointGRPOPolicyUpdateConfig(
            update_epochs=1,
            clip_epsilon_low=clip_epsilon_low,
            clip_epsilon_high=clip_epsilon_high,
        )
        advantages = normalize_same_mode_advantages(
            rewards,
            active_mode_mask,
            trajectories_per_mode=self.config.trajectories_per_mode,
            eps=self.config.advantage_eps,
        ).to(device=rollout.old_trajectory_log_prob.device).detach()
        active_mode_mask = active_mode_mask.to(
            device=rollout.old_trajectory_log_prob.device
        )
        return self._compute_loss_from_advantages(
            rollout,
            advantages,
            active_mode_mask,
            clip_epsilon_low=policy_update.clip_epsilon_low,
            clip_epsilon_high=policy_update.clip_epsilon_high,
        )

    def _compute_loss_from_advantages(
        self,
        rollout: JointGRPORollout,
        advantages: Tensor,
        active_mode_mask: Tensor,
        *,
        clip_epsilon_low: float,
        clip_epsilon_high: float,
    ) -> JointGRPOLossResult:
        batch_size = rollout.batch_size
        trajectories = rollout.trajectories_per_mode
        expected_advantage = (
            batch_size,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            trajectories,
        )
        if tuple(advantages.shape) != expected_advantage:
            raise JointGRPOError("reward batch does not match rollout")
        expected_mask = expected_advantage[:3]
        if (
            active_mode_mask.dtype != torch.bool
            or tuple(active_mode_mask.shape) != expected_mask
            or active_mode_mask.device != advantages.device
        ):
            raise JointGRPOError(
                "active_mode_mask must be colocated bool with shape [B,3,10]"
            )
        valid_mode_mask = rollout.mode_valid_mask.to(active_mode_mask.device)
        if bool((active_mode_mask & ~valid_mode_mask).any()):
            raise JointGRPOError("active modes must be a subset of hard-valid modes")
        stochastic_steps = len(self.config.stochastic_timesteps)
        if tuple(rollout.old_trajectory_log_prob.shape) != (
            *expected_advantage,
            stochastic_steps,
        ):
            raise JointGRPOError("rollout trajectory log-probability shape is invalid")
        if tuple(rollout.candidate_trajectories.shape) != (
            *expected_advantage,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ):
            raise JointGRPOError("rollout candidate trajectory shape is invalid")
        flat_count = batch_size * trajectories
        context = _repeat_context(rollout.context, trajectories)
        coarse = _repeat_groups(rollout.coarse_trajectories, trajectories)
        valid_mask = _repeat_groups(rollout.mode_valid_mask, trajectories)
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
        for step_index, (timestep, previous_timestep) in enumerate(
            DEFAULT_DDIM_PATH.transitions()
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
                _,
                reference_candidates,
                _,
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
                eta=DEFAULT_DDIM_PATH.eta,
                prev_sample=next_sample.float(),
            )
            with torch.no_grad():
                reference_transition = self.transition.step(
                    model_output=reference_output,
                    timestep=timestep,
                    previous_timestep=previous_timestep,
                    sample=sample.float(),
                    eta=DEFAULT_DDIM_PATH.eta,
                    prev_sample=next_sample.float(),
            )
            if current_transition.log_prob is not None:
                current_log_probs.append(current_transition.log_prob)
                sigma_squared = current_transition.std.square()
                trajectory_kls.append(
                    (
                        (
                            current_transition.mean
                            - reference_transition.mean
                        ).square()
                        / (2.0 * sigma_squared)
                    ).mean(dim=(-2, -1))
                )
            final_current_candidates = current_candidates
            final_reference_candidates = reference_candidates

        if (
            final_current_candidates is None
            or final_reference_candidates is None
        ):
            raise JointGRPOError("joint replay produced no decoder output")
        new_trajectory_log_prob = torch.stack(
            current_log_probs, dim=-1
        ).reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            stochastic_steps,
        ).permute(0, 2, 3, 1, 4).contiguous()
        trajectory_log_ratio = (
            new_trajectory_log_prob - rollout.old_trajectory_log_prob
        )
        trajectory_ratio = torch.exp(trajectory_log_ratio)
        trajectory_pg = _hierarchical_active_mean(
            _clipped_grpo_terms(
                trajectory_ratio,
                advantages.unsqueeze(-1).expand_as(trajectory_ratio),
                clip_epsilon_low=clip_epsilon_low,
                clip_epsilon_high=clip_epsilon_high,
            ),
            active_mode_mask,
        )
        detached_ratio = trajectory_ratio.detach()
        detached_log_ratio = trajectory_log_ratio.detach()
        trajectory_clip_fraction_low = _hierarchical_active_mean(
            (detached_ratio < 1.0 - float(clip_epsilon_low)).float(),
            active_mode_mask,
        )
        trajectory_clip_fraction_high = _hierarchical_active_mean(
            (detached_ratio > 1.0 + float(clip_epsilon_high)).float(),
            active_mode_mask,
        )
        trajectory_old_policy_approx_kl = _hierarchical_active_mean(
            detached_ratio - 1.0 - detached_log_ratio,
            active_mode_mask,
        ).clamp_min(0.0)

        current_all_modes = final_current_candidates.reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ).permute(0, 2, 3, 1, 4, 5)
        reference_all_modes = final_reference_candidates.reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        ).permute(0, 2, 3, 1, 4, 5)
        behavior_cloning = _hierarchical_active_mean(
            _trajectory_bc_blocks(
                current_all_modes,
                reference_all_modes,
                self.config,
            ),
            valid_mode_mask,
        )
        trajectory_kl_blocks = torch.stack(
            trajectory_kls, dim=-1
        ).reshape(
            batch_size,
            trajectories,
            NUM_PLATOON_ROLES,
            NUM_MODES,
            stochastic_steps,
        ).permute(0, 2, 3, 1, 4)
        trajectory_reference_kl = _hierarchical_active_mean(
            trajectory_kl_blocks,
            valid_mode_mask,
        )
        reference_kl = trajectory_reference_kl
        total = (
            float(self.config.trajectory_pg_weight) * trajectory_pg
            + float(self.config.bc_weight) * behavior_cloning
            + float(self.config.reference_kl_weight) * reference_kl
        )
        tensors = (
            total,
            trajectory_pg,
            behavior_cloning,
            trajectory_reference_kl,
            reference_kl,
            new_trajectory_log_prob,
            trajectory_ratio,
            trajectory_old_policy_approx_kl,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise JointGRPOError("joint GRPO loss contains non-finite values")
        if bool((trajectory_reference_kl < -1e-7).item()):
            raise JointGRPOError("reference KL must be non-negative")
        return JointGRPOLossResult(
            total=total,
            trajectory_pg=trajectory_pg,
            behavior_cloning=behavior_cloning,
            trajectory_reference_kl=trajectory_reference_kl,
            reference_kl=reference_kl,
            advantages=advantages,
            active_mode_mask=active_mode_mask,
            new_trajectory_log_prob=new_trajectory_log_prob,
            trajectory_importance_ratio=trajectory_ratio,
            trajectory_clip_fraction_low=trajectory_clip_fraction_low,
            trajectory_clip_fraction_high=trajectory_clip_fraction_high,
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

    def _trajectory_head_relative_drift(self) -> float:
        numerator: list[Tensor] = []
        denominator: list[Tensor] = []
        current = self.planner.diffusion_decoder.trajectory_head.parameters()
        frozen = self.reference.diffusion_decoder.trajectory_head.parameters()
        for current_parameter, frozen_parameter in zip(current, frozen):
            current_value = current_parameter.detach().float()
            frozen_value = frozen_parameter.detach().float()
            numerator.append((current_value - frozen_value).square().sum())
            denominator.append(frozen_value.square().sum())
        numerator_norm = torch.stack(numerator).sum().sqrt()
        denominator_norm = torch.stack(denominator).sum().sqrt().clamp_min(1e-12)
        return float((numerator_norm / denominator_norm).cpu())

    def update(
        self,
        rollout: JointGRPORollout,
        rewards: Tensor,
        active_mode_mask: Tensor,
        *,
        policy_update: JointGRPOPolicyUpdateConfig | None = None,
    ) -> JointGRPOUpdateResult:
        update_config = (
            policy_update
            if policy_update is not None
            else JointGRPOPolicyUpdateConfig()
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
        advantages = normalize_same_mode_advantages(
            rewards,
            active_mode_mask,
            trajectories_per_mode=self.config.trajectories_per_mode,
            eps=self.config.advantage_eps,
        ).to(device=rollout.old_trajectory_log_prob.device).detach()
        active_mode_mask = active_mode_mask.to(
            device=rollout.old_trajectory_log_prob.device
        )
        self.optimizer.zero_grad(set_to_none=True)
        first_loss = self._compute_loss_from_advantages(
            rollout,
            advantages,
            active_mode_mask,
            clip_epsilon_low=update_config.clip_epsilon_low,
            clip_epsilon_high=update_config.clip_epsilon_high,
        )
        # Register consumption before the first optimizer mutation.  A failed
        # later epoch must never make a partially-updated rollout reusable.
        self._consumed_rollouts[rollout_id] = rollout
        trainable = list(
            self.planner.diffusion_decoder.trajectory_head.parameters()
        )
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
                    active_mode_mask,
                    clip_epsilon_low=update_config.clip_epsilon_low,
                    clip_epsilon_high=update_config.clip_epsilon_high,
                )
            loss.total.backward()
            gradient_norms = {
                "trajectory_head": self._module_gradient_norm(
                    self.planner.diffusion_decoder.trajectory_head
                ),
            }
            if any(
                not math.isfinite(value) for value in gradient_norms.values()
            ):
                raise JointGRPOError("joint GRPO gradients must be finite")
            zero_signal_epoch = gradient_norms["trajectory_head"] == 0.0
            if zero_signal_epoch:
                total_gradient_norm = 0.0
                clipped_gradient_norms = dict(gradient_norms)
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, float(self.config.max_grad_norm)
                )
                if not bool(torch.isfinite(total_norm)):
                    raise JointGRPOError(
                        "joint GRPO total gradient norm is non-finite"
                    )
                total_gradient_norm = float(total_norm.detach().cpu())
                clipped_gradient_norms = {
                    "trajectory_head": self._module_gradient_norm(
                        self.planner.diffusion_decoder.trajectory_head
                    )
                }
                self.optimizer.step()
                self.optimizer_step += 1
            relative_drift = self._trajectory_head_relative_drift()
            epoch_results.append(
                JointGRPOEpochUpdateResult(
                    epoch_in_rollout=epoch_index + 1,
                    loss=loss.detached(),
                    gradient_norms=gradient_norms,
                    clipped_gradient_norms=clipped_gradient_norms,
                    total_gradient_norm=total_gradient_norm,
                    trajectory_head_relative_drift=relative_drift,
                    optimizer_step=self.optimizer_step,
                    zero_signal_epoch=zero_signal_epoch,
                )
            )
            del loss
        frozen_epoch_results = tuple(epoch_results)
        final_epoch = frozen_epoch_results[-1]
        return JointGRPOUpdateResult(
            loss=final_epoch.loss,
            gradient_norms=final_epoch.gradient_norms,
            clipped_gradient_norms=final_epoch.clipped_gradient_norms,
            total_gradient_norm=final_epoch.total_gradient_norm,
            trajectory_head_relative_drift=(
                final_epoch.trajectory_head_relative_drift
            ),
            optimizer_step=final_epoch.optimizer_step,
            epoch_results=frozen_epoch_results,
            zero_signal_epochs=sum(
                int(epoch.zero_signal_epoch) for epoch in frozen_epoch_results
            ),
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
        candidates: Tensor,
        trajectory_log_prob: Tensor,
        step_histories: list[Tensor],
    ) -> JointGRPORolloutB:
        if len(step_histories) != len(self.config.roll_timesteps):
            raise JointGRPOError("variant B rollout action history count mismatch")
        batch_size, trajectories = chains.shape[:2]
        history = torch.stack(step_histories, dim=1).reshape(
            batch_size,
            trajectories,
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
            candidate_trajectories=candidates.detach(),
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
            rollout.trajectories_per_mode,
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

__all__ = [
    "FrozenGRPOReference",
    "FrozenVariantAReference",
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
    "joint_grpo_optimizer_contract",
    "joint_grpo_optimizer_contract_sha256",
    "normalize_same_mode_advantages",
    "same_mode_active_mask",
]
