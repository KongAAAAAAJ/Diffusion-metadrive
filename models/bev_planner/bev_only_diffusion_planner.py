"""Joint-first BEV-only diffusion planner.

The planner consumes only simulator semantic BEV, ego/platoon state, fixed
roles, dynamic coarse trajectories, and the hard mode mask.  Expert labels,
global poses, and legacy sensor/model inputs are deliberately outside this
module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

import torch
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from torch import Tensor, nn

from envs.observations.semantic_bev import SemanticBEVConfig
from models.bev_planner.bev_resnet18_backbone import (
    BEVResNet18Backbone,
    BEVResNet18Config,
)
from models.bev_planner.mode_contract import NUM_MODES, TRAJECTORY_STEPS


NUM_PLATOON_ROLES: Final[int] = 3
EGO_STATE_DIM: Final[int] = 8
RELATION_STATE_DIM: Final[int] = 12
RELATION_NEIGHBORS: Final[int] = 2
TRAJECTORY_DIM: Final[int] = 3
STOP_MODE_INDEX: Final[int] = NUM_MODES - 1
MAX_BACKGROUND_ACTORS: Final[int] = 16
BACKGROUND_ACTOR_STATE_DIM: Final[int] = 8
NUM_SCENARIO_CODES: Final[int] = 6


class BEVPlannerError(RuntimeError):
    """Raised when the strict BEV planner contract is violated."""


@dataclass(frozen=True)
class BEVOnlyDiffusionPlannerConfig:
    """Immutable architecture and diffusion configuration."""

    backbone: BEVResNet18Config = field(default_factory=BEVResNet18Config)
    d_model: int = 128
    num_heads: int = 4
    ffn_dim: int = 256
    decoder_layers: int = 2
    dropout: float = 0.0
    num_train_timesteps: int = 1000
    train_timestep_upper: int = 50
    inference_noise_timestep: int = 8
    inference_denoise_steps: int = 2
    inference_seed: int = 0
    max_xy_residual_m: float = 12.0
    predecessor_condition: Literal["none", "predicted_detached"] = "none"
    model_version: Literal["v2"] = "v2"

    def __post_init__(self) -> None:
        integer_fields = (
            "d_model",
            "num_heads",
            "ffn_dim",
            "decoder_layers",
            "num_train_timesteps",
            "train_timestep_upper",
            "inference_noise_timestep",
            "inference_denoise_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BEVPlannerError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads != 0:
            raise BEVPlannerError("d_model must be divisible by num_heads")
        if not 0 <= self.inference_noise_timestep < self.num_train_timesteps:
            raise BEVPlannerError(
                "inference_noise_timestep must be smaller than num_train_timesteps"
            )
        if self.train_timestep_upper > self.num_train_timesteps:
            raise BEVPlannerError(
                "train_timestep_upper must not exceed num_train_timesteps"
            )
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise BEVPlannerError("dropout must be finite and in [0,1)")
        if not math.isfinite(self.max_xy_residual_m) or self.max_xy_residual_m <= 0.0:
            raise BEVPlannerError("max_xy_residual_m must be positive and finite")
        if self.predecessor_condition not in ("none", "predicted_detached"):
            raise BEVPlannerError(
                "predecessor_condition must be none or predicted_detached"
            )
        if self.model_version != "v2":
            raise BEVPlannerError("model_version must be v2")


@dataclass(frozen=True)
class BEVPlannerContext:
    """Reusable encoded context for Stage 1 inference and later GRPO."""

    bev_feature: Tensor
    role_tokens: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.role_tokens.shape[0])


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding with an MLP projection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise BEVPlannerError("d_model must be even for timestep embedding")
        self.d_model = int(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.d_model // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * exponent
        )
        angles = timesteps.to(dtype=torch.float32).unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.projection(embedding)


class LightweightBEVFusion(nn.Module):
    """Fuse the four ResNet feature scales into one stride-four BEV map."""

    output_stride: Final[int] = 4

    def __init__(
        self,
        input_channels: tuple[int, int, int, int],
        d_model: int,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, d_model, kernel_size=1) for channels in input_channels
        )
        self.output = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        if len(features) != len(self.lateral):
            raise BEVPlannerError("BEV fusion requires exactly four feature scales")
        projected = [layer(value) for layer, value in zip(self.lateral, features)]
        fused = projected[-1]
        for lateral in reversed(projected[:-1]):
            fused = F.interpolate(
                fused,
                size=lateral.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = fused + lateral
        return self.output(fused)


class JointStateRelationEncoder(nn.Module):
    """Encode ego state, masked neighbor relations, role, and BEV context."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "ego_scale",
            torch.tensor(
                [30.0, 8.0, 1.0, 1.0, 1.0, 8.0, math.pi, 30.0],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        # Relation delta-v is currently collected in km/h; all other distances
        # are metres and heading is radians.
        relation_slot_scale = torch.tensor(
            [64.0, 32.0, math.pi, 100.0, 64.0, 32.0],
            dtype=torch.float32,
        )
        self.register_buffer(
            "relation_scale",
            relation_slot_scale.repeat(RELATION_NEIGHBORS),
            persistent=True,
        )
        self.ego_encoder = nn.Sequential(
            nn.Linear(EGO_STATE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(RELATION_STATE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.bev_encoder = nn.Linear(d_model, d_model)
        self.role_embedding = nn.Embedding(NUM_PLATOON_ROLES, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.joint_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.joint_norm1 = nn.LayerNorm(d_model)
        self.joint_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.joint_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "actor_scale",
            torch.tensor(
                [64.0, 32.0, 1.0, 1.0, 30.0, 30.0, 10.0, 4.0],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.actor_encoder = nn.Sequential(
            nn.Linear(BACKGROUND_ACTOR_STATE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.actor_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.actor_norm = nn.LayerNorm(d_model)
        self.scenario_embedding = nn.Embedding(NUM_SCENARIO_CODES, d_model)
        self.formation_embedding = nn.Embedding(2, d_model)
        self.rule_action_embedding = nn.Embedding(3, d_model)
        self.rule_condition_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        ego_state: Tensor,
        relation_state: Tensor,
        relation_valid_mask: Tensor,
        agent_role: Tensor,
        bev_feature: Tensor,
        background_actor_state: Tensor | None = None,
        background_actor_valid_mask: Tensor | None = None,
        scenario_code: Tensor | None = None,
        rule_formation_state: Tensor | None = None,
        rule_action_condition: Tensor | None = None,
    ) -> Tensor:
        batch_size = int(ego_state.shape[0])
        ego_normalized = (ego_state / self.ego_scale).clamp(-5.0, 5.0)
        relation_slot_mask = relation_valid_mask.unsqueeze(-1).expand(
            -1, -1, -1, RELATION_STATE_DIM // RELATION_NEIGHBORS
        )
        relation_masked = relation_state.reshape(
            batch_size,
            NUM_PLATOON_ROLES,
            RELATION_NEIGHBORS,
            RELATION_STATE_DIM // RELATION_NEIGHBORS,
        )
        relation_masked = relation_masked * relation_slot_mask.to(
            dtype=relation_state.dtype
        )
        relation_normalized = (
            relation_masked.reshape(batch_size, NUM_PLATOON_ROLES, RELATION_STATE_DIM)
            / self.relation_scale
        ).clamp(-5.0, 5.0)
        bev_global = F.adaptive_avg_pool2d(
            bev_feature.flatten(0, 1), output_size=1
        ).flatten(1)
        bev_global = bev_global.reshape(batch_size, NUM_PLATOON_ROLES, -1)

        tokens = self.fusion(
            torch.cat(
                (
                    self.ego_encoder(ego_normalized),
                    self.relation_encoder(relation_normalized),
                    self.role_embedding(agent_role),
                    self.bev_encoder(bev_global),
                ),
                dim=-1,
            )
        )
        if any(
            value is None
            for value in (
                background_actor_state,
                background_actor_valid_mask,
                scenario_code,
                rule_formation_state,
                rule_action_condition,
            )
        ):
            raise BEVPlannerError("v2 context inputs are incomplete")
        actor_state = background_actor_state.reshape(
            batch_size * NUM_PLATOON_ROLES,
            MAX_BACKGROUND_ACTORS,
            BACKGROUND_ACTOR_STATE_DIM,
        )
        actor_valid = background_actor_valid_mask.reshape(
            batch_size * NUM_PLATOON_ROLES, MAX_BACKGROUND_ACTORS
        )
        actor_tokens = self.actor_encoder(
            (actor_state / self.actor_scale).clamp(-5.0, 5.0)
        )
        safe_valid = actor_valid.clone()
        empty_rows = ~safe_valid.any(dim=1)
        if bool(empty_rows.any()):
            safe_valid[empty_rows, 0] = True
            actor_tokens = actor_tokens.clone()
            actor_tokens[empty_rows, 0] = 0.0
        actor_update = self.actor_attention(
            tokens.reshape(batch_size * NUM_PLATOON_ROLES, 1, -1),
            actor_tokens,
            actor_tokens,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )[0].reshape(batch_size, NUM_PLATOON_ROLES, -1)
        actor_update = actor_update.masked_fill(
            empty_rows.reshape(batch_size, NUM_PLATOON_ROLES, 1), 0.0
        )
        tokens = self.actor_norm(tokens + self.dropout(actor_update))
        scenario_token = self.scenario_embedding(scenario_code).unsqueeze(1).expand(
            -1, NUM_PLATOON_ROLES, -1
        )
        formation_token = self.formation_embedding(rule_formation_state).unsqueeze(
            1
        ).expand(-1, NUM_PLATOON_ROLES, -1)
        action_token = self.rule_action_embedding(rule_action_condition + 1)
        tokens = tokens + self.rule_condition_fusion(
            torch.cat((scenario_token, formation_token, action_token), dim=-1)
        )
        attended = self.joint_attention(tokens, tokens, tokens, need_weights=False)[0]
        tokens = self.joint_norm1(tokens + self.dropout(attended))
        return self.joint_norm2(tokens + self.dropout(self.joint_ffn(tokens)))


class MetricTrajectoryBEVSampler(nn.Module):
    """Bilinearly sample ego-local trajectory points from a semantic BEV map."""

    def __init__(self) -> None:
        super().__init__()
        bev_config = SemanticBEVConfig()
        self.x_min_m = float(bev_config.x_min_m)
        self.x_max_m = float(bev_config.x_max_m)
        self.y_min_m = float(bev_config.y_min_m)
        self.y_max_m = float(bev_config.y_max_m)

    def metric_to_grid(self, trajectory_xy: Tensor) -> Tensor:
        x = trajectory_xy[..., 0]
        y = trajectory_xy[..., 1]
        # Semantic BEV image coordinates: forward points up and left points left.
        grid_x = 1.0 - 2.0 * (y - self.y_min_m) / (self.y_max_m - self.y_min_m)
        grid_y = 1.0 - 2.0 * (x - self.x_min_m) / (self.x_max_m - self.x_min_m)
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, bev_feature: Tensor, trajectory_xy: Tensor) -> Tensor:
        if trajectory_xy.ndim != 4 or tuple(trajectory_xy.shape[-2:]) != (
            TRAJECTORY_STEPS,
            2,
        ):
            raise BEVPlannerError(
                "trajectory_xy must have shape [N,10,8,2] for BEV sampling"
            )
        grid = self.metric_to_grid(trajectory_xy)
        sampled = F.grid_sample(
            bev_feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.permute(0, 2, 3, 1).contiguous()


class CrossBEVDecoderBlock(nn.Module):
    """One cross-BEV and cross-role decoder block."""

    def __init__(
        self, d_model: int, num_heads: int, ffn_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.bev_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.role_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        sampled_bev: Tensor,
        joint_tokens: Tensor,
    ) -> Tensor:
        bev_update = self.bev_attention(
            query, sampled_bev, sampled_bev, need_weights=False
        )[0]
        query = self.norm1(query + self.dropout(bev_update))
        role_update = self.role_attention(
            query, joint_tokens, joint_tokens, need_weights=False
        )[0]
        query = self.norm2(query + self.dropout(role_update))
        return self.norm3(query + self.dropout(self.ffn(query)))


class PredecessorActionEncoder(nn.Module):
    """Encode one normalized predecessor trajectory into a shared action token."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(TRAJECTORY_STEPS * TRAJECTORY_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, normalized_trajectory: Tensor) -> Tensor:
        if normalized_trajectory.ndim != 3 or tuple(
            normalized_trajectory.shape[1:]
        ) != (TRAJECTORY_STEPS, TRAJECTORY_DIM):
            raise BEVPlannerError("predecessor trajectory must have shape [B,8,3]")
        if not normalized_trajectory.is_floating_point() or not bool(
            torch.isfinite(normalized_trajectory).all()
        ):
            raise BEVPlannerError(
                "predecessor trajectory must contain finite floating-point values"
            )
        return self.network(normalized_trajectory.flatten(1))


class CrossBEVDiffusionDecoder(nn.Module):
    """Predict denoised trajectory candidates from noisy dynamic anchors."""

    def __init__(self, config: BEVOnlyDiffusionPlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.sampler = MetricTrajectoryBEVSampler()
        self.anchor_encoder = nn.Sequential(
            nn.Linear(TRAJECTORY_STEPS * TRAJECTORY_DIM, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.mode_embedding = nn.Embedding(NUM_MODES, config.d_model)
        self.timestep_embedding = SinusoidalTimestepEmbedding(config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList(
            CrossBEVDecoderBlock(
                config.d_model,
                config.num_heads,
                config.ffn_dim,
                config.dropout,
            )
            for _ in range(config.decoder_layers)
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(
                config.d_model,
                TRAJECTORY_STEPS * TRAJECTORY_DIM,
            ),
        )
        nn.init.zeros_(self.trajectory_head[-1].weight)
        nn.init.zeros_(self.trajectory_head[-1].bias)
        if config.predecessor_condition == "predicted_detached":
            # Variant B owns extra randomly initialized parameters.  Preserve the
            # caller RNG state so constructing B with the same seed does not
            # perturb any later shared parameters (notably ``mode_head``).
            with torch.random.fork_rng(devices=[]):
                self.predecessor_action_encoder: PredecessorActionEncoder | None = (
                    PredecessorActionEncoder(config.d_model)
                )
            self.predecessor_residual_gate = nn.Parameter(torch.zeros(()))
        else:
            self.predecessor_action_encoder = None
            self.register_parameter("predecessor_residual_gate", None)

    def _decode_flat(
        self,
        noisy_flat: Tensor,
        coarse_flat: Tensor,
        bev_flat: Tensor,
        role_token_flat: Tensor,
        joint_tokens: Tensor,
        timesteps_flat: Tensor,
        *,
        predecessor_action: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        flat_count = int(noisy_flat.shape[0])
        sampled = self.sampler(bev_flat, noisy_flat)
        sampled = sampled.reshape(
            flat_count * NUM_MODES,
            TRAJECTORY_STEPS,
            self.config.d_model,
        )

        heading_normalized = coarse_flat[..., 2:3] / math.pi
        anchor_input = torch.cat((noisy_flat, heading_normalized), dim=-1)
        query = self.anchor_encoder(anchor_input.flatten(-2))
        mode_ids = torch.arange(
            NUM_MODES, device=query.device, dtype=torch.long
        ).unsqueeze(0)
        query = query + self.mode_embedding(mode_ids)
        query = query + role_token_flat.unsqueeze(1)
        time_embedding = self.timestep_embedding(timesteps_flat)
        query = query + time_embedding.unsqueeze(1)
        if predecessor_action is not None:
            if (
                self.predecessor_action_encoder is None
                or self.predecessor_residual_gate is None
            ):
                raise BEVPlannerError(
                    "predecessor action requires predicted_detached configuration"
                )
            action_token = self.predecessor_action_encoder(predecessor_action)
            query = query + torch.tanh(self.predecessor_residual_gate) * (
                action_token.unsqueeze(1)
            )
        query = self.input_norm(query).reshape(
            flat_count * NUM_MODES, 1, self.config.d_model
        )

        joint_tokens = joint_tokens.repeat_interleave(NUM_MODES, dim=0)
        for block in self.blocks:
            query = block(query, sampled, joint_tokens)
        mode_features = query.squeeze(1).reshape(
            flat_count, NUM_MODES, self.config.d_model
        )

        residual = self.trajectory_head(mode_features).reshape(
            flat_count,
            NUM_MODES,
            TRAJECTORY_STEPS,
            TRAJECTORY_DIM,
        )
        xy = noisy_flat + torch.tanh(residual[..., :2]) * float(
            self.config.max_xy_residual_m
        )
        heading = coarse_flat[..., 2] + math.pi * torch.tanh(residual[..., 2])
        heading = torch.atan2(torch.sin(heading), torch.cos(heading))
        candidates = torch.cat((xy, heading.unsqueeze(-1)), dim=-1)
        return candidates, mode_features

    def forward(
        self,
        noisy_xy_metric: Tensor,
        coarse_trajectories: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
    ) -> tuple[Tensor, Tensor]:
        """Vectorized three-role decoder used by variant A."""

        batch_size = context.batch_size
        flat_count = batch_size * NUM_PLATOON_ROLES
        noisy_flat = noisy_xy_metric.reshape(flat_count, NUM_MODES, TRAJECTORY_STEPS, 2)
        coarse_flat = coarse_trajectories.reshape(
            flat_count, NUM_MODES, TRAJECTORY_STEPS, TRAJECTORY_DIM
        )
        bev_flat = context.bev_feature.reshape(
            flat_count,
            context.bev_feature.shape[-3],
            context.bev_feature.shape[-2],
            context.bev_feature.shape[-1],
        )
        role_token_flat = context.role_tokens.reshape(flat_count, -1)
        joint_tokens = (
            context.role_tokens.unsqueeze(1)
            .expand(-1, NUM_PLATOON_ROLES, -1, -1)
            .reshape(flat_count, NUM_PLATOON_ROLES, self.config.d_model)
        )
        candidates, mode_features = self._decode_flat(
            noisy_flat,
            coarse_flat,
            bev_flat,
            role_token_flat,
            joint_tokens,
            timesteps.reshape(flat_count),
            predecessor_action=None,
        )
        return (
            candidates.reshape(
                batch_size,
                NUM_PLATOON_ROLES,
                NUM_MODES,
                TRAJECTORY_STEPS,
                TRAJECTORY_DIM,
            ),
            mode_features.reshape(
                batch_size,
                NUM_PLATOON_ROLES,
                NUM_MODES,
                self.config.d_model,
            ),
        )

    def forward_role(
        self,
        noisy_xy_metric: Tensor,
        coarse_trajectories: Tensor,
        timesteps: Tensor,
        context: BEVPlannerContext,
        *,
        role_index: int,
        predecessor_action: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Decode one role for the ordered predicted-detached path."""

        if self.config.predecessor_condition != "predicted_detached":
            raise BEVPlannerError(
                "forward_role requires predicted_detached configuration"
            )
        if (
            isinstance(role_index, bool)
            or not isinstance(role_index, int)
            or role_index < 0
            or role_index >= NUM_PLATOON_ROLES
        ):
            raise BEVPlannerError("role_index must be 0, 1, or 2")
        if role_index == 0 and predecessor_action is not None:
            raise BEVPlannerError("leader must not receive predecessor action")
        if role_index > 0 and predecessor_action is None:
            raise BEVPlannerError("middle and rear require a predecessor action")
        batch_size = context.batch_size
        if predecessor_action is not None:
            if (
                predecessor_action.ndim != 3
                or tuple(predecessor_action.shape[1:])
                != (TRAJECTORY_STEPS, TRAJECTORY_DIM)
                or int(predecessor_action.shape[0]) != batch_size
            ):
                raise BEVPlannerError(
                    "predecessor trajectory must have shape [B,8,3]"
                )
            if predecessor_action.device != context.role_tokens.device:
                raise BEVPlannerError(
                    "predecessor trajectory must be on the context device"
                )
        return self._decode_flat(
            noisy_xy_metric[:, role_index],
            coarse_trajectories[:, role_index],
            context.bev_feature[:, role_index],
            context.role_tokens[:, role_index],
            context.role_tokens,
            timesteps[:, role_index],
            predecessor_action=predecessor_action,
        )


class BEVOnlyDiffusionPlanner(nn.Module):
    """Shared three-role BEV-only diffusion planner."""

    input_roles: Final[int] = NUM_PLATOON_ROLES
    num_modes: Final[int] = NUM_MODES
    trajectory_steps: Final[int] = TRAJECTORY_STEPS

    def __init__(self, config: BEVOnlyDiffusionPlannerConfig | None = None) -> None:
        super().__init__()
        self.config = config or BEVOnlyDiffusionPlannerConfig()
        self.backbone = BEVResNet18Backbone(self.config.backbone)
        self.bev_fusion = LightweightBEVFusion(
            self.backbone.feature_channels,
            self.config.d_model,
        )
        self.context_encoder = JointStateRelationEncoder(
            self.config.d_model,
            self.config.num_heads,
            self.config.ffn_dim,
            self.config.dropout,
        )
        self.diffusion_decoder = CrossBEVDiffusionDecoder(self.config)
        self.mode_head = nn.Linear(self.config.d_model, 1)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        bev_config = SemanticBEVConfig()
        self.register_buffer(
            "trajectory_xy_min",
            torch.tensor([bev_config.x_min_m, bev_config.y_min_m], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "trajectory_xy_max",
            torch.tensor([bev_config.x_max_m, bev_config.y_max_m], dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def inference_roll_timesteps(denoise_steps: int = 2) -> tuple[int, ...]:
        if isinstance(denoise_steps, bool) or not isinstance(denoise_steps, int):
            raise BEVPlannerError("denoise_steps must be a positive integer")
        if denoise_steps <= 0:
            raise BEVPlannerError("denoise_steps must be a positive integer")
        step_ratio = 20.0 / float(denoise_steps)
        return tuple(
            int(round(index * step_ratio)) for index in reversed(range(denoise_steps))
        )

    @staticmethod
    def _require_tensor(
        value: Tensor,
        *,
        name: str,
        dtype: torch.dtype,
        shape_tail: tuple[int, ...],
        batch_size: int | None = None,
        finite: bool = False,
    ) -> int:
        if not isinstance(value, Tensor):
            raise BEVPlannerError(f"{name} must be a torch.Tensor")
        if value.dtype is not dtype:
            raise BEVPlannerError(f"{name} must use dtype {dtype}")
        if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail:
            raise BEVPlannerError(
                f"{name} must have shape [B,{','.join(map(str, shape_tail))}]"
            )
        current_batch = int(value.shape[0])
        if current_batch <= 0:
            raise BEVPlannerError(f"{name} batch dimension must be positive")
        if batch_size is not None and current_batch != batch_size:
            raise BEVPlannerError(f"{name} batch dimension does not match bev")
        if finite and not bool(torch.isfinite(value).all()):
            raise BEVPlannerError(f"{name} contains non-finite values")
        return current_batch

    @staticmethod
    def _require_same_device(reference: Tensor, values: dict[str, Tensor]) -> None:
        for name, value in values.items():
            if value.device != reference.device:
                raise BEVPlannerError(f"{name} must be on the same device as bev")

    def _validate_context_inputs(
        self,
        bev: Tensor,
        ego_state: Tensor,
        formation_relation_state: Tensor,
        relation_valid_mask: Tensor,
        agent_role: Tensor,
        background_actor_state: Tensor | None = None,
        background_actor_valid_mask: Tensor | None = None,
        scenario_code: Tensor | None = None,
        rule_formation_state: Tensor | None = None,
        rule_action_condition: Tensor | None = None,
    ) -> int:
        batch_size = self._require_tensor(
            bev,
            name="bev",
            dtype=torch.uint8,
            shape_tail=(NUM_PLATOON_ROLES, 8, 256, 256),
        )
        self._require_tensor(
            ego_state,
            name="ego_state",
            dtype=torch.float32,
            shape_tail=(NUM_PLATOON_ROLES, EGO_STATE_DIM),
            batch_size=batch_size,
            finite=True,
        )
        self._require_tensor(
            formation_relation_state,
            name="formation_relation_state",
            dtype=torch.float32,
            shape_tail=(NUM_PLATOON_ROLES, RELATION_STATE_DIM),
            batch_size=batch_size,
            finite=True,
        )
        self._require_tensor(
            relation_valid_mask,
            name="relation_valid_mask",
            dtype=torch.bool,
            shape_tail=(NUM_PLATOON_ROLES, RELATION_NEIGHBORS),
            batch_size=batch_size,
        )
        self._require_tensor(
            agent_role,
            name="agent_role",
            dtype=torch.int64,
            shape_tail=(NUM_PLATOON_ROLES,),
            batch_size=batch_size,
        )
        expected_roles = torch.arange(
            NUM_PLATOON_ROLES, device=agent_role.device, dtype=torch.int64
        ).expand(batch_size, -1)
        if not torch.equal(agent_role, expected_roles):
            raise BEVPlannerError(
                "agent_role must be [LEADER,MIDDLE,REAR] for every joint sample"
            )
        device_values = {
                "ego_state": ego_state,
                "formation_relation_state": formation_relation_state,
                "relation_valid_mask": relation_valid_mask,
                "agent_role": agent_role,
            }
        if any(
            value is None
            for value in (
                background_actor_state,
                background_actor_valid_mask,
                scenario_code,
                rule_formation_state,
                rule_action_condition,
            )
        ):
            raise BEVPlannerError("v2 planner requires all explicit condition inputs")
        self._require_tensor(
            background_actor_state,
            name="background_actor_state",
            dtype=torch.float32,
            shape_tail=(
                NUM_PLATOON_ROLES,
                MAX_BACKGROUND_ACTORS,
                BACKGROUND_ACTOR_STATE_DIM,
            ),
            batch_size=batch_size,
            finite=True,
        )
        self._require_tensor(
            background_actor_valid_mask,
            name="background_actor_valid_mask",
            dtype=torch.bool,
            shape_tail=(NUM_PLATOON_ROLES, MAX_BACKGROUND_ACTORS),
            batch_size=batch_size,
        )
        self._require_tensor(
            scenario_code,
            name="scenario_code",
            dtype=torch.int64,
            shape_tail=(),
            batch_size=batch_size,
        )
        self._require_tensor(
            rule_formation_state,
            name="rule_formation_state",
            dtype=torch.int64,
            shape_tail=(),
            batch_size=batch_size,
        )
        self._require_tensor(
            rule_action_condition,
            name="rule_action_condition",
            dtype=torch.int64,
            shape_tail=(NUM_PLATOON_ROLES,),
            batch_size=batch_size,
        )
        if bool(((scenario_code < 1) | (scenario_code > 5)).any()):
            raise BEVPlannerError("scenario_code must be in [1,5]")
        if bool(((rule_formation_state < 0) | (rule_formation_state > 1)).any()):
            raise BEVPlannerError("rule_formation_state must be 0 or 1")
        if bool(((rule_action_condition < -1) | (rule_action_condition > 1)).any()):
            raise BEVPlannerError("rule_action_condition must be -1, 0, or 1")
        device_values.update(
            {
                "background_actor_state": background_actor_state,
                "background_actor_valid_mask": background_actor_valid_mask,
                "scenario_code": scenario_code,
                "rule_formation_state": rule_formation_state,
                "rule_action_condition": rule_action_condition,
            }
        )
        self._require_same_device(bev, device_values)
        return batch_size

    def _validate_trajectory_inputs(
        self,
        coarse_trajectories: Tensor,
        mode_valid_mask: Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self._require_tensor(
            coarse_trajectories,
            name="coarse_trajectories",
            dtype=torch.float32,
            shape_tail=(
                NUM_PLATOON_ROLES,
                NUM_MODES,
                TRAJECTORY_STEPS,
                TRAJECTORY_DIM,
            ),
            batch_size=batch_size,
            finite=True,
        )
        self._require_tensor(
            mode_valid_mask,
            name="mode_valid_mask",
            dtype=torch.bool,
            shape_tail=(NUM_PLATOON_ROLES, NUM_MODES),
            batch_size=batch_size,
        )
        if coarse_trajectories.device != device or mode_valid_mask.device != device:
            raise BEVPlannerError(
                "coarse_trajectories and mode_valid_mask must be on the same device as bev"
            )
        if not bool(mode_valid_mask[..., STOP_MODE_INDEX].all()):
            raise BEVPlannerError("STOP must be valid for every role")

    def _normalize_xy(self, trajectory_xy: Tensor) -> Tensor:
        return (
            2.0
            * (trajectory_xy - self.trajectory_xy_min)
            / (self.trajectory_xy_max - self.trajectory_xy_min)
            - 1.0
        )

    def _denormalize_xy(self, trajectory_xy: Tensor) -> Tensor:
        return (trajectory_xy + 1.0) * 0.5 * (
            self.trajectory_xy_max - self.trajectory_xy_min
        ) + self.trajectory_xy_min

    def encode_context(
        self,
        bev: Tensor,
        ego_state: Tensor,
        formation_relation_state: Tensor,
        relation_valid_mask: Tensor,
        agent_role: Tensor,
        *,
        background_actor_state: Tensor | None = None,
        background_actor_valid_mask: Tensor | None = None,
        scenario_code: Tensor | None = None,
        rule_formation_state: Tensor | None = None,
        rule_action_condition: Tensor | None = None,
    ) -> BEVPlannerContext:
        """Encode reusable BEV and joint role context."""

        batch_size = self._validate_context_inputs(
            bev,
            ego_state,
            formation_relation_state,
            relation_valid_mask,
            agent_role,
            background_actor_state,
            background_actor_valid_mask,
            scenario_code,
            rule_formation_state,
            rule_action_condition,
        )
        flattened_bev = bev.reshape(batch_size * NUM_PLATOON_ROLES, 8, 256, 256)
        features = self.backbone(flattened_bev)
        fused = self.bev_fusion(features)
        fused = fused.reshape(
            batch_size,
            NUM_PLATOON_ROLES,
            self.config.d_model,
            fused.shape[-2],
            fused.shape[-1],
        )
        role_tokens = self.context_encoder(
            ego_state,
            formation_relation_state,
            relation_valid_mask,
            agent_role,
            fused,
            background_actor_state,
            background_actor_valid_mask,
            scenario_code,
            rule_formation_state,
            rule_action_condition,
        )
        return BEVPlannerContext(bev_feature=fused, role_tokens=role_tokens)

    def _validate_explicit_noise(
        self,
        noise: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        if noise is None:
            return None
        self._require_tensor(
            noise,
            name="diffusion_noise",
            dtype=torch.float32,
            shape_tail=(
                NUM_PLATOON_ROLES,
                NUM_MODES,
                TRAJECTORY_STEPS,
                2,
            ),
            batch_size=batch_size,
            finite=True,
        )
        if noise.device != device:
            raise BEVPlannerError("diffusion_noise must be on the same device as bev")
        return noise

    def _validate_timesteps(
        self,
        timesteps: Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self._require_tensor(
            timesteps,
            name="diffusion_timesteps",
            dtype=torch.int64,
            shape_tail=(NUM_PLATOON_ROLES,),
            batch_size=batch_size,
        )
        if timesteps.device != device:
            raise BEVPlannerError(
                "diffusion_timesteps must be on the same device as bev"
            )
        if bool(
            ((timesteps < 0) | (timesteps >= self.config.num_train_timesteps)).any()
        ):
            raise BEVPlannerError("diffusion_timesteps contains an invalid timestep")

    def predict_denoised_candidates(
        self,
        noisy_xy_normalized: Tensor,
        diffusion_timesteps: Tensor,
        context: BEVPlannerContext,
        coarse_trajectories: Tensor,
        mode_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run one differentiable denoising prediction.

        Returns physical ``[x,y,heading]`` candidates and unmasked raw mode
        logits.  The public ``forward`` applies the hard mask.
        """

        if not isinstance(context, BEVPlannerContext):
            raise BEVPlannerError("context must be a BEVPlannerContext")
        if context.role_tokens.ndim != 3 or context.bev_feature.ndim != 5:
            raise BEVPlannerError("context does not match the frozen planner rank")
        batch_size = context.batch_size
        self._require_tensor(
            noisy_xy_normalized,
            name="noisy_xy_normalized",
            dtype=torch.float32,
            shape_tail=(
                NUM_PLATOON_ROLES,
                NUM_MODES,
                TRAJECTORY_STEPS,
                2,
            ),
            batch_size=batch_size,
            finite=True,
        )
        self._require_tensor(
            coarse_trajectories,
            name="coarse_trajectories",
            dtype=torch.float32,
            shape_tail=(
                NUM_PLATOON_ROLES,
                NUM_MODES,
                TRAJECTORY_STEPS,
                TRAJECTORY_DIM,
            ),
            batch_size=batch_size,
            finite=True,
        )
        self._require_tensor(
            mode_valid_mask,
            name="mode_valid_mask",
            dtype=torch.bool,
            shape_tail=(NUM_PLATOON_ROLES, NUM_MODES),
            batch_size=batch_size,
        )
        if not bool(mode_valid_mask[..., STOP_MODE_INDEX].all()):
            raise BEVPlannerError("STOP must be valid for every role")
        self._validate_timesteps(
            diffusion_timesteps,
            batch_size=batch_size,
            device=context.role_tokens.device,
        )
        expected_context_shapes = (
            (
                batch_size,
                NUM_PLATOON_ROLES,
                self.config.d_model,
                64,
                64,
            ),
            (batch_size, NUM_PLATOON_ROLES, self.config.d_model),
        )
        if (
            tuple(context.bev_feature.shape) != expected_context_shapes[0]
            or tuple(context.role_tokens.shape) != expected_context_shapes[1]
        ):
            raise BEVPlannerError("context does not match the frozen planner shape")
        if (
            not context.bev_feature.is_floating_point()
            or not context.role_tokens.is_floating_point()
            or not bool(torch.isfinite(context.bev_feature).all())
            or not bool(torch.isfinite(context.role_tokens).all())
        ):
            raise BEVPlannerError("context must contain finite floating-point tensors")
        device = context.role_tokens.device
        if (
            noisy_xy_normalized.device != device
            or coarse_trajectories.device != device
            or mode_valid_mask.device != device
            or context.bev_feature.device != device
        ):
            raise BEVPlannerError(
                "denoising inputs and context must be on the same device"
            )
        return self._predict_denoised_candidates_impl(
            noisy_xy_normalized,
            diffusion_timesteps,
            context,
            coarse_trajectories,
            mode_valid_mask,
        )

    def _predict_denoised_candidates_impl(
        self,
        noisy_xy_normalized: Tensor,
        diffusion_timesteps: Tensor,
        context: BEVPlannerContext,
        coarse_trajectories: Tensor,
        mode_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Denoise inputs already validated by a public entry point."""

        noisy_xy_metric = self._denormalize_xy(noisy_xy_normalized.clamp(-1.0, 1.0))
        if self.config.predecessor_condition == "none":
            candidates, mode_features = self.diffusion_decoder(
                noisy_xy_metric,
                coarse_trajectories,
                diffusion_timesteps,
                context,
            )
            raw_logits = self.mode_head(mode_features).squeeze(-1)
            return candidates, raw_logits
        return self._predict_detached_candidates(
            noisy_xy_metric,
            diffusion_timesteps,
            context,
            coarse_trajectories,
            mode_valid_mask,
        )

    def _normalize_predecessor_action(self, trajectory: Tensor) -> Tensor:
        normalized_xy = self._normalize_xy(trajectory[..., :2]).clamp(-1.0, 1.0)
        wrapped_heading = torch.atan2(
            torch.sin(trajectory[..., 2]),
            torch.cos(trajectory[..., 2]),
        )
        normalized_heading = (wrapped_heading / math.pi).unsqueeze(-1)
        return torch.cat((normalized_xy, normalized_heading), dim=-1).detach()

    @staticmethod
    def _select_predecessor_trajectory(
        candidates: Tensor,
        raw_logits: Tensor,
        mode_valid_mask: Tensor,
    ) -> Tensor:
        masked_logits = raw_logits.masked_fill(~mode_valid_mask, float("-inf"))
        selected_mode = masked_logits.argmax(dim=-1)
        gather_index = selected_mode[:, None, None, None].expand(
            -1, 1, TRAJECTORY_STEPS, TRAJECTORY_DIM
        )
        return candidates.gather(1, gather_index).squeeze(1)

    def _predict_detached_candidates(
        self,
        noisy_xy_metric: Tensor,
        diffusion_timesteps: Tensor,
        context: BEVPlannerContext,
        coarse_trajectories: Tensor,
        mode_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        role_candidates = []
        role_logits = []
        predecessor_action: Tensor | None = None
        for role_index in range(NUM_PLATOON_ROLES):
            candidates, mode_features = self.diffusion_decoder.forward_role(
                noisy_xy_metric,
                coarse_trajectories,
                diffusion_timesteps,
                context,
                role_index=role_index,
                predecessor_action=predecessor_action,
            )
            raw_logits = self.mode_head(mode_features).squeeze(-1)
            role_candidates.append(candidates)
            role_logits.append(raw_logits)
            if role_index + 1 < NUM_PLATOON_ROLES:
                selected = self._select_predecessor_trajectory(
                    candidates,
                    raw_logits,
                    mode_valid_mask[:, role_index],
                )
                predecessor_action = self._normalize_predecessor_action(selected)
        return torch.stack(role_candidates, dim=1), torch.stack(role_logits, dim=1)

    def _training_forward(
        self,
        context: BEVPlannerContext,
        coarse_trajectories: Tensor,
        *,
        diffusion_noise: Tensor | None,
        diffusion_timesteps: Tensor | None,
        mode_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = context.batch_size
        device = coarse_trajectories.device
        if diffusion_timesteps is None:
            timesteps = torch.randint(
                0,
                self.config.train_timestep_upper,
                (batch_size, NUM_PLATOON_ROLES),
                device=device,
                dtype=torch.int64,
            )
        else:
            timesteps = diffusion_timesteps
            self._validate_timesteps(timesteps, batch_size=batch_size, device=device)
            if bool((timesteps >= self.config.train_timestep_upper).any()):
                raise BEVPlannerError("training diffusion_timesteps must be in [0,50)")
        anchor_normalized = self._normalize_xy(coarse_trajectories[..., :2])
        noise = (
            torch.randn_like(anchor_normalized)
            if diffusion_noise is None
            else diffusion_noise
        )
        flat_shape = (
            batch_size * NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        noisy = self.diffusion_scheduler.add_noise(
            anchor_normalized.reshape(flat_shape),
            noise.reshape(flat_shape),
            timesteps.reshape(-1),
        ).reshape_as(anchor_normalized)
        return self._predict_denoised_candidates_impl(
            noisy.clamp(-1.0, 1.0),
            timesteps,
            context,
            coarse_trajectories,
            mode_valid_mask,
        )

    def _inference_noise(
        self,
        shape: torch.Size,
        *,
        device: torch.device,
    ) -> Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self.config.inference_seed))
        return torch.randn(
            shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

    def _inference_forward(
        self,
        context: BEVPlannerContext,
        coarse_trajectories: Tensor,
        *,
        diffusion_noise: Tensor | None,
        mode_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = context.batch_size
        device = coarse_trajectories.device
        anchor_normalized = self._normalize_xy(coarse_trajectories[..., :2])
        noise = (
            self._inference_noise(anchor_normalized.shape, device=device)
            if diffusion_noise is None
            else diffusion_noise
        )
        noise_timesteps = torch.full(
            (batch_size * NUM_PLATOON_ROLES,),
            self.config.inference_noise_timestep,
            device=device,
            dtype=torch.int64,
        )
        flat_shape = (
            batch_size * NUM_PLATOON_ROLES,
            NUM_MODES,
            TRAJECTORY_STEPS,
            2,
        )
        sample = self.diffusion_scheduler.add_noise(
            anchor_normalized.reshape(flat_shape),
            noise.reshape(flat_shape),
            noise_timesteps,
        ).reshape_as(anchor_normalized)
        self.diffusion_scheduler.set_timesteps(
            self.config.num_train_timesteps, device=device
        )

        candidates: Tensor | None = None
        raw_logits: Tensor | None = None
        for timestep in self.inference_roll_timesteps(
            self.config.inference_denoise_steps
        ):
            batch_timesteps = torch.full(
                (batch_size, NUM_PLATOON_ROLES),
                timestep,
                device=device,
                dtype=torch.int64,
            )
            candidates, raw_logits = self._predict_denoised_candidates_impl(
                sample.clamp(-1.0, 1.0),
                batch_timesteps,
                context,
                coarse_trajectories,
                mode_valid_mask,
            )
            predicted_normalized = self._normalize_xy(candidates[..., :2])
            sample = self.diffusion_scheduler.step(
                model_output=predicted_normalized.reshape(flat_shape),
                timestep=timestep,
                sample=sample.reshape(flat_shape),
            ).prev_sample.reshape_as(sample)
        if candidates is None or raw_logits is None:
            raise BEVPlannerError("inference produced no denoising steps")
        return candidates, raw_logits

    @staticmethod
    def _select(
        candidates: Tensor,
        raw_logits: Tensor,
        mode_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        mode_logits = raw_logits.masked_fill(~mode_valid_mask, float("-inf"))
        selected_mode = mode_logits.argmax(dim=-1)
        gather_index = selected_mode[..., None, None, None].expand(
            -1, -1, 1, TRAJECTORY_STEPS, TRAJECTORY_DIM
        )
        selected_trajectory = candidates.gather(2, gather_index).squeeze(2)
        return {
            "trajectory_candidates": candidates,
            "mode_logits": mode_logits,
            "selected_mode": selected_mode,
            "selected_trajectory": selected_trajectory,
        }

    def forward(
        self,
        bev: Tensor,
        ego_state: Tensor,
        formation_relation_state: Tensor,
        relation_valid_mask: Tensor,
        agent_role: Tensor,
        coarse_trajectories: Tensor,
        mode_valid_mask: Tensor,
        *,
        background_actor_state: Tensor | None = None,
        background_actor_valid_mask: Tensor | None = None,
        scenario_code: Tensor | None = None,
        rule_formation_state: Tensor | None = None,
        rule_action_condition: Tensor | None = None,
        diffusion_noise: Tensor | None = None,
        diffusion_timesteps: Tensor | None = None,
    ) -> dict[str, Tensor]:
        context = self.encode_context(
            bev,
            ego_state,
            formation_relation_state,
            relation_valid_mask,
            agent_role,
            background_actor_state=background_actor_state,
            background_actor_valid_mask=background_actor_valid_mask,
            scenario_code=scenario_code,
            rule_formation_state=rule_formation_state,
            rule_action_condition=rule_action_condition,
        )
        batch_size = context.batch_size
        self._validate_trajectory_inputs(
            coarse_trajectories,
            mode_valid_mask,
            batch_size=batch_size,
            device=bev.device,
        )
        noise = self._validate_explicit_noise(
            diffusion_noise,
            batch_size=batch_size,
            device=bev.device,
        )
        if self.training:
            candidates, raw_logits = self._training_forward(
                context,
                coarse_trajectories,
                diffusion_noise=noise,
                diffusion_timesteps=diffusion_timesteps,
                mode_valid_mask=mode_valid_mask,
            )
        else:
            if diffusion_timesteps is not None:
                raise BEVPlannerError(
                    "diffusion_timesteps override is only valid during training"
                )
            candidates, raw_logits = self._inference_forward(
                context,
                coarse_trajectories,
                diffusion_noise=noise,
                mode_valid_mask=mode_valid_mask,
            )
        return self._select(candidates, raw_logits, mode_valid_mask)


__all__ = [
    "BEVOnlyDiffusionPlanner",
    "BEVOnlyDiffusionPlannerConfig",
    "BEVPlannerContext",
    "BEVPlannerError",
    "CrossBEVDiffusionDecoder",
    "LightweightBEVFusion",
    "MetricTrajectoryBEVSampler",
    "PredecessorActionEncoder",
]
