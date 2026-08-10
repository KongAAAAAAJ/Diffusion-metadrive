"""CF-4 probabilistic temporal chassis execution surrogate."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROLLER_CONTEXT_FIELDS,
    CONTROL_FIELDS,
    ENSEMBLE_SIZE,
    EXECUTION_STEPS,
    INITIAL_STATE_FIELDS,
    NUM_GROUPS,
    NUM_ROLES,
    TRAJECTORY_DIM,
    VEHICLE_CONDITION_FIELDS,
    ChassisExecutionCommand,
    ChassisExecutionMemberPrediction,
    ChassisExecutionPrediction,
    aggregate_ensemble_predictions,
)


class ChassisSurrogateError(ValueError):
    """Raised when a CF-4 model or normalization contract is violated."""


@dataclass(frozen=True)
class ChassisSurrogateConfig:
    hidden_dim: int = 128
    static_dim: int = 96
    role_embedding_dim: int = 8
    mode_embedding_dim: int = 8
    num_gru_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.05
    minimum_std: float = 1e-4
    minimum_log_variance: float = -8.0
    maximum_log_variance: float = 4.0

    def __post_init__(self) -> None:
        if min(
            self.hidden_dim,
            self.static_dim,
            self.role_embedding_dim,
            self.mode_embedding_dim,
            self.num_gru_layers,
            self.transformer_heads,
        ) <= 0:
            raise ChassisSurrogateError("surrogate dimensions and layer counts must be positive")
        if self.static_dim % self.transformer_heads != 0:
            raise ChassisSurrogateError("static_dim must be divisible by transformer_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ChassisSurrogateError("dropout must be in [0,1)")
        if self.minimum_std <= 0.0:
            raise ChassisSurrogateError("minimum_std must be positive")
        if self.minimum_log_variance >= self.maximum_log_variance:
            raise ChassisSurrogateError("log-variance bounds are invalid")


def command_to_dense(tau_cmd: Tensor) -> Tensor:
    """Interpolate [N,3,8,3] commands onto the frozen 0.1-second axis."""

    if tau_cmd.dtype != torch.float32 or tau_cmd.ndim != 4 or tuple(tau_cmd.shape[1:]) != (
        NUM_ROLES,
        8,
        TRAJECTORY_DIM,
    ):
        raise ChassisSurrogateError("tau_cmd must be float32 [N,3,8,3]")
    origin = torch.zeros(
        (tau_cmd.shape[0], NUM_ROLES, 1, TRAJECTORY_DIM),
        dtype=tau_cmd.dtype,
        device=tau_cmd.device,
    )
    values = torch.cat((origin, tau_cmd), dim=2)
    wrapped_delta = torch.atan2(
        torch.sin(torch.diff(values[..., 2], dim=2)),
        torch.cos(torch.diff(values[..., 2], dim=2)),
    )
    heading = torch.cat(
        (
            values[..., :1, 2],
            values[..., :1, 2] + torch.cumsum(wrapped_delta, dim=2),
        ),
        dim=2,
    )
    values = torch.cat((values[..., :2], heading.unsqueeze(-1)), dim=-1)
    flat = values.permute(0, 1, 3, 2).reshape(-1, TRAJECTORY_DIM, 9)
    dense = F.interpolate(flat, size=41, mode="linear", align_corners=True)[..., 1:]
    dense = dense.reshape(tau_cmd.shape[0], NUM_ROLES, TRAJECTORY_DIM, EXECUTION_STEPS)
    dense = dense.permute(0, 1, 3, 2).contiguous()
    dense[..., 2] = torch.atan2(torch.sin(dense[..., 2]), torch.cos(dense[..., 2]))
    return dense


class ChassisFeatureNormalizer(nn.Module):
    """Frozen feature statistics fitted only on the training split."""

    FEATURE_DIMS = {
        "command_dense": TRAJECTORY_DIM,
        "initial_state": len(INITIAL_STATE_FIELDS),
        "vehicle_condition": len(VEHICLE_CONDITION_FIELDS),
        "controller_context": len(CONTROLLER_CONTEXT_FIELDS),
        "executed_trajectory": TRAJECTORY_DIM,
        "chassis_state": len(CHASSIS_STATE_FIELDS),
        "applied_control": len(CONTROL_FIELDS),
    }

    def __init__(
        self,
        statistics: Mapping[str, tuple[Tensor, Tensor]],
        *,
        minimum_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if set(statistics) != set(self.FEATURE_DIMS):
            raise ChassisSurrogateError("normalizer statistics field set mismatch")
        self.minimum_std = float(minimum_std)
        if self.minimum_std <= 0.0:
            raise ChassisSurrogateError("normalizer minimum_std must be positive")
        for name, dimension in self.FEATURE_DIMS.items():
            mean, std = statistics[name]
            if mean.dtype != torch.float32 or std.dtype != torch.float32:
                raise ChassisSurrogateError(f"{name} statistics must be float32")
            if mean.shape != (dimension,) or std.shape != (dimension,):
                raise ChassisSurrogateError(f"{name} statistics shape mismatch")
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
                raise ChassisSurrogateError(f"{name} statistics must be finite")
            if not bool((std >= self.minimum_std).all()):
                raise ChassisSurrogateError(f"{name} std is below minimum_std")
            self.register_buffer(f"{name}_mean", mean.clone())
            self.register_buffer(f"{name}_std", std.clone())

    @classmethod
    def fit(cls, arrays: Mapping[str, object], *, minimum_std: float = 1e-4) -> "ChassisFeatureNormalizer":
        required = {
            "tau_cmd",
            "initial_state",
            "vehicle_condition",
            "controller_context",
            "executed_trajectory",
            "chassis_state",
            "applied_control",
        }
        if not required.issubset(arrays):
            raise ChassisSurrogateError("training arrays are incomplete for normalizer fitting")
        source: dict[str, Tensor] = {}
        for name in required:
            value = torch.tensor(arrays[name], dtype=torch.float32)
            if not bool(torch.isfinite(value).all()):
                raise ChassisSurrogateError(f"normalizer source {name} is non-finite")
            source[name] = value
        source["command_dense"] = command_to_dense(source.pop("tau_cmd"))
        statistics: dict[str, tuple[Tensor, Tensor]] = {}
        for name in cls.FEATURE_DIMS:
            value = source[name].reshape(-1, cls.FEATURE_DIMS[name])
            mean = value.mean(dim=0)
            std = value.std(dim=0, unbiased=False).clamp_min(minimum_std)
            statistics[name] = (mean.to(torch.float32), std.to(torch.float32))
        return cls(statistics, minimum_std=minimum_std)

    def normalize(self, name: str, value: Tensor) -> Tensor:
        return (value - getattr(self, f"{name}_mean")) / getattr(self, f"{name}_std")

    def denormalize(self, name: str, value: Tensor) -> Tensor:
        return value * getattr(self, f"{name}_std") + getattr(self, f"{name}_mean")

    def physical_log_variance(self, name: str, normalized_log_variance: Tensor) -> Tensor:
        std = getattr(self, f"{name}_std")
        return normalized_log_variance + 2.0 * torch.log(std)


@dataclass(frozen=True)
class NormalizedMemberOutput:
    trajectory_mean: Tensor
    trajectory_log_variance: Tensor
    chassis_mean: Tensor
    chassis_log_variance: Tensor
    control_mean: Tensor
    control_log_variance: Tensor


class ChassisExecutionMember(nn.Module):
    """One independently initialized temporal probabilistic surrogate."""

    def __init__(self, config: ChassisSurrogateConfig) -> None:
        super().__init__()
        self.config = config
        self.role_embedding = nn.Embedding(NUM_ROLES, config.role_embedding_dim)
        self.mode_embedding = nn.Embedding(2, config.mode_embedding_dim)
        static_input = (
            len(INITIAL_STATE_FIELDS)
            + len(VEHICLE_CONDITION_FIELDS)
            + len(CONTROLLER_CONTEXT_FIELDS)
            + config.role_embedding_dim
            + config.mode_embedding_dim
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input, config.static_dim),
            nn.LayerNorm(config.static_dim),
            nn.SiLU(),
            nn.Linear(config.static_dim, config.static_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.static_dim,
            nhead=config.transformer_heads,
            dim_feedforward=2 * config.static_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.role_interaction = nn.TransformerEncoder(layer, num_layers=1)
        self.command_encoder = nn.Sequential(
            nn.Linear(TRAJECTORY_DIM + 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.static_to_hidden = nn.Linear(config.static_dim, config.hidden_dim)
        self.temporal = nn.GRU(
            input_size=2 * config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_gru_layers,
            dropout=config.dropout if config.num_gru_layers > 1 else 0.0,
            batch_first=True,
        )
        output_dim = 2 * (
            TRAJECTORY_DIM + len(CHASSIS_STATE_FIELDS) + len(CONTROL_FIELDS)
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, output_dim),
        )

    def forward(
        self,
        *,
        command_dense: Tensor,
        initial_state: Tensor,
        vehicle_condition: Tensor,
        controller_context: Tensor,
        controller_mode: Tensor,
        agent_role: Tensor,
    ) -> NormalizedMemberOutput:
        count = int(command_dense.shape[0])
        expected = (count, NUM_ROLES)
        if command_dense.shape != expected + (EXECUTION_STEPS, TRAJECTORY_DIM):
            raise ChassisSurrogateError("normalized dense command shape mismatch")
        if controller_mode.shape != expected or agent_role.shape != expected:
            raise ChassisSurrogateError("mode/role shape mismatch")
        static = torch.cat(
            (
                initial_state,
                vehicle_condition,
                controller_context,
                self.role_embedding(agent_role),
                self.mode_embedding(controller_mode),
            ),
            dim=-1,
        )
        static = self.role_interaction(self.static_encoder(static))
        times = torch.arange(
            1, EXECUTION_STEPS + 1, dtype=command_dense.dtype, device=command_dense.device
        ) / EXECUTION_STEPS
        time_features = torch.stack(
            (times, torch.sin(math.pi * times), torch.cos(math.pi * times)), dim=-1
        )
        time_features = time_features.reshape(1, 1, EXECUTION_STEPS, 3).expand(
            count, NUM_ROLES, -1, -1
        )
        command_features = self.command_encoder(torch.cat((command_dense, time_features), dim=-1))
        static_features = self.static_to_hidden(static).unsqueeze(2).expand(-1, -1, EXECUTION_STEPS, -1)
        sequence = torch.cat((command_features, static_features), dim=-1)
        sequence = sequence.reshape(count * NUM_ROLES, EXECUTION_STEPS, -1)
        hidden, _ = self.temporal(sequence)
        output = self.output_head(hidden).reshape(count, NUM_ROLES, EXECUTION_STEPS, -1)
        widths = (TRAJECTORY_DIM, len(CHASSIS_STATE_FIELDS), len(CONTROL_FIELDS))
        trajectory_mean, chassis_mean, control_mean, trajectory_logvar, chassis_logvar, control_logvar = torch.split(
            output, widths + widths, dim=-1
        )
        clamp = lambda value: value.clamp(
            self.config.minimum_log_variance, self.config.maximum_log_variance
        )
        return NormalizedMemberOutput(
            trajectory_mean=trajectory_mean,
            trajectory_log_variance=clamp(trajectory_logvar),
            chassis_mean=chassis_mean,
            chassis_log_variance=clamp(chassis_logvar),
            control_mean=control_mean,
            control_log_variance=clamp(control_logvar),
        )


class ChassisExecutionEnsemble(nn.Module):
    """Exactly three independently trained members implementing the frozen Protocol."""

    def __init__(
        self,
        config: ChassisSurrogateConfig,
        normalizer: ChassisFeatureNormalizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.members = nn.ModuleList(
            ChassisExecutionMember(config) for _ in range(ENSEMBLE_SIZE)
        )

    def _normalized_inputs(
        self,
        tau_cmd: Tensor,
        initial_state: Tensor,
        vehicle_condition: Tensor,
        controller_context: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dense = command_to_dense(tau_cmd)
        return (
            self.normalizer.normalize("command_dense", dense),
            self.normalizer.normalize("initial_state", initial_state),
            self.normalizer.normalize("vehicle_condition", vehicle_condition),
            self.normalizer.normalize("controller_context", controller_context),
        )

    def forward_member(
        self,
        member_index: int,
        batch: Mapping[str, Tensor],
    ) -> NormalizedMemberOutput:
        if not 0 <= member_index < ENSEMBLE_SIZE:
            raise ChassisSurrogateError("member_index is out of range")
        inputs = self._normalized_inputs(
            batch["tau_cmd"],
            batch["initial_state"],
            batch["vehicle_condition"],
            batch["controller_context"],
        )
        return self.members[member_index](
            command_dense=inputs[0],
            initial_state=inputs[1],
            vehicle_condition=inputs[2],
            controller_context=inputs[3],
            controller_mode=batch["controller_mode"],
            agent_role=batch["agent_role"],
        )

    def _physical_output(self, output: NormalizedMemberOutput) -> NormalizedMemberOutput:
        return NormalizedMemberOutput(
            trajectory_mean=self.normalizer.denormalize(
                "executed_trajectory", output.trajectory_mean
            ),
            trajectory_log_variance=self.normalizer.physical_log_variance(
                "executed_trajectory", output.trajectory_log_variance
            ),
            chassis_mean=self.normalizer.denormalize("chassis_state", output.chassis_mean),
            chassis_log_variance=self.normalizer.physical_log_variance(
                "chassis_state", output.chassis_log_variance
            ),
            control_mean=self.normalizer.denormalize("applied_control", output.control_mean),
            control_log_variance=self.normalizer.physical_log_variance(
                "applied_control", output.control_log_variance
            ),
        )

    def predict(self, command: ChassisExecutionCommand) -> ChassisExecutionPrediction:
        batch_size = int(command.tau_cmd.shape[0])
        flattened = command.tau_cmd.reshape(-1, NUM_ROLES, 8, TRAJECTORY_DIM)
        repeat = lambda value: value.unsqueeze(1).expand(-1, NUM_GROUPS, *value.shape[1:]).reshape(
            batch_size * NUM_GROUPS, *value.shape[1:]
        )
        batch = {
            "tau_cmd": flattened,
            "initial_state": repeat(command.initial_state),
            "vehicle_condition": repeat(command.vehicle_condition),
            "controller_context": repeat(command.controller_context),
            "controller_mode": repeat(command.controller_mode),
            "agent_role": repeat(command.agent_role),
        }
        stacks: dict[str, list[Tensor]] = {
            field: [] for field in NormalizedMemberOutput.__dataclass_fields__
        }
        for member_index in range(ENSEMBLE_SIZE):
            physical = self._physical_output(self.forward_member(member_index, batch))
            for field in stacks:
                value = getattr(physical, field).reshape(
                    batch_size, NUM_GROUPS, NUM_ROLES, EXECUTION_STEPS, -1
                )
                stacks[field].append(value)
        members = ChassisExecutionMemberPrediction(
            **{field: torch.stack(values, dim=0) for field, values in stacks.items()}
        )
        return aggregate_ensemble_predictions(members)

    def frozen_copy(self) -> "ChassisExecutionEnsemble":
        result = copy.deepcopy(self).eval()
        for parameter in result.parameters():
            parameter.requires_grad_(False)
        return result


def surrogate_config_payload(config: ChassisSurrogateConfig) -> dict[str, object]:
    return asdict(config)
