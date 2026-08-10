"""CF-5 execution-aware reward over frozen chassis-surrogate predictions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from models.bev_planner.joint_reward import (
    JointRewardConfig,
    JointRewardResult,
    JointTrajectoryProxyReward,
)

from .contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROL_FIELDS,
    NUM_GROUPS,
    ChassisExecutionCommand,
    ChassisExecutionPrediction,
    ChassisExecutionSurrogate,
)
from .training import load_surrogate_checkpoint


class ChassisExecutionRewardError(RuntimeError):
    """Raised when execution-aware reward evaluation violates CF-5."""


@dataclass(frozen=True)
class ChassisExecutionRewardConfig:
    lateral_acceleration_limit_mps2: float = 6.0
    yaw_rate_limit_rad_s: float = 1.0
    roll_limit_rad: float = 0.25
    rollover_index_limit: float = 0.80
    minimum_speed_mps: float = -0.05
    control_lower_tolerance: float = -0.05
    control_upper_tolerance: float = 1.05
    uncertainty_soft_scale_m: float = 0.50
    uncertainty_hard_std_m: float = 2.0
    chassis_comfort_weight: float = 0.35
    stability_weight: float = 0.50
    control_effort_weight: float = 0.10
    uncertainty_weight: float = 0.25
    unsafe_base_reward: float = -20.0

    def __post_init__(self) -> None:
        positive = (
            "lateral_acceleration_limit_mps2",
            "yaw_rate_limit_rad_s",
            "roll_limit_rad",
            "rollover_index_limit",
            "uncertainty_soft_scale_m",
            "uncertainty_hard_std_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ChassisExecutionRewardError(f"{name} must be positive and finite")
        if self.rollover_index_limit > 1.0:
            raise ChassisExecutionRewardError("rollover_index_limit cannot exceed one")
        if not math.isfinite(self.minimum_speed_mps) or self.minimum_speed_mps > 0.0:
            raise ChassisExecutionRewardError("minimum_speed_mps must be finite and non-positive")
        if (
            not math.isfinite(self.control_lower_tolerance)
            or not math.isfinite(self.control_upper_tolerance)
            or self.control_lower_tolerance > 0.0
            or self.control_upper_tolerance < 1.0
        ):
            raise ChassisExecutionRewardError("control tolerance must contain [0,1]")
        for name in (
            "chassis_comfort_weight",
            "stability_weight",
            "control_effort_weight",
            "uncertainty_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ChassisExecutionRewardError(f"{name} must be non-negative and finite")
        if not math.isfinite(self.unsafe_base_reward) or self.unsafe_base_reward >= -5.0:
            raise ChassisExecutionRewardError("unsafe_base_reward must be below -5")

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChassisExecutionRewardResult:
    rewards: Tensor
    unsafe: Tensor
    geometric_unsafe: Tensor
    chassis_unsafe: Tensor
    uncertainty_unsafe: Tensor
    prediction: ChassisExecutionPrediction
    components: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if self.rewards.dtype != torch.float32 or self.rewards.shape != (1, NUM_GROUPS):
            raise ChassisExecutionRewardError("rewards must be float32 [1,4]")
        if self.rewards.requires_grad or not bool(torch.isfinite(self.rewards).all()):
            raise ChassisExecutionRewardError("rewards must be finite and gradient-free")
        for name in (
            "unsafe",
            "geometric_unsafe",
            "chassis_unsafe",
            "uncertainty_unsafe",
        ):
            value = getattr(self, name)
            if value.dtype != torch.bool or value.shape != (1, NUM_GROUPS):
                raise ChassisExecutionRewardError(f"{name} must be bool [1,4]")
            if value.device != self.rewards.device:
                raise ChassisExecutionRewardError(f"{name} device must match rewards")
        for name, value in self.components.items():
            if value.dtype != torch.float32 or value.shape != (1, NUM_GROUPS):
                raise ChassisExecutionRewardError(
                    f"component {name} must be float32 [1,4]"
                )
            if value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ChassisExecutionRewardError(
                    f"component {name} must be finite and gradient-free"
                )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_reduce_max(value: Tensor) -> Tensor:
    return value.abs().amax(dim=(2, 3))


def _group_reduce_mean(value: Tensor) -> Tensor:
    return value.abs().mean(dim=(2, 3))


class ChassisExecutionRewardEvaluator:
    """Evaluate G=4 optimized commands without candidate MetaDrive branches."""

    def __init__(
        self,
        surrogate: ChassisExecutionSurrogate,
        *,
        config: ChassisExecutionRewardConfig | None = None,
        geometric_config: JointRewardConfig | None = None,
        checkpoint_metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(surrogate, nn.Module):
            raise ChassisExecutionRewardError("surrogate must be a frozen torch module")
        if any(parameter.requires_grad for parameter in surrogate.parameters()):
            raise ChassisExecutionRewardError("surrogate parameters must be frozen")
        self.surrogate = surrogate.eval()
        self.config = config or ChassisExecutionRewardConfig()
        self.geometric = JointTrajectoryProxyReward(geometric_config or JointRewardConfig())
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.evaluation_count = 0
        self.metadrive_candidate_branch_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        expected_dataset_fingerprint: str,
        allow_diagnostic: bool,
        device: str | torch.device,
        config: ChassisExecutionRewardConfig | None = None,
        geometric_config: JointRewardConfig | None = None,
    ) -> "ChassisExecutionRewardEvaluator":
        path = Path(checkpoint).expanduser().resolve()
        surrogate, payload = load_surrogate_checkpoint(
            path,
            expected_dataset_fingerprint=expected_dataset_fingerprint,
            allow_diagnostic=allow_diagnostic,
            map_location=device,
        )
        frozen = surrogate.frozen_copy().to(device)
        metadata = {
            "checkpoint": str(path),
            "checkpoint_sha256": _file_sha256(path),
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "data_origin": payload["data_origin"],
            "diagnostic_only": payload["diagnostic_only"],
            "eligible_for_formal_training": payload["eligible_for_formal_training"],
        }
        return cls(
            frozen,
            config=config,
            geometric_config=geometric_config,
            checkpoint_metadata=metadata,
        )

    def _ensure_device(self, command: ChassisExecutionCommand) -> None:
        parameter = next(self.surrogate.parameters(), None)
        if parameter is not None and parameter.device != command.tau_cmd.device:
            raise ChassisExecutionRewardError(
                "surrogate and ChassisExecutionCommand must use the same device"
            )

    def score(
        self,
        env: object,
        model_inputs: object,
        command: ChassisExecutionCommand,
    ) -> ChassisExecutionRewardResult:
        if command.tau_cmd.shape[0] != 1:
            raise ChassisExecutionRewardError(
                "online execution-aware reward requires exactly one physical state"
            )
        self._ensure_device(command)
        with torch.inference_mode():
            prediction = self.surrogate.predict(command)
        for field in prediction.__dataclass_fields__:
            value = getattr(prediction, field)
            if value.device != command.tau_cmd.device or value.requires_grad:
                raise ChassisExecutionRewardError(
                    f"surrogate prediction {field} must be gradient-free on the command device"
                )
        self.evaluation_count += 1
        # The geometric backend consumes the same 0.5-second semantic points,
        # but they are sampled from predicted execution tau_a, never tau_cmd.
        executed_half_second = prediction.executed_trajectory_mean[0, :, :, 4::5]
        if executed_half_second.shape != (NUM_GROUPS, 3, 8, 3):
            raise ChassisExecutionRewardError("surrogate execution time axis is invalid")
        geometric: JointRewardResult = self.geometric.score(
            env,
            model_inputs,
            executed_half_second.detach().cpu().numpy().astype(np.float32, copy=False),
        )
        chassis = prediction.chassis_state_mean
        control = prediction.control_mean
        lateral_acceleration = chassis[..., CHASSIS_STATE_FIELDS.index("lateral_acceleration_mps2")]
        yaw_rate = chassis[..., CHASSIS_STATE_FIELDS.index("yaw_rate_rad_s")]
        roll = chassis[..., CHASSIS_STATE_FIELDS.index("roll_rad")]
        rollover = chassis[..., CHASSIS_STATE_FIELDS.index("rollover_index")]
        speed = chassis[..., CHASSIS_STATE_FIELDS.index("longitudinal_speed_mps")]
        throttle = control[..., CONTROL_FIELDS.index("throttle_normalized")]
        brake = control[..., CONTROL_FIELDS.index("brake_normalized")]
        maximum_lateral_acceleration = _group_reduce_max(lateral_acceleration)
        maximum_yaw_rate = _group_reduce_max(yaw_rate)
        maximum_roll = _group_reduce_max(roll)
        maximum_rollover = _group_reduce_max(rollover)
        minimum_speed = speed.amin(dim=(2, 3))
        control_minimum = torch.minimum(throttle, brake).amin(dim=(2, 3))
        control_maximum = torch.maximum(throttle, brake).amax(dim=(2, 3))
        spatial_variance = prediction.trajectory_total_variance[..., :2]
        maximum_spatial_std = torch.sqrt(spatial_variance.clamp_min(0.0)).amax(
            dim=(2, 3, 4)
        )
        mean_spatial_std = torch.sqrt(
            spatial_variance.mean(dim=(2, 3, 4)).clamp_min(0.0)
        )
        comfort_penalty = 0.5 * (
            _group_reduce_mean(lateral_acceleration)
            / self.config.lateral_acceleration_limit_mps2
            + _group_reduce_mean(yaw_rate) / self.config.yaw_rate_limit_rad_s
        ).clamp(0.0, 1.0)
        stability_penalty = 0.5 * (
            (_group_reduce_mean(roll) / self.config.roll_limit_rad)
            + (_group_reduce_mean(rollover) / self.config.rollover_index_limit)
        ).clamp(0.0, 1.0)
        control_penalty = (0.5 * (throttle.abs() + brake.abs()).mean(dim=(2, 3))).clamp(
            0.0, 1.0
        )
        uncertainty_penalty = (
            mean_spatial_std / self.config.uncertainty_soft_scale_m
        ).clamp(0.0, 1.0)
        chassis_unsafe = (
            (maximum_lateral_acceleration > self.config.lateral_acceleration_limit_mps2)
            | (maximum_yaw_rate > self.config.yaw_rate_limit_rad_s)
            | (maximum_roll > self.config.roll_limit_rad)
            | (maximum_rollover > self.config.rollover_index_limit)
            | (minimum_speed < self.config.minimum_speed_mps)
            | (control_minimum < self.config.control_lower_tolerance)
            | (control_maximum > self.config.control_upper_tolerance)
        )
        uncertainty_unsafe = maximum_spatial_std > self.config.uncertainty_hard_std_m
        device = command.tau_cmd.device
        geometric_reward = torch.from_numpy(geometric.rewards).to(device=device).unsqueeze(0)
        geometric_unsafe = torch.from_numpy(geometric.unsafe).to(device=device).unsqueeze(0)
        total_penalty = (
            self.config.chassis_comfort_weight * comfort_penalty
            + self.config.stability_weight * stability_penalty
            + self.config.control_effort_weight * control_penalty
            + self.config.uncertainty_weight * uncertainty_penalty
        )
        unsafe = geometric_unsafe | chassis_unsafe | uncertainty_unsafe
        hard_count = (
            geometric_unsafe.to(torch.int64)
            + chassis_unsafe.to(torch.int64)
            + uncertainty_unsafe.to(torch.int64)
        ).clamp_min(1)
        safe_reward = (geometric_reward - total_penalty).clamp(-5.0, 5.0)
        unsafe_reward = self.config.unsafe_base_reward - hard_count.to(torch.float32)
        rewards = torch.where(unsafe, unsafe_reward, safe_reward).to(torch.float32).detach()
        components = {
            "geometric_reward": geometric_reward.to(torch.float32).detach(),
            "chassis_comfort_penalty": comfort_penalty.to(torch.float32).detach(),
            "stability_penalty": stability_penalty.to(torch.float32).detach(),
            "control_effort_penalty": control_penalty.to(torch.float32).detach(),
            "uncertainty_penalty": uncertainty_penalty.to(torch.float32).detach(),
            "maximum_lateral_acceleration_mps2": maximum_lateral_acceleration.to(torch.float32).detach(),
            "maximum_yaw_rate_rad_s": maximum_yaw_rate.to(torch.float32).detach(),
            "maximum_abs_roll_rad": maximum_roll.to(torch.float32).detach(),
            "maximum_abs_rollover_index": maximum_rollover.to(torch.float32).detach(),
            "maximum_spatial_std_m": maximum_spatial_std.to(torch.float32).detach(),
            "mean_spatial_std_m": mean_spatial_std.to(torch.float32).detach(),
        }
        return ChassisExecutionRewardResult(
            rewards=rewards,
            unsafe=unsafe.detach(),
            geometric_unsafe=geometric_unsafe.detach(),
            chassis_unsafe=chassis_unsafe.detach(),
            uncertainty_unsafe=uncertainty_unsafe.detach(),
            prediction=prediction,
            components=components,
        )


__all__ = [
    "ChassisExecutionRewardConfig",
    "ChassisExecutionRewardError",
    "ChassisExecutionRewardEvaluator",
    "ChassisExecutionRewardResult",
]
