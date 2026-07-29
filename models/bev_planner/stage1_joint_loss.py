"""Stage 1 joint-mean supervision for the BEV-only diffusion planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.bev_planner.mode_contract import NUM_MODES, TRAJECTORY_STEPS


NUM_ROLES: Final[int] = 3
TRAJECTORY_DIM: Final[int] = 3
PAIR_INDICES: Final[tuple[tuple[int, int], ...]] = ((0, 1), (1, 2), (0, 2))


class Stage1LossError(RuntimeError):
    """Raised when Stage 1 supervision violates the joint-first contract."""


@dataclass(frozen=True)
class Stage1LossConfig:
    """Frozen Stage 1 loss weights and physical discretization."""

    xy_weight: float = 1.0
    heading_weight: float = 0.2
    mode_weight: float = 1.0
    motion_weight: float = 0.05
    pair_weight: float = 0.25
    xy_beta_m: float = 1.0
    heading_beta_rad: float = 0.1
    motion_beta: float = 1.0
    pair_beta_m: float = 1.0
    dt_s: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "xy_weight",
            "heading_weight",
            "mode_weight",
            "motion_weight",
            "pair_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise Stage1LossError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "xy_beta_m",
            "heading_beta_rad",
            "motion_beta",
            "pair_beta_m",
            "dt_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise Stage1LossError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Stage1LossResult:
    """Differentiable loss tensors plus detached-ready scalar metrics."""

    total: Tensor
    local_joint: Tensor
    pair_joint: Tensor
    xy: Tensor
    heading: Tensor
    mode_ce: Tensor
    velocity: Tensor
    acceleration: Tensor
    role_total: Tensor
    role_xy: Tensor
    role_heading: Tensor
    role_mode_ce: Tensor
    role_velocity: Tensor
    role_acceleration: Tensor
    gt_mode_ade: Tensor
    gt_mode_fde: Tensor
    selected_ade: Tensor
    selected_fde: Tensor
    mode_accuracy: Tensor

    def scalar_metrics(self) -> dict[str, float]:
        """Return a detached flat metric mapping for logs and checkpoints."""

        values = {
            "loss/total": self.total,
            "loss/local_joint": self.local_joint,
            "loss/pair_joint": self.pair_joint,
            "loss/xy": self.xy,
            "loss/heading": self.heading,
            "loss/mode_ce": self.mode_ce,
            "loss/velocity": self.velocity,
            "loss/acceleration": self.acceleration,
            "metric/gt_mode_ade": self.gt_mode_ade,
            "metric/gt_mode_fde": self.gt_mode_fde,
            "metric/selected_ade": self.selected_ade,
            "metric/selected_fde": self.selected_fde,
            "metric/mode_accuracy": self.mode_accuracy,
        }
        for role, name in enumerate(("leader", "middle", "rear")):
            values[f"role/{name}_total"] = self.role_total[:, role].mean()
            values[f"role/{name}_xy"] = self.role_xy[:, role].mean()
            values[f"role/{name}_heading"] = self.role_heading[:, role].mean()
            values[f"role/{name}_mode_ce"] = self.role_mode_ce[:, role].mean()
        return {
            name: float(value.detach().to(dtype=torch.float32).cpu())
            for name, value in values.items()
        }


def wrapped_heading_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Return the differentiable signed angle error in ``[-pi, pi]``."""

    difference = predicted - target
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def trajectory_speed_acceleration(
    trajectory_xy: Tensor,
    current_speed_mps: Tensor,
    *,
    dt_s: float,
) -> tuple[Tensor, Tensor]:
    """Compute eight future speeds and accelerations from ego-local XY."""

    origin = torch.zeros_like(trajectory_xy[..., :1, :])
    positions = torch.cat((origin, trajectory_xy), dim=-2)
    displacements = positions[..., 1:, :] - positions[..., :-1, :]
    speeds = torch.linalg.vector_norm(displacements, dim=-1) / float(dt_s)
    speed_history = torch.cat((current_speed_mps.unsqueeze(-1), speeds), dim=-1)
    accelerations = (speed_history[..., 1:] - speed_history[..., :-1]) / float(dt_s)
    return speeds, accelerations


def local_trajectory_to_world_xy(
    trajectory_xy: Tensor,
    ego_pose_global: Tensor,
) -> Tensor:
    """Transform ego-local XY trajectories into the simulator world frame."""

    heading = ego_pose_global[..., 2]
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    local_x = trajectory_xy[..., 0]
    local_y = trajectory_xy[..., 1]
    world_x = ego_pose_global[..., 0].unsqueeze(-1) + cos_h * local_x - sin_h * local_y
    world_y = ego_pose_global[..., 1].unsqueeze(-1) + sin_h * local_x + cos_h * local_y
    return torch.stack((world_x, world_y), dim=-1)


class JointStage1Loss(nn.Module):
    """Compute local imitation and expert-relative joint pair supervision."""

    required_output_fields: Final[frozenset[str]] = frozenset(
        {
            "trajectory_candidates",
            "mode_logits",
            "selected_mode",
            "selected_trajectory",
        }
    )

    def __init__(self, config: Stage1LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or Stage1LossConfig()

    @staticmethod
    def _require_tensor(
        value: Tensor,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
        floating: bool = False,
        finite: bool = False,
        device: torch.device | None = None,
    ) -> None:
        if not isinstance(value, Tensor):
            raise Stage1LossError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise Stage1LossError(
                f"{name} must have shape {shape}, got {tuple(value.shape)}"
            )
        if dtype is not None and value.dtype is not dtype:
            raise Stage1LossError(f"{name} must use dtype {dtype}")
        if floating and not value.is_floating_point():
            raise Stage1LossError(f"{name} must be floating point")
        if device is not None and value.device != device:
            raise Stage1LossError(f"{name} must be on device {device}")
        if finite and not bool(torch.isfinite(value).all()):
            raise Stage1LossError(f"{name} contains non-finite values")

    def _validate(
        self,
        planner_output: Mapping[str, Tensor],
        *,
        expert_trajectory: Tensor,
        gt_mode: Tensor,
        mode_valid_mask: Tensor,
        ego_state: Tensor,
        ego_pose_global: Tensor,
    ) -> tuple[int, torch.device]:
        if not isinstance(planner_output, Mapping):
            raise Stage1LossError("planner_output must be a mapping")
        missing = self.required_output_fields - set(planner_output)
        if missing:
            raise Stage1LossError(
                f"planner_output is missing fields: {sorted(missing)}"
            )
        candidates = planner_output["trajectory_candidates"]
        if not isinstance(candidates, Tensor) or candidates.ndim != 5:
            raise Stage1LossError(
                "trajectory_candidates must have rank five [B,3,10,8,3]"
            )
        batch_size = int(candidates.shape[0])
        if batch_size <= 0:
            raise Stage1LossError("Stage 1 batch must be non-empty")
        device = candidates.device
        self._require_tensor(
            candidates,
            name="trajectory_candidates",
            shape=(batch_size, NUM_ROLES, NUM_MODES, TRAJECTORY_STEPS, TRAJECTORY_DIM),
            floating=True,
            finite=True,
        )
        logits = planner_output["mode_logits"]
        self._require_tensor(
            logits,
            name="mode_logits",
            shape=(batch_size, NUM_ROLES, NUM_MODES),
            floating=True,
            device=device,
        )
        self._require_tensor(
            planner_output["selected_mode"],
            name="selected_mode",
            shape=(batch_size, NUM_ROLES),
            dtype=torch.int64,
            device=device,
        )
        self._require_tensor(
            planner_output["selected_trajectory"],
            name="selected_trajectory",
            shape=(batch_size, NUM_ROLES, TRAJECTORY_STEPS, TRAJECTORY_DIM),
            floating=True,
            finite=True,
            device=device,
        )
        self._require_tensor(
            expert_trajectory,
            name="expert_trajectory",
            shape=(batch_size, NUM_ROLES, TRAJECTORY_STEPS, TRAJECTORY_DIM),
            dtype=torch.float32,
            finite=True,
            device=device,
        )
        self._require_tensor(
            gt_mode,
            name="gt_mode",
            shape=(batch_size, NUM_ROLES),
            dtype=torch.int64,
            device=device,
        )
        self._require_tensor(
            mode_valid_mask,
            name="mode_valid_mask",
            shape=(batch_size, NUM_ROLES, NUM_MODES),
            dtype=torch.bool,
            device=device,
        )
        self._require_tensor(
            ego_state,
            name="ego_state",
            shape=(batch_size, NUM_ROLES, 8),
            dtype=torch.float32,
            finite=True,
            device=device,
        )
        self._require_tensor(
            ego_pose_global,
            name="ego_pose_global",
            shape=(batch_size, NUM_ROLES, 3),
            dtype=torch.float32,
            finite=True,
            device=device,
        )
        if bool(((gt_mode < 0) | (gt_mode >= NUM_MODES)).any()):
            raise Stage1LossError("gt_mode contains an index outside [0,10)")
        if not bool(
            mode_valid_mask.gather(-1, gt_mode.unsqueeze(-1)).squeeze(-1).all()
        ):
            raise Stage1LossError("every gt_mode must be enabled by mode_valid_mask")
        valid_logits = logits[mode_valid_mask]
        invalid_logits = logits[~mode_valid_mask]
        if not bool(torch.isfinite(valid_logits).all()):
            raise Stage1LossError("valid mode logits must be finite")
        if invalid_logits.numel() and not bool(torch.isneginf(invalid_logits).all()):
            raise Stage1LossError("invalid mode logits must be negative infinity")
        selected_mode = planner_output["selected_mode"]
        if bool(((selected_mode < 0) | (selected_mode >= NUM_MODES)).any()):
            raise Stage1LossError("selected_mode contains an index outside [0,10)")
        if not bool(
            mode_valid_mask.gather(-1, selected_mode.unsqueeze(-1)).squeeze(-1).all()
        ):
            raise Stage1LossError("selected_mode must be enabled by mode_valid_mask")
        return batch_size, device

    @staticmethod
    def _gather_gt_candidates(candidates: Tensor, gt_mode: Tensor) -> Tensor:
        gather_index = gt_mode[..., None, None, None].expand(
            -1, -1, 1, TRAJECTORY_STEPS, TRAJECTORY_DIM
        )
        return candidates.gather(2, gather_index).squeeze(2)

    def forward(
        self,
        planner_output: Mapping[str, Tensor],
        *,
        expert_trajectory: Tensor,
        gt_mode: Tensor,
        mode_valid_mask: Tensor,
        ego_state: Tensor,
        ego_pose_global: Tensor,
    ) -> Stage1LossResult:
        self._validate(
            planner_output,
            expert_trajectory=expert_trajectory,
            gt_mode=gt_mode,
            mode_valid_mask=mode_valid_mask,
            ego_state=ego_state,
            ego_pose_global=ego_pose_global,
        )
        config = self.config
        candidates = planner_output["trajectory_candidates"].float()
        expert = expert_trajectory.float()
        predicted = self._gather_gt_candidates(candidates, gt_mode)

        xy_element = F.smooth_l1_loss(
            predicted[..., :2],
            expert[..., :2],
            reduction="none",
            beta=config.xy_beta_m,
        )
        role_xy = xy_element.mean(dim=(-1, -2))
        heading_error = wrapped_heading_error(predicted[..., 2], expert[..., 2])
        role_heading = F.smooth_l1_loss(
            heading_error,
            torch.zeros_like(heading_error),
            reduction="none",
            beta=config.heading_beta_rad,
        ).mean(dim=-1)
        role_mode_ce = F.cross_entropy(
            planner_output["mode_logits"].float().reshape(-1, NUM_MODES),
            gt_mode.reshape(-1),
            reduction="none",
        ).reshape_as(gt_mode)

        predicted_speed, predicted_acceleration = trajectory_speed_acceleration(
            predicted[..., :2],
            ego_state[..., 0].float(),
            dt_s=config.dt_s,
        )
        expert_speed, expert_acceleration = trajectory_speed_acceleration(
            expert[..., :2],
            ego_state[..., 0].float(),
            dt_s=config.dt_s,
        )
        role_velocity = F.smooth_l1_loss(
            predicted_speed,
            expert_speed,
            reduction="none",
            beta=config.motion_beta,
        ).mean(dim=-1)
        role_acceleration = F.smooth_l1_loss(
            predicted_acceleration,
            expert_acceleration,
            reduction="none",
            beta=config.motion_beta,
        ).mean(dim=-1)

        role_total = (
            config.xy_weight * role_xy
            + config.heading_weight * role_heading
            + config.mode_weight * role_mode_ce
            + config.motion_weight * (role_velocity + role_acceleration)
        )
        local_joint = role_total.mean()

        predicted_world = local_trajectory_to_world_xy(
            predicted[..., :2], ego_pose_global.float()
        )
        expert_world = local_trajectory_to_world_xy(
            expert[..., :2], ego_pose_global.float()
        )
        pair_losses = []
        for first, second in PAIR_INDICES:
            predicted_relative = predicted_world[:, first] - predicted_world[:, second]
            expert_relative = expert_world[:, first] - expert_world[:, second]
            pair_losses.append(
                F.smooth_l1_loss(
                    predicted_relative,
                    expert_relative,
                    reduction="none",
                    beta=config.pair_beta_m,
                ).mean(dim=(-1, -2))
            )
        pair_joint = torch.stack(pair_losses, dim=-1).mean()
        total = local_joint + config.pair_weight * pair_joint
        if not bool(torch.isfinite(total)):
            raise Stage1LossError("Stage 1 total loss is non-finite")

        gt_distance = torch.linalg.vector_norm(
            predicted[..., :2] - expert[..., :2], dim=-1
        )
        selected = planner_output["selected_trajectory"].float()
        selected_distance = torch.linalg.vector_norm(
            selected[..., :2] - expert[..., :2], dim=-1
        )
        selected_mode = planner_output["selected_mode"]
        return Stage1LossResult(
            total=total,
            local_joint=local_joint,
            pair_joint=pair_joint,
            xy=role_xy.mean(),
            heading=role_heading.mean(),
            mode_ce=role_mode_ce.mean(),
            velocity=role_velocity.mean(),
            acceleration=role_acceleration.mean(),
            role_total=role_total,
            role_xy=role_xy,
            role_heading=role_heading,
            role_mode_ce=role_mode_ce,
            role_velocity=role_velocity,
            role_acceleration=role_acceleration,
            gt_mode_ade=gt_distance.mean(),
            gt_mode_fde=gt_distance[..., -1].mean(),
            selected_ade=selected_distance.mean(),
            selected_fde=selected_distance[..., -1].mean(),
            mode_accuracy=(selected_mode == gt_mode).float().mean(),
        )


__all__ = [
    "JointStage1Loss",
    "PAIR_INDICES",
    "Stage1LossConfig",
    "Stage1LossError",
    "Stage1LossResult",
    "local_trajectory_to_world_xy",
    "trajectory_speed_acceleration",
    "wrapped_heading_error",
]
