"""CF-6 adapter joining raw joint GRPO actions to the frozen chassis surrogate.

The probability action is always the diffusion rollout trajectory ``tau_d``.
The kinematic optimizer is a detached environment-boundary transform producing
``tau_cmd``.  Only ``tau_cmd`` enters the frozen surrogate, and the resulting
gradient-free rewards update the original rollout exactly once.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from models.bev_planner.joint_grpo import (
    JointGRPORollout,
    JointGRPOUpdateResult,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizer,
    TrajectoryOptimizationResult,
)

from .contracts import (
    CONTROLLER_CONTEXT_FIELDS,
    INITIAL_STATE_FIELDS,
    NUM_GROUPS,
    NUM_ROLES,
    VEHICLE_CONDITION_FIELDS,
    ChassisExecutionCommand,
    ChassisExecutionContractError,
)
from .reward import (
    ChassisExecutionRewardEvaluator,
    ChassisExecutionRewardResult,
)


class ChassisFusionGRPOError(RuntimeError):
    """Raised when the CF-6 probability/execution boundary is violated."""


def _require_context_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != dtype:
        raise ChassisFusionGRPOError(f"{name} must be a {dtype} tensor")
    if tuple(value.shape) != shape:
        raise ChassisFusionGRPOError(f"{name} must have shape {shape}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ChassisFusionGRPOError(f"{name} must be finite")
    if value.requires_grad:
        raise ChassisFusionGRPOError(f"{name} must not require gradients")
    return value


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _policy_sha256(trainer: object) -> str:
    planner = getattr(trainer, "planner", None)
    if not isinstance(planner, torch.nn.Module):
        raise ChassisFusionGRPOError("trainer must expose a torch planner")
    digest = hashlib.sha256()
    for prefix, module in (
        ("diffusion_decoder", getattr(planner, "diffusion_decoder", None)),
        ("mode_head", getattr(planner, "mode_head", None)),
    ):
        if not isinstance(module, torch.nn.Module):
            raise ChassisFusionGRPOError(
                "trainer planner must expose diffusion_decoder and mode_head"
            )
        digest.update(prefix.encode("utf-8"))
        digest.update(_module_sha256(module).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ChassisExecutionContext:
    """The non-trajectory state shared by all four candidates."""

    initial_state: Tensor
    vehicle_condition: Tensor
    controller_context: Tensor
    controller_mode: Tensor
    agent_role: Tensor

    def __post_init__(self) -> None:
        _require_context_tensor(
            self.initial_state,
            name="initial_state",
            dtype=torch.float32,
            shape=(1, NUM_ROLES, len(INITIAL_STATE_FIELDS)),
        )
        _require_context_tensor(
            self.vehicle_condition,
            name="vehicle_condition",
            dtype=torch.float32,
            shape=(1, NUM_ROLES, len(VEHICLE_CONDITION_FIELDS)),
        )
        _require_context_tensor(
            self.controller_context,
            name="controller_context",
            dtype=torch.float32,
            shape=(1, NUM_ROLES, len(CONTROLLER_CONTEXT_FIELDS)),
        )
        _require_context_tensor(
            self.controller_mode,
            name="controller_mode",
            dtype=torch.int64,
            shape=(1, NUM_ROLES),
        )
        _require_context_tensor(
            self.agent_role,
            name="agent_role",
            dtype=torch.int64,
            shape=(1, NUM_ROLES),
        )
        expected = torch.arange(
            NUM_ROLES, dtype=torch.int64, device=self.agent_role.device
        ).unsqueeze(0)
        if not bool(torch.equal(self.agent_role, expected)):
            raise ChassisFusionGRPOError("agent_role must equal [[0,1,2]]")

    def command(self, tau_cmd: Tensor) -> ChassisExecutionCommand:
        device = tau_cmd.device
        try:
            return ChassisExecutionCommand(
                tau_cmd=tau_cmd,
                initial_state=self.initial_state.to(device),
                vehicle_condition=self.vehicle_condition.to(device),
                controller_context=self.controller_context.to(device),
                controller_mode=self.controller_mode.to(device),
                agent_role=self.agent_role.to(device),
                source="tau_cmd",
            )
        except ChassisExecutionContractError as exc:
            raise ChassisFusionGRPOError(
                "unable to construct the strict tau_cmd surrogate command"
            ) from exc


@dataclass(frozen=True)
class ChassisFusionGRPOStepResult:
    rollout: JointGRPORollout
    tau_d: Tensor
    tau_cmd: Tensor
    optimization: TrajectoryOptimizationResult
    reward: ChassisExecutionRewardResult
    update: JointGRPOUpdateResult
    selected_group: Tensor
    selected_tau_cmd: Tensor
    policy_sha256_before: str
    policy_sha256_after: str
    surrogate_sha256: str
    optimizer_config_sha256: str
    metadrive_candidate_branches: int

    def __post_init__(self) -> None:
        expected = (1, NUM_GROUPS, NUM_ROLES, 8, 3)
        if (
            self.tau_d.dtype != torch.float32
            or self.tau_cmd.dtype != torch.float32
            or tuple(self.tau_d.shape) != expected
            or tuple(self.tau_cmd.shape) != expected
        ):
            raise ChassisFusionGRPOError(
                "tau_d and tau_cmd must be float32 [1,4,3,8,3]"
            )
        if self.tau_d.requires_grad or self.tau_cmd.requires_grad:
            raise ChassisFusionGRPOError("tau_d and tau_cmd must be detached")
        if self.selected_group.dtype != torch.int64 or tuple(
            self.selected_group.shape
        ) != (1,):
            raise ChassisFusionGRPOError("selected_group must be int64 [1]")
        if self.selected_tau_cmd.dtype != torch.float32 or tuple(
            self.selected_tau_cmd.shape
        ) != (1, NUM_ROLES, 8, 3):
            raise ChassisFusionGRPOError(
                "selected_tau_cmd must be float32 [1,3,8,3]"
            )
        if self.metadrive_candidate_branches != 0:
            raise ChassisFusionGRPOError(
                "candidate MetaDrive branch execution is forbidden"
            )


@dataclass(frozen=True)
class ChassisFusionCandidateResult:
    """Gradient-free candidate evaluation used by CF-7 inference."""

    rollout: JointGRPORollout
    tau_d: Tensor
    tau_cmd: Tensor
    optimization: TrajectoryOptimizationResult
    reward: ChassisExecutionRewardResult
    selected_group: Tensor
    selected_tau_cmd: Tensor
    policy_sha256: str
    surrogate_sha256: str
    optimizer_config_sha256: str
    metadrive_candidate_branches: int
    sampling_ms: float
    reward_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        expected = (1, NUM_GROUPS, NUM_ROLES, 8, 3)
        if (
            self.tau_d.dtype != torch.float32
            or self.tau_cmd.dtype != torch.float32
            or tuple(self.tau_d.shape) != expected
            or tuple(self.tau_cmd.shape) != expected
            or self.tau_d.requires_grad
            or self.tau_cmd.requires_grad
        ):
            raise ChassisFusionGRPOError(
                "candidate tau_d/tau_cmd must be detached float32 [1,4,3,8,3]"
            )
        if self.selected_group.dtype != torch.int64 or tuple(
            self.selected_group.shape
        ) != (1,):
            raise ChassisFusionGRPOError("selected_group must be int64 [1]")
        if self.selected_tau_cmd.dtype != torch.float32 or tuple(
            self.selected_tau_cmd.shape
        ) != (1, NUM_ROLES, 8, 3):
            raise ChassisFusionGRPOError(
                "selected_tau_cmd must be float32 [1,3,8,3]"
            )
        timings = (self.sampling_ms, self.reward_ms, self.total_ms)
        if any(not np.isfinite(value) or value < 0.0 for value in timings):
            raise ChassisFusionGRPOError("candidate timings must be finite and non-negative")
        if self.metadrive_candidate_branches != 0:
            raise ChassisFusionGRPOError(
                "candidate MetaDrive branch execution is forbidden"
            )


class ChassisFusionGRPOAdapter:
    """Run one execution-aware GRPO update and expose one executable action."""

    def __init__(
        self,
        trainer: object,
        trajectory_optimizer: KinematicTrajectoryOptimizer,
        reward_evaluator: ChassisExecutionRewardEvaluator,
    ) -> None:
        for name in ("sample_groups", "update"):
            if not callable(getattr(trainer, name, None)):
                raise ChassisFusionGRPOError(f"trainer must implement {name}()")
        if not isinstance(trajectory_optimizer, KinematicTrajectoryOptimizer):
            raise ChassisFusionGRPOError(
                "trajectory_optimizer must be KinematicTrajectoryOptimizer"
            )
        if not isinstance(reward_evaluator, ChassisExecutionRewardEvaluator):
            raise ChassisFusionGRPOError(
                "reward_evaluator must be ChassisExecutionRewardEvaluator"
            )
        self.trainer = trainer
        self.trajectory_optimizer = trajectory_optimizer
        self.reward_evaluator = reward_evaluator
        self._executed_results: set[int] = set()

    @staticmethod
    def _validate_model_inputs(model_inputs: Mapping[str, Tensor]) -> None:
        required = {"ego_state", "coarse_trajectories", "mode_valid_mask"}
        if not isinstance(model_inputs, Mapping) or not required.issubset(model_inputs):
            raise ChassisFusionGRPOError(
                "model_inputs must contain ego_state, coarse_trajectories and mode_valid_mask"
            )
        ego = model_inputs["ego_state"]
        coarse = model_inputs["coarse_trajectories"]
        if (
            not isinstance(ego, Tensor)
            or ego.dtype != torch.float32
            or tuple(ego.shape) != (1, NUM_ROLES, 8)
            or not bool(torch.isfinite(ego).all())
            or bool((ego[..., 0] < 0.0).any())
        ):
            raise ChassisFusionGRPOError(
                "ego_state must be finite float32 [1,3,8] with non-negative speed"
            )
        if (
            not isinstance(coarse, Tensor)
            or coarse.dtype != torch.float32
            or tuple(coarse.shape) != (1, NUM_ROLES, 10, 8, 3)
            or not bool(torch.isfinite(coarse).all())
        ):
            raise ChassisFusionGRPOError(
                "coarse_trajectories must be finite float32 [1,3,10,8,3]"
            )

    def run_update(
        self,
        *,
        env: object,
        trainer_model_inputs: Mapping[str, Tensor],
        reward_model_inputs: object,
        chassis_context: ChassisExecutionContext,
        generator: torch.Generator,
    ) -> ChassisFusionGRPOStepResult:
        """Sample raw actions, transform them, score execution, and update once."""

        policy_before = _policy_sha256(self.trainer)
        optimizer_step_before = int(getattr(self.trainer, "optimizer_step", -1))
        candidate = self.evaluate_candidates(
            env=env,
            trainer_model_inputs=trainer_model_inputs,
            reward_model_inputs=reward_model_inputs,
            chassis_context=chassis_context,
            generator=generator,
        )
        update = self.trainer.update(candidate.rollout, candidate.reward.rewards)
        if int(update.optimizer_step) != optimizer_step_before + 1:
            raise ChassisFusionGRPOError(
                "CF-6 must execute exactly one GRPO optimizer step"
            )
        surrogate_after = _module_sha256(self.reward_evaluator.surrogate)
        if surrogate_after != candidate.surrogate_sha256:
            raise ChassisFusionGRPOError("frozen chassis surrogate changed during GRPO update")
        if self.trajectory_optimizer.config.sha256() != candidate.optimizer_config_sha256:
            raise ChassisFusionGRPOError(
                "trajectory optimizer changed during GRPO update"
            )
        policy_after = _policy_sha256(self.trainer)
        return ChassisFusionGRPOStepResult(
            rollout=candidate.rollout,
            tau_d=candidate.tau_d,
            tau_cmd=candidate.tau_cmd,
            optimization=candidate.optimization,
            reward=candidate.reward,
            update=update,
            selected_group=candidate.selected_group,
            selected_tau_cmd=candidate.selected_tau_cmd,
            policy_sha256_before=policy_before,
            policy_sha256_after=policy_after,
            surrogate_sha256=surrogate_after,
            optimizer_config_sha256=candidate.optimizer_config_sha256,
            metadrive_candidate_branches=candidate.metadrive_candidate_branches,
        )

    @staticmethod
    def _synchronize(tensor: Tensor) -> None:
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)

    def evaluate_candidates(
        self,
        *,
        env: object,
        trainer_model_inputs: Mapping[str, Tensor],
        reward_model_inputs: object,
        chassis_context: ChassisExecutionContext,
        generator: torch.Generator,
    ) -> ChassisFusionCandidateResult:
        """Evaluate G raw candidates without updating policy or stepping MetaDrive."""

        self._validate_model_inputs(trainer_model_inputs)
        if not isinstance(generator, torch.Generator):
            raise ChassisFusionGRPOError("an explicit torch.Generator is required")
        policy_before = _policy_sha256(self.trainer)
        surrogate_before = _module_sha256(self.reward_evaluator.surrogate)
        optimizer_sha = self.trajectory_optimizer.config.sha256()
        reward_count_before = self.reward_evaluator.evaluation_count
        branch_count_before = self.reward_evaluator.metadrive_candidate_branch_count
        timing_tensor = trainer_model_inputs["ego_state"]
        self._synchronize(timing_tensor)
        total_start = time.perf_counter()
        sampling_start = total_start
        rollout = self.trainer.sample_groups(
            trainer_model_inputs, generator=generator
        )
        self._synchronize(timing_tensor)
        sampling_ms = (time.perf_counter() - sampling_start) * 1000.0
        if not isinstance(rollout, JointGRPORollout):
            raise ChassisFusionGRPOError(
                "trainer.sample_groups must return JointGRPORollout"
            )
        tau_d = rollout.selected_trajectories.detach().to(torch.float32).clone()
        if tuple(tau_d.shape) != (1, NUM_GROUPS, NUM_ROLES, 8, 3):
            raise ChassisFusionGRPOError(
                "online chassis-fusion rollout must contain one state and four joint candidates"
            )
        raw_snapshot = tau_d.clone()
        coarse = trainer_model_inputs["coarse_trajectories"].detach().cpu().numpy()
        coarse = np.repeat(coarse[:, None], NUM_GROUPS, axis=1)
        speeds = (
            trainer_model_inputs["ego_state"][..., 0]
            .detach()
            .cpu()
            .numpy()[:, None, :]
        )
        modes = rollout.sampled_modes.detach().cpu().numpy()
        optimization = self.trajectory_optimizer.optimize(
            tau_d.cpu().numpy(), coarse, speeds, modes
        )
        if optimization.config_sha256 != optimizer_sha:
            raise ChassisFusionGRPOError(
                "trajectory optimizer config changed during action transformation"
            )
        if not np.array_equal(optimization.raw_trajectories, tau_d.cpu().numpy()):
            raise ChassisFusionGRPOError("trajectory optimizer mutated raw tau_d")
        tau_cmd = torch.from_numpy(
            np.array(optimization.optimized_trajectories, copy=True)
        ).to(device=tau_d.device, dtype=torch.float32)
        command = chassis_context.command(tau_cmd)
        self._synchronize(timing_tensor)
        reward_start = time.perf_counter()
        reward = self.reward_evaluator.score(env, reward_model_inputs, command)
        self._synchronize(timing_tensor)
        reward_ms = (time.perf_counter() - reward_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        if reward.rewards.requires_grad:
            raise ChassisFusionGRPOError("surrogate reward must be gradient-free")
        if not torch.equal(tau_d, raw_snapshot) or not torch.equal(
            rollout.selected_trajectories, raw_snapshot
        ):
            raise ChassisFusionGRPOError("raw rollout tau_d changed during evaluation")
        if self.reward_evaluator.evaluation_count != reward_count_before + 1:
            raise ChassisFusionGRPOError(
                "candidate evaluation must execute the frozen surrogate exactly once"
            )
        branches = self.reward_evaluator.metadrive_candidate_branch_count - branch_count_before
        if branches != 0:
            raise ChassisFusionGRPOError(
                "candidate evaluation attempted a MetaDrive branch rollout"
            )
        surrogate_after = _module_sha256(self.reward_evaluator.surrogate)
        if surrogate_after != surrogate_before:
            raise ChassisFusionGRPOError("frozen chassis surrogate changed during evaluation")
        policy_after = _policy_sha256(self.trainer)
        if policy_after != policy_before:
            raise ChassisFusionGRPOError("policy changed during inference-only evaluation")
        selected_group = reward.rewards.argmax(dim=1).to(torch.int64)
        row = torch.arange(1, device=tau_cmd.device)
        selected_tau_cmd = tau_cmd[row, selected_group].detach().clone()
        return ChassisFusionCandidateResult(
            rollout=rollout,
            tau_d=tau_d,
            tau_cmd=tau_cmd.detach().clone(),
            optimization=optimization,
            reward=reward,
            selected_group=selected_group.detach().clone(),
            selected_tau_cmd=selected_tau_cmd,
            policy_sha256=policy_after,
            surrogate_sha256=surrogate_after,
            optimizer_config_sha256=optimizer_sha,
            metadrive_candidate_branches=branches,
            sampling_ms=sampling_ms,
            reward_ms=reward_ms,
            total_ms=total_ms,
        )

    def execute_selected_once(
        self,
        env: object,
        result: ChassisFusionGRPOStepResult | ChassisFusionCandidateResult,
    ) -> Any:
        """Execute only the best optimized joint command in MetaDrive once."""

        result_id = id(result)
        if result_id in self._executed_results:
            raise ChassisFusionGRPOError(
                "a CF-6 selected command may call env.step exactly once"
            )
        step = getattr(env, "step", None)
        if not callable(step):
            raise ChassisFusionGRPOError("environment must implement step(actions)")
        trajectories = result.selected_tau_cmd[0].detach().cpu().numpy()
        actions = {
            f"agent{role}": np.ascontiguousarray(trajectories[role], dtype=np.float32)
            for role in range(NUM_ROLES)
        }
        # Consume before invoking external state mutation.  Even when env.step
        # raises, retrying the same result would constitute a second physical
        # execution attempt and is therefore forbidden.
        self._executed_results.add(result_id)
        transition = step(actions)
        return transition


__all__ = [
    "ChassisExecutionContext",
    "ChassisFusionCandidateResult",
    "ChassisFusionGRPOAdapter",
    "ChassisFusionGRPOError",
    "ChassisFusionGRPOStepResult",
]
