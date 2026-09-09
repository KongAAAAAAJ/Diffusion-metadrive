"""Typed results and reusable geometry context for reward evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import JointRewardError
from .constants import NUM_MODES, NUM_ROLES

class VehicleModePretrainRewardResult:
    """Frozen same-mode target rewards cached once for one live state."""

    rewards: np.ndarray
    valid_mode_mask: np.ndarray
    unsafe: np.ndarray
    collision: np.ndarray
    out_of_drivable: np.ndarray
    clearance_violation: np.ndarray
    components: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        expected = (NUM_ROLES, NUM_MODES)
        rewards = np.asarray(self.rewards)
        if (
            rewards.shape != expected
            or rewards.dtype != np.float32
            or not np.isfinite(rewards).all()
        ):
            raise JointRewardError(
                "pretrain rewards must be finite float32 [3,10]"
            )
        valid = np.asarray(self.valid_mode_mask)
        if valid.shape != expected or valid.dtype != np.bool_:
            raise JointRewardError("valid_mode_mask must be bool [3,10]")
        for name in (
            "unsafe",
            "collision",
            "out_of_drivable",
            "clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != expected or value.dtype != np.bool_:
                raise JointRewardError(f"pretrain {name} must be bool [3,10]")
        for name, value in self.components.items():
            array = np.asarray(value)
            if array.shape != expected or not np.isfinite(array).all():
                raise JointRewardError(
                    f"pretrain reward component {name} must be finite [3,10]"
                )


@dataclass(frozen=True)
class RewardGeometryContext:
    """Immutable geometry shared by every reward pass for one live state.

    The context is intentionally process-local: it is valid only while the
    simulator state, BEV observation, and frozen teammate trajectories remain
    the state used to build it.  It avoids rebuilding background predictions
    and signed-distance fields for same-state pretrain/current/frozen scoring.
    """

    bev: np.ndarray
    poses: tuple[np.ndarray, ...]
    backgrounds: tuple[
        Mapping[str, list[tuple[str, np.ndarray, tuple[float, float]]]], ...
    ]
    road_fields: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class VehicleModeRewardResult:
    """Current and frozen same-mode rewards under identical teammate context."""

    rewards: np.ndarray
    pretrain_rewards: np.ndarray
    valid_mode_mask: np.ndarray
    unsafe: np.ndarray
    collision: np.ndarray
    out_of_drivable: np.ndarray
    clearance_violation: np.ndarray
    pretrain_unsafe: np.ndarray
    pretrain_collision: np.ndarray
    pretrain_out_of_drivable: np.ndarray
    pretrain_clearance_violation: np.ndarray
    components: Mapping[str, np.ndarray]
    pretrain_components: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        rewards = np.asarray(self.rewards)
        expected_prefix = (NUM_ROLES, NUM_MODES)
        if (
            rewards.ndim != 3
            or rewards.shape[:2] != expected_prefix
            or rewards.shape[2] <= 0
            or rewards.dtype != np.float32
            or not np.isfinite(rewards).all()
        ):
            raise JointRewardError("rewards must be finite float32 [3,10,N]")
        pretrain = np.asarray(self.pretrain_rewards)
        if (
            pretrain.shape != expected_prefix
            or pretrain.dtype != np.float32
            or not np.isfinite(pretrain).all()
        ):
            raise JointRewardError(
                "pretrain_rewards must be finite float32 [3,10]"
            )
        valid = np.asarray(self.valid_mode_mask)
        if valid.shape != expected_prefix or valid.dtype != np.bool_:
            raise JointRewardError("valid_mode_mask must be bool [3,10]")
        current_shape = rewards.shape
        for name in (
            "unsafe",
            "collision",
            "out_of_drivable",
            "clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != current_shape or value.dtype != np.bool_:
                raise JointRewardError(f"{name} must be bool [3,10,N]")
        for name in (
            "pretrain_unsafe",
            "pretrain_collision",
            "pretrain_out_of_drivable",
            "pretrain_clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != expected_prefix or value.dtype != np.bool_:
                raise JointRewardError(f"{name} must be bool [3,10]")
        for name, value in self.components.items():
            array = np.asarray(value)
            if array.shape != current_shape or not np.isfinite(array).all():
                raise JointRewardError(
                    f"vehicle reward component {name} must be finite [3,10,N]"
                )
        for name, value in self.pretrain_components.items():
            array = np.asarray(value)
            if array.shape != expected_prefix or not np.isfinite(array).all():
                raise JointRewardError(
                    f"pretrain reward component {name} must be finite [3,10]"
                )
