"""Online BEV joint GRPO training package.

Public symbols are loaded lazily from :mod:`.train` to keep package import light
and avoid eager execution when launching ``python -m train.bev_joint_grpo_online.train``.
"""

from __future__ import annotations

__all__ = [
    "DEVELOPMENT_SEEDS",
    "HOLDOUT_SEEDS",
    "PRIMARY_S5_S9_SCENARIOS",
    "JointGRPOOnlineConfig",
    "JointGRPOTrainingConfig",
    "OnlineGRPOError",
    "constant_velocity_actions",
    "episode_has_ended",
    "execute_cached_frozen_baseline",
    "build_grpo_validation_state_bank",
    "joint_trajectory_action",
    "model_inputs_to_batch",
    "optimize_selected_model_trajectories",
    "run_joint_grpo_training",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import train as _train
    return getattr(_train, name)
