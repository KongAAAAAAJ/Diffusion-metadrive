"""Project environment entrypoints without eager simulator/training imports."""

from __future__ import annotations

from typing import Any


__all__ = [
    "ModeSelectionSB3Env",
    "PlatoonEnv",
    "PlatoonEnvConfig",
    "PlatoonMetrics",
    "compute_step_reward",
    "compute_team_reward",
    "compute_trajectory_reward",
]


def __getattr__(name: str) -> Any:
    if name in {"PlatoonEnv", "PlatoonEnvConfig"}:
        from envs.platoon_env import PlatoonEnv, PlatoonEnvConfig

        return {"PlatoonEnv": PlatoonEnv, "PlatoonEnvConfig": PlatoonEnvConfig}[name]
    if name == "ModeSelectionSB3Env":
        from envs.wrap_platoon_env import ModeSelectionSB3Env

        return ModeSelectionSB3Env
    if name == "PlatoonMetrics":
        from evaluation.platoon_metrics import PlatoonMetrics

        return PlatoonMetrics
    if name in {
        "compute_step_reward",
        "compute_team_reward",
        "compute_trajectory_reward",
    }:
        from envs.reward_terms import (
            compute_step_reward,
            compute_team_reward,
            compute_trajectory_reward,
        )

        return {
            "compute_step_reward": compute_step_reward,
            "compute_team_reward": compute_team_reward,
            "compute_trajectory_reward": compute_trajectory_reward,
        }[name]
    raise AttributeError(name)
