"""Project-level platoon environment entrypoints."""

try:
    from .platoon_env import PlatoonEnv, PlatoonEnvConfig
except Exception:  # pragma: no cover - keep package import lightweight
    PlatoonEnv = None  # type: ignore
    PlatoonEnvConfig = None  # type: ignore

from .wrap_platoon_env import ModeSelectionSB3Env
from evaluation.platoon_metrics import PlatoonMetrics
from .reward_terms import compute_step_reward, compute_team_reward, compute_trajectory_reward

__all__ = [
    "ModeSelectionSB3Env",
    "PlatoonEnv",
    "PlatoonEnvConfig",
    "PlatoonMetrics",
    "compute_step_reward",
    "compute_team_reward",
    "compute_trajectory_reward",
]
