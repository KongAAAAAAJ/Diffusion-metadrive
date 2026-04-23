"""Project-level platoon environment entrypoints."""

try:
    from .platoon_env import PlatoonEnv, PlatoonEnvConfig
except Exception:  # pragma: no cover - keep package import lightweight
    PlatoonEnv = None  # type: ignore
    PlatoonEnvConfig = None  # type: ignore

from .co_preference_platoon_env import CoPreferencePlatoonEnv

__all__ = ["CoPreferencePlatoonEnv", "PlatoonEnv", "PlatoonEnvConfig"]
