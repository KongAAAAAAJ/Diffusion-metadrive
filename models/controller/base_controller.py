"""Base interfaces for trajectory tracking controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseController(ABC):
    """Common low-level controller interface for platoon agents."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def reset(self) -> None:
        """Clear controller state at episode boundaries."""

    @abstractmethod
    def compute_actions(self, env, trajectories_world: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return low-level `{agent_id: [steering, throttle]}` actions."""
