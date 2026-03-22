from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


class _EpisodeAccumulator:
    def __init__(self) -> None:
        self.success = False
        self.collision = False
        self.formation_errors: list[float] = []
        self.min_gap = float("inf")
        self.recovery_time_steps: int | None = None
        self._above_threshold_steps = 0


class PlatoonMetrics:
    """Episode-level platoon metric accumulator for Phase 1 verification."""

    def __init__(self, formation_error_threshold: float = 2.0):
        self.formation_error_threshold = float(formation_error_threshold)
        self._episodes: list[dict[str, float | bool]] = []
        self._current: _EpisodeAccumulator | None = None

    def start_episode(self) -> None:
        self._current = _EpisodeAccumulator()

    def update(self, info: Mapping[str, Mapping[str, object]]) -> None:
        if self._current is None:
            self.start_episode()
        assert self._current is not None

        per_agent_infos = [agent_info for agent_id, agent_info in info.items() if agent_id != "__all__"]
        if not per_agent_infos:
            return

        self._current.success = self._current.success or all(bool(agent.get("arrive_dest", False)) for agent in per_agent_infos)
        self._current.collision = self._current.collision or any(
            bool(agent.get("crash", False) or agent.get("crash_vehicle", False)) for agent in per_agent_infos
        )

        step_formation_error = float(
            np.mean([float(agent.get("formation_error", 0.0)) for agent in per_agent_infos], dtype=np.float32)
        )
        self._current.formation_errors.append(step_formation_error)

        step_min_gap = min(float(agent.get("min_gap", float("inf"))) for agent in per_agent_infos)
        self._current.min_gap = min(self._current.min_gap, step_min_gap)

        if step_formation_error > self.formation_error_threshold:
            self._current._above_threshold_steps += 1
        elif self._current._above_threshold_steps > 0 and self._current.recovery_time_steps is None:
            self._current.recovery_time_steps = self._current._above_threshold_steps

    def end_episode(self) -> None:
        if self._current is None:
            return

        formation_error = (
            float(np.mean(self._current.formation_errors, dtype=np.float32))
            if self._current.formation_errors else 0.0
        )
        recovery_time = (
            float(self._current.recovery_time_steps)
            if self._current.recovery_time_steps is not None else 0.0
        )
        min_gap = 0.0 if not np.isfinite(self._current.min_gap) else float(self._current.min_gap)
        self._episodes.append(
            {
                "success": bool(self._current.success),
                "collision": bool(self._current.collision),
                "formation_error": formation_error,
                "recovery_time": recovery_time,
                "min_inter_vehicle_gap": min_gap,
            }
        )
        self._current = None

    def compute(self) -> Dict[str, float]:
        if not self._episodes:
            return {
                "success_rate": 0.0,
                "collision_rate": 0.0,
                "formation_error": 0.0,
                "recovery_time": 0.0,
                "min_inter_vehicle_gap": 0.0,
            }

        success = np.asarray([float(ep["success"]) for ep in self._episodes], dtype=np.float32)
        collision = np.asarray([float(ep["collision"]) for ep in self._episodes], dtype=np.float32)
        formation_error = np.asarray([float(ep["formation_error"]) for ep in self._episodes], dtype=np.float32)
        recovery_time = np.asarray([float(ep["recovery_time"]) for ep in self._episodes], dtype=np.float32)
        min_gap = np.asarray([float(ep["min_inter_vehicle_gap"]) for ep in self._episodes], dtype=np.float32)

        return {
            "success_rate": float(success.mean()),
            "collision_rate": float(collision.mean()),
            "formation_error": float(formation_error.mean()),
            "recovery_time": float(recovery_time.mean()),
            "min_inter_vehicle_gap": float(min_gap.min()),
        }

    @property
    def episode_count(self) -> int:
        return len(self._episodes)
