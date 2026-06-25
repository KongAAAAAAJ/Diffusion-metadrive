"""
判断编队是否从 locked 模式解锁
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RiskDetector(ABC):
    """Interface for platoon unlock/risk detection."""

    @abstractmethod
    def detect(self, env, agent_ids: list[str], traffic_vehicles: list) -> dict:
        """Return structured risk info.

        Expected keys:
            triggered (bool): whether a risk has been detected.
            reasons (list[str]): human-readable trigger reasons.
            per_agent (dict[str, dict]): optional per-agent details.
        """


class SimpleRuleRiskDetector(RiskDetector):
    """Simple unlock detector based on nearby traffic gaps.

    This is intentionally lightweight. We keep the interface stable so a more
    advanced risk model can replace it later without changing rule_maker.
    """

    def __init__(
        self,
        *,
        front_gap_trigger_m: float = 12.0,
        side_gap_trigger_m: float = 6.0,
    ) -> None:
        self.front_gap_trigger_m = float(front_gap_trigger_m)
        self.side_gap_trigger_m = float(side_gap_trigger_m)

    def detect(self, env, agent_ids: list[str], traffic_vehicles: list) -> dict:
        agents = getattr(env, "agents", {}) or {}
        reasons: list[str] = []
        per_agent: dict[str, dict] = {}

        for agent_id in agent_ids:
            vehicle = agents.get(agent_id)
            if vehicle is None:
                continue
            obs = self._nearest_background_observation(vehicle, traffic_vehicles)
            agent_reasons: list[str] = []

            front_dist = obs["front"]["distance_m"] if obs["front"] is not None else None
            left_dist = obs["left"]["distance_m"] if obs["left"] is not None else None
            right_dist = obs["right"]["distance_m"] if obs["right"] is not None else None

            if self.front_gap_trigger_m > 0.0 and front_dist is not None and front_dist <= self.front_gap_trigger_m:
                agent_reasons.append(f"front_gap<={self.front_gap_trigger_m:.1f}m")
            if self.side_gap_trigger_m > 0.0 and left_dist is not None and left_dist <= self.side_gap_trigger_m:
                agent_reasons.append(f"left_gap<={self.side_gap_trigger_m:.1f}m")
            if self.side_gap_trigger_m > 0.0 and right_dist is not None and right_dist <= self.side_gap_trigger_m:
                agent_reasons.append(f"right_gap<={self.side_gap_trigger_m:.1f}m")

            per_agent[agent_id] = {
                "front_distance_m": front_dist,
                "left_distance_m": left_dist,
                "right_distance_m": right_dist,
                "reasons": list(agent_reasons),
            }
            for reason in agent_reasons:
                reasons.append(f"{agent_id}:{reason}")

        return {
            "triggered": bool(reasons),
            "reasons": reasons,
            "per_agent": per_agent,
        }

    @staticmethod
    def _nearest_background_observation(vehicle, traffic_vehicles: list) -> dict:
        nearest = {"front": None, "back": None, "left": None, "right": None}
        best = {key: float("inf") for key in nearest}
        ego_pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        ego_heading = float(getattr(vehicle, "heading_theta", 0.0))

        for traffic_vehicle in traffic_vehicles:
            if traffic_vehicle is vehicle:
                continue
            rel = SimpleRuleRiskDetector._world_to_ego_local_from_pose(
                ego_pos,
                ego_heading,
                np.asarray(getattr(traffic_vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32),
            )
            lon, lat = float(rel[0]), float(rel[1])
            if abs(lat) <= 2.0:
                key = "front" if lon >= 0.0 else "back"
                dist = abs(lon)
            elif lat > 0.0:
                key = "left"
                dist = float(np.linalg.norm(rel))
            else:
                key = "right"
                dist = float(np.linalg.norm(rel))
            if dist < best[key]:
                best[key] = dist
                nearest[key] = {
                    "vehicle": traffic_vehicle,
                    "distance_m": dist,
                }
        return nearest

    @staticmethod
    def _world_to_ego_local_from_pose(ego_pos: np.ndarray, heading: float, target_world: np.ndarray) -> np.ndarray:
        delta = np.asarray(target_world[:2], dtype=np.float32) - np.asarray(ego_pos[:2], dtype=np.float32)
        cos_h, sin_h = np.cos(float(heading)), np.sin(float(heading))
        return np.asarray(
            [
                cos_h * float(delta[0]) + sin_h * float(delta[1]),
                -sin_h * float(delta[0]) + cos_h * float(delta[1]),
            ],
            dtype=np.float32,
        )
