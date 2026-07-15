"""State-machine driven platoon lock/unlock detection."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math


LOCKED = "LOCKED"
UNLOCKED = "UNLOCKED"
VALID_STATES = {LOCKED, UNLOCKED}


class RiskDetector(ABC):
    """Interface for platoon lock-state transition detection."""

    @abstractmethod
    def detect(
        self,
        env,
        agent_ids: list[str],
        traffic_vehicles: list,
        current_state: str,
        forced_lane_wait_info: dict | None = None,
    ) -> dict:
        """Return the state transition and the metrics used to decide it."""


class SimpleRuleRiskDetector(RiskDetector):
    """Switch formation state using leader TTC and follower spacing rules."""

    def __init__(
        self,
        *,
        ttc_trigger_s: float = 5.0,
        relock_ttc_threshold_s: float = 5.0,
        ideal_following_distance_m: float = 10.0,
        relock_gap_ratio: float = 1.5,
    ) -> None:
        self.ttc_trigger_s = float(ttc_trigger_s)
        self.relock_ttc_threshold_s = float(relock_ttc_threshold_s)
        self.ideal_following_distance_m = float(ideal_following_distance_m)
        self.relock_gap_ratio = float(relock_gap_ratio)

    def detect(
        self,
        env,
        agent_ids: list[str],
        traffic_vehicles: list,
        current_state: str,
        forced_lane_wait_info: dict | None = None,
    ) -> dict:
        state = str(current_state).upper()
        if state not in VALID_STATES:
            raise ValueError(f"Unsupported formation state: {current_state!r}")

        agents = getattr(env, "agents", {}) or {}
        if state == LOCKED:
            return self._detect_unlock(agents, agent_ids, traffic_vehicles, forced_lane_wait_info=forced_lane_wait_info)
        return self._detect_relock(agents, agent_ids, traffic_vehicles)

    def _detect_unlock(
        self,
        agents: dict,
        agent_ids: list[str],
        traffic_vehicles: list,
        *,
        forced_lane_wait_info: dict | None = None,
    ) -> dict:
        leader_metrics = self._leader_ttc_metrics(agents, agent_ids, traffic_vehicles)
        forced_lane_wait_info = dict(forced_lane_wait_info or {})
        forced_wait_steps = int(forced_lane_wait_info.get("partial_forced_wait_steps", 0) or 0)
        forced_wait_threshold = int(forced_lane_wait_info.get("threshold_steps", 0) or 0)
        forced_wait_transitioned = forced_wait_threshold > 0 and forced_wait_steps >= forced_wait_threshold
        if forced_wait_transitioned: # *强制换道切换
            return self._result(
                current_state=LOCKED,
                next_state=UNLOCKED,
                transition="LOCKED_TO_UNLOCKED",
                reason=f"partial_forced_lane_wait>={forced_wait_threshold}_steps",
                leader=leader_metrics,
                follower_pairs=[],
                min_follower_gap_m=None,
                forced_lane_wait_info=forced_lane_wait_info,
            )

        ttc = leader_metrics["ttc_s"]
        transitioned = ttc is not None and ttc < self.ttc_trigger_s

        if transitioned: # *风险切换
            print(f"ttc = {ttc}s")
            
        return self._result(
            current_state=LOCKED,
            next_state=UNLOCKED if transitioned else LOCKED,
            transition="LOCKED_TO_UNLOCKED" if transitioned else None,
            reason=f"leader_ttc<{self.ttc_trigger_s:.1f}s" if transitioned else None,
            leader=leader_metrics,
            follower_pairs=[],
            min_follower_gap_m=None,
            forced_lane_wait_info=forced_lane_wait_info,
        )

    def _detect_relock(self, agents: dict, agent_ids: list[str], traffic_vehicles: list) -> dict:
        pairs = []
        valid_gaps = []
        for index in range(1, len(agent_ids)):
            front_id, follower_id = agent_ids[index - 1], agent_ids[index]
            front, follower = agents.get(front_id), agents.get(follower_id)
            pair = {"front_agent_id": front_id, "follower_agent_id": follower_id, "net_gap_m": None}
            if front is not None and follower is not None and self._same_lane(front, follower):
                lane = getattr(front, "lane", None)
                front_s = self._longitudinal(lane, front)
                follower_s = self._longitudinal(lane, follower)
                if front_s is not None and follower_s is not None and follower_s < front_s:
                    gap = max(
                        front_s - follower_s - 0.5 * self._length(front) - 0.5 * self._length(follower),
                        0.0,
                    )
                    pair["net_gap_m"] = gap
                    valid_gaps.append(gap)
            pairs.append(pair)

        min_gap = min(valid_gaps) if valid_gaps else None
        relock_threshold = self.relock_gap_ratio * self.ideal_following_distance_m
        leader_metrics = self._leader_ttc_metrics(agents, agent_ids, traffic_vehicles)
        follower_gap_condition_met = min_gap is not None and min_gap < relock_threshold
        leader_ttc_condition_met = leader_metrics["ttc_s"] > self.relock_ttc_threshold_s
        transitioned = follower_gap_condition_met and leader_ttc_condition_met
        return self._result(
            current_state=UNLOCKED,
            next_state=LOCKED if transitioned else UNLOCKED,
            transition="UNLOCKED_TO_LOCKED" if transitioned else None,
            reason=(
                f"min_follower_gap<{relock_threshold:.1f}m_and_"
                f"leader_ttc>{self.relock_ttc_threshold_s:.1f}s"
                if transitioned
                else None
            ),
            leader=leader_metrics,
            follower_pairs=pairs,
            min_follower_gap_m=min_gap,
            relock_threshold_m=relock_threshold,
            relock_ttc_threshold_s=self.relock_ttc_threshold_s,
            follower_gap_condition_met=follower_gap_condition_met,
            leader_ttc_condition_met=leader_ttc_condition_met,
        )

    def _leader_ttc_metrics(self, agents: dict, agent_ids: list[str], traffic_vehicles: list) -> dict:
        leader = agents.get(agent_ids[0]) if agent_ids else None
        metrics = {
            "agent_id": agent_ids[0] if agent_ids else None,
            "front_vehicle_id": None,
            "front_net_gap_m": None,
            "closing_speed_mps": None,
            "ttc_s": math.inf,
        }
        if leader is None:
            return metrics
        front_vehicle, net_gap = self._nearest_front_vehicle(leader, traffic_vehicles)
        if front_vehicle is None:
            return metrics
        closing_speed = self._speed_mps(leader) - self._speed_mps(front_vehicle)
        metrics.update(
            {
                "front_vehicle_id": self._vehicle_id(front_vehicle),
                "front_net_gap_m": max(net_gap, 0.0),
                "closing_speed_mps": closing_speed,
                "ttc_s": max(net_gap, 0.0) / closing_speed if closing_speed > 0.0 else math.inf,
            }
        )
        return metrics

    @staticmethod
    def _result(*, current_state: str, next_state: str, transition, reason, **metrics) -> dict:
        transitioned = current_state != next_state
        return {
            "triggered": transitioned,
            "transitioned": transitioned,
            "current_state": current_state,
            "next_state": next_state,
            "transition": transition,
            "reason": reason,
            "reasons": [reason] if reason else [],
            **metrics,
        }

    @classmethod
    def _nearest_front_vehicle(cls, leader, traffic_vehicles: list):
        lane = getattr(leader, "lane", None)
        leader_s = cls._longitudinal(lane, leader)
        if lane is None or leader_s is None:
            return None, None
        nearest = None
        nearest_center_gap = math.inf
        for traffic_vehicle in traffic_vehicles:
            if traffic_vehicle is leader:
                continue
            if not cls._same_lane(leader, traffic_vehicle) and not cls._intrudes_lane_corridor(lane, traffic_vehicle):
                continue
            traffic_s = cls._longitudinal(lane, traffic_vehicle)
            if traffic_s is None or traffic_s <= leader_s:
                continue
            center_gap = traffic_s - leader_s
            if center_gap < nearest_center_gap:
                nearest = traffic_vehicle
                nearest_center_gap = center_gap
        if nearest is None:
            return None, None
        net_gap = nearest_center_gap - 0.5 * cls._length(leader) - 0.5 * cls._length(nearest)
        return nearest, net_gap

    @staticmethod
    def _same_lane(first, second) -> bool:
        first_lane = getattr(getattr(first, "lane", None), "index", None)
        second_lane = getattr(getattr(second, "lane", None), "index", None)
        return first_lane is not None and second_lane is not None and tuple(first_lane) == tuple(second_lane)

    @staticmethod
    def _longitudinal(lane, vehicle):
        if lane is None or not hasattr(lane, "local_coordinates"):
            return None
        return float(lane.local_coordinates(getattr(vehicle, "position", (0.0, 0.0)))[0])

    @classmethod
    def _intrudes_lane_corridor(cls, lane, vehicle) -> bool:
        if lane is None or not hasattr(lane, "local_coordinates"):
            return False
        try:
            _longitudinal, lateral = lane.local_coordinates(getattr(vehicle, "position", (0.0, 0.0)))
        except Exception:
            return False
        return abs(float(lateral)) <= 0.5 * cls._lane_width(lane) + 0.5 * cls._vehicle_width(vehicle)

    @staticmethod
    def _lane_width(lane) -> float:
        return float(getattr(lane, "width", 3.5) or 3.5)

    @staticmethod
    def _vehicle_width(vehicle) -> float:
        return float(getattr(vehicle, "WIDTH", 2.0) or 2.0)

    @staticmethod
    def _speed_mps(vehicle) -> float:
        return max(0.0, float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6)

    @staticmethod
    def _length(vehicle) -> float:
        return float(getattr(vehicle, "LENGTH", 4.5) or 4.5)

    @staticmethod
    def _vehicle_id(vehicle):
        return getattr(vehicle, "id", getattr(vehicle, "name", None))
