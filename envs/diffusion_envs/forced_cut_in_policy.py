"""Deterministic external-vehicle cut-in policy for bundle-v2 scenarios."""

from __future__ import annotations

from math import inf

from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy

from envs.diffusion_envs.ground_truth_idm_policy import GroundTruthIDMMixin


class ForcedCutInPolicy(GroundTruthIDMMixin, IDMPolicy):
    """Hold the source lane, then request one adjacent-lane transition.

    The activation step is measured in simulator decision boundaries.  Before
    activation this policy explicitly suppresses IDM lane changes, so the
    recorded entry onset cannot be caused by an incidental policy decision.
    """

    def __init__(
        self,
        control_object,
        random_seed,
        *,
        activation_step: int = 80,
        target_lane_offset: int = 1,
        front_gap_m: float = 5.0,
        rear_gap_m: float = 5.0,
        cruise_speed_kmh: float = 24.0,
    ):
        super().__init__(control_object, random_seed)
        self.activation_step = int(activation_step)
        self.target_lane_offset = int(target_lane_offset)
        self.front_gap_m = float(front_gap_m)
        self.rear_gap_m = float(rear_gap_m)
        self.cruise_speed_kmh = float(cruise_speed_kmh)
        if self.activation_step < 0:
            raise ValueError("activation_step must be non-negative")
        if self.target_lane_offset not in {-1, 1}:
            raise ValueError("target_lane_offset must be -1 or +1")
        if self.front_gap_m < 0.0 or self.rear_gap_m < 0.0:
            raise ValueError("cut-in gaps must be non-negative")
        self.NORMAL_SPEED = self.cruise_speed_kmh
        self.target_speed = self.cruise_speed_kmh
        self.cut_in_policy_step = 0
        self.cut_in_completed = False

    def reset(self):
        super().reset()
        self.cut_in_policy_step = 0
        self.cut_in_completed = False

    def lane_change_policy(self, all_objects):
        step = self.cut_in_policy_step
        self.cut_in_policy_step += 1
        front_object, front_distance, _ = super().lane_change_policy(all_objects)
        current_lane = getattr(self.control_object, "lane", None)
        target_lane = self._target_lane()
        self._update_completed(target_lane)
        active = step >= self.activation_step and not self.cut_in_completed
        self._record_state(active=active, completed=self.cut_in_completed)
        if not active or target_lane is None:
            return front_object, front_distance, current_lane

        surrounding = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            target_lane,
            self.control_object.position,
            max_distance=max(float(self.MAX_LONG_DIST), self.front_gap_m, self.rear_gap_m),
        )
        front_gap = (
            float(surrounding.front_min_distance())
            if surrounding.has_front_object()
            else inf
        )
        rear_gap = (
            float(surrounding.back_min_distance())
            if surrounding.has_back_object()
            else inf
        )
        accepted = front_gap >= self.front_gap_m and rear_gap >= self.rear_gap_m
        self._record_state(
            active=True,
            accepted=accepted,
            front_gap=front_gap,
            rear_gap=rear_gap,
            completed=False,
        )
        if accepted:
            return surrounding.front_object(), surrounding.front_min_distance(), target_lane
        return front_object, front_distance, current_lane

    def _target_lane(self):
        lane = getattr(self.control_object, "lane", None)
        lane_index = getattr(lane, "index", None)
        current_map = getattr(getattr(self.control_object, "engine", None), "current_map", None)
        graph = getattr(getattr(current_map, "road_network", None), "graph", {})
        if lane_index is None:
            return None
        start, end, lane_id = tuple(lane_index)
        lanes = graph.get(start, {}).get(end)
        target_id = int(lane_id) + self.target_lane_offset
        if not lanes or target_id < 0 or target_id >= len(lanes):
            return None
        return lanes[target_id]

    def _update_completed(self, target_lane) -> None:
        if self.cut_in_completed or target_lane is None:
            return
        current_index = getattr(getattr(self.control_object, "lane", None), "index", None)
        target_index = getattr(target_lane, "index", None)
        if current_index is not None and target_index is not None and tuple(current_index) == tuple(target_index):
            self.cut_in_completed = True

    def _record_state(
        self,
        *,
        active: bool,
        accepted: bool = False,
        front_gap: float = inf,
        rear_gap: float = inf,
        completed: bool,
    ) -> None:
        self.action_info["cut_in_active"] = bool(active)
        self.action_info["cut_in_gap_accepted"] = bool(accepted)
        self.action_info["cut_in_front_gap"] = float(front_gap)
        self.action_info["cut_in_rear_gap"] = float(rear_gap)
        self.action_info["cut_in_completed"] = bool(completed)


__all__ = ["ForcedCutInPolicy"]
