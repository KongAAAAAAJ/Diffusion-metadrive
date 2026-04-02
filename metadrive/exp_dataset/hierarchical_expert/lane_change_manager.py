from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

from metadrive.policy.idm_policy import FrontBackObjects

from metadrive.exp_dataset.hierarchical_expert.driving_style import DrivingStyleProfile
from metadrive.exp_dataset.hierarchical_expert.safety_assessor import SafetyAssessor
from metadrive.exp_dataset.hierarchical_expert.trajectory_planner import QuinticLaneChangePlanner


class ManeuverState(IntEnum):
    IDLE = 0
    EXECUTING = 1
    ABORTING = 2


@dataclass
class ManeuverCommand:
    direction: int
    target_lane: object
    source_lane: object
    urgency: float
    is_mandatory: bool


class LaneChangeManager:
    def __init__(self, safety: SafetyAssessor, planner: QuinticLaneChangePlanner, style: DrivingStyleProfile):
        self.safety = safety
        self.planner = planner
        self.style = style
        self.state = ManeuverState.IDLE
        self.active_trajectory = None
        self.active_command: Optional[ManeuverCommand] = None
        self.source_lane = None
        self.cooldown_timer = 0

    def update(self, ego, all_objects, routing_target_lane, current_lanes, next_lanes):
        # Use ego.lane for perception so left/right slots reflect the vehicle's
        # actual physical lane, not the (possibly stale) routing target lane.
        physical_lane = ego.lane if (ego.lane is not None and ego.lane in current_lanes) else (routing_target_lane or ego.lane)
        perception = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            physical_lane,
            ego.position,
            max_distance=999.0,
            ref_lanes=current_lanes,
        )
        acc_front_obj = perception.front_object()
        acc_front_dist = perception.front_min_distance()

        if self.state == ManeuverState.EXECUTING:
            return self._handle_executing(ego, perception, acc_front_obj, acc_front_dist, current_lanes)
        if self.state == ManeuverState.ABORTING:
            return self._handle_aborting(ego, acc_front_obj, acc_front_dist)
        return self._handle_idle(ego, perception, routing_target_lane, current_lanes, next_lanes, physical_lane, acc_front_obj, acc_front_dist)

    # Minimum speed [m/s] to attempt a lane change. Below this threshold the quintic
    # planner produces a near-zero-longitudinal-displacement trajectory which causes
    # the Pure Pursuit tracker to steer at maximum angle while barely moving forward.
    MIN_LANE_CHANGE_SPEED: float = 5.0

    def _handle_idle(self, ego, perception, routing_target_lane, current_lanes, next_lanes, physical_lane, acc_front_obj, acc_front_dist):
        # Follow the vehicle's actual physical lane for steering, not the routing target.
        # After a voluntary lane change, routing_target_lane is stale (points to old lane),
        # which would cause the PID to steer back — using physical_lane avoids this.
        fallback = physical_lane
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return fallback, acc_front_obj, acc_front_dist

        if float(ego.speed) < self.MIN_LANE_CHANGE_SPEED:
            return fallback, acc_front_obj, acc_front_dist

        urgency, is_mandatory, forced_direction = self._compute_route_urgency(
            routing_target_lane or ego.lane, current_lanes, next_lanes, ego
        )
        candidates = []
        for direction in (-1, 1):
            # Use physical_lane (ego's actual lane) so adjacent targets are computed
            # relative to where the vehicle really is, not the routing target lane.
            target_lane = self._get_adjacent_lane(physical_lane, current_lanes, direction)
            if target_lane is None:
                continue
            if direction < 0 and hasattr(perception, "left_lane_exist") and not perception.left_lane_exist():
                continue
            if direction > 0 and hasattr(perception, "right_lane_exist") and not perception.right_lane_exist():
                continue
            utility = self._compute_lane_change_utility(direction, ego, perception, urgency if direction == forced_direction else 0.0, is_mandatory and direction == forced_direction)
            candidates.append((utility, direction, target_lane))

        if not candidates:
            return fallback, acc_front_obj, acc_front_dist

        utility, direction, target_lane = max(candidates, key=lambda item: item[0])
        candidate_mandatory = is_mandatory and direction == forced_direction
        # For mandatory lane changes, check gap against the routing direction; otherwise use physical direction
        allowed = self.safety.is_gap_acceptable(direction, ego.speed, perception)
        should_execute = allowed and (candidate_mandatory or utility >= self.style.lane_change_threshold)
        if not should_execute:
            return fallback, acc_front_obj, acc_front_dist

        trajectory = self.planner.plan(
            ego_position=ego.position,
            ego_heading=ego.heading_theta,
            ego_speed=ego.speed,
            source_lane=physical_lane,  # plan from actual physical lane
            target_lane=target_lane,
            direction=direction,
            urgency=urgency if candidate_mandatory else 0.0,
        )
        if trajectory is None:
            return fallback, acc_front_obj, acc_front_dist

        self.state = ManeuverState.EXECUTING
        self.active_trajectory = trajectory
        self.source_lane = physical_lane
        self.active_command = ManeuverCommand(
            direction=direction,
            target_lane=target_lane,
            source_lane=self.source_lane,
            urgency=urgency if candidate_mandatory else 0.0,
            is_mandatory=candidate_mandatory,
        )
        if direction < 0:
            acc_front_obj = perception.left_front_object() if perception.has_left_front_object() else None
            acc_front_dist = perception.left_front_min_distance()
        else:
            acc_front_obj = perception.right_front_object() if perception.has_right_front_object() else None
            acc_front_dist = perception.right_front_min_distance()
        return self.active_trajectory, acc_front_obj, acc_front_dist

    def _handle_executing(self, ego, perception, acc_front_obj, acc_front_dist, current_lanes=None):
        assert self.active_command is not None

        # Hard abort: crossing a solid line means the trajectory is unsafe regardless of gap.
        if getattr(ego, 'on_yellow_continuous_line', False) or getattr(ego, 'on_white_continuous_line', False):
            self.active_trajectory = self.planner.plan_abort(
                ego_position=ego.position,
                ego_heading=ego.heading_theta,
                ego_speed=ego.speed,
                source_lane=self.source_lane,
            )
            self.state = ManeuverState.ABORTING
            return self.active_trajectory, acc_front_obj, acc_front_dist

        # Abort if target lane is no longer in the current road segment (stale reference).
        if current_lanes is not None and self.active_command.target_lane not in current_lanes:
            if self.source_lane is not None and self.source_lane in current_lanes:
                self.active_trajectory = self.planner.plan_abort(
                    ego_position=ego.position,
                    ego_heading=ego.heading_theta,
                    ego_speed=ego.speed,
                    source_lane=self.source_lane,
                )
                self.state = ManeuverState.ABORTING
            else:
                # Both source and target lanes are stale; hand off to PID on physical lane.
                physical_lane = ego.lane if ego.lane is not None else current_lanes[0]
                self.state = ManeuverState.IDLE
                self.active_trajectory = None
                self.active_command = None
                self.source_lane = None
                self.cooldown_timer = int(self.style.lane_change_cooldown)
                return physical_lane, acc_front_obj, acc_front_dist
            return self.active_trajectory, acc_front_obj, acc_front_dist

        if not self.safety.monitor_ongoing(self.active_command.direction, ego.speed, perception):
            self.active_trajectory = self.planner.plan_abort(
                ego_position=ego.position,
                ego_heading=ego.heading_theta,
                ego_speed=ego.speed,
                source_lane=self.source_lane,
            )
            self.state = ManeuverState.ABORTING
            return self.active_trajectory, acc_front_obj, acc_front_dist

        _, lateral_error = self.active_command.target_lane.local_coordinates(ego.position)
        if abs(lateral_error) < 0.3:
            target_lane = self.active_command.target_lane
            self.state = ManeuverState.IDLE
            self.active_trajectory = None
            self.source_lane = None
            self.cooldown_timer = int(self.style.lane_change_cooldown)
            return target_lane, acc_front_obj, acc_front_dist
        return self.active_trajectory, acc_front_obj, acc_front_dist

    def _handle_aborting(self, ego, acc_front_obj, acc_front_dist):
        _, lateral_error = self.source_lane.local_coordinates(ego.position)
        if abs(lateral_error) < 0.5:
            source_lane = self.source_lane
            self.state = ManeuverState.IDLE
            self.active_trajectory = None
            self.active_command = None
            self.source_lane = None
            self.cooldown_timer = int(self.style.lane_change_cooldown)
            return source_lane, acc_front_obj, acc_front_dist
        return self.active_trajectory, acc_front_obj, acc_front_dist

    def _compute_route_urgency(self, routing_target_lane, current_lanes, next_lanes, ego) -> Tuple[float, bool, int]:
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes else 0
        if lane_num_diff <= 0:
            return 0.0, False, 0

        current_index = routing_target_lane.index[-1]
        if current_lanes[0].is_previous_lane_of(next_lanes[0]):
            legal_indices = list(range(len(next_lanes)))
        else:
            legal_indices = list(range(lane_num_diff, len(current_lanes)))

        if current_index in legal_indices:
            return 0.0, False, 0

        ego_longitudinal, _ = routing_target_lane.local_coordinates(ego.position)
        remaining = float(routing_target_lane.length - ego_longitudinal)
        urgency = float(min(max(1.0 - remaining / 200.0, 0.0), 1.0))
        direction = -1 if current_index > legal_indices[-1] else 1
        return urgency, True, direction

    def _compute_lane_change_utility(self, direction, ego, perception, urgency, is_mandatory) -> float:
        if is_mandatory:
            return 1.0 if self.safety.is_gap_acceptable(direction, ego.speed, perception) else -1.0

        current_front = perception.front_object() if perception.has_front_object() else None
        current_distance = perception.front_min_distance()
        current_acc = ego.policy.acceleration(current_front, current_distance) if hasattr(ego, "policy") else 0.0

        if direction < 0:
            target_front = perception.left_front_object() if perception.has_left_front_object() else None
            target_distance = perception.left_front_min_distance()
        else:
            target_front = perception.right_front_object() if perception.has_right_front_object() else None
            target_distance = perception.right_front_min_distance()
        target_acc = ego.policy.acceleration(target_front, target_distance) if hasattr(ego, "policy") else 0.0

        u_mobil = float(target_acc - current_acc)
        u_route = float(urgency * self.style.route_urgency_weight)
        u_safety = float(self.safety.gap_acceptance_score(direction, ego.speed, perception))
        u_style = 0.0
        return 0.3 * u_mobil + 0.4 * u_route + 0.2 * u_safety + 0.1 * u_style

    @staticmethod
    def _get_adjacent_lane(routing_target_lane, current_lanes, direction):
        if routing_target_lane not in current_lanes:
            return None
        current_idx = current_lanes.index(routing_target_lane)
        target_idx = current_idx - 1 if direction < 0 else current_idx + 1
        if target_idx < 0 or target_idx >= len(current_lanes):
            return None
        return current_lanes[target_idx]
