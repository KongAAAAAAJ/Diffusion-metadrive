from __future__ import annotations

import math

import numpy as np

from metadrive.exp_dataset.expert_idm_policy import ExpertIDMPolicy
from metadrive.exp_dataset.hierarchical_expert.driving_style import DrivingStyleProfile
from metadrive.exp_dataset.hierarchical_expert.intersection_regulator import IntersectionConflictRegulator
from metadrive.exp_dataset.hierarchical_expert.lane_change_manager import LaneChangeManager, ManeuverState
from metadrive.exp_dataset.hierarchical_expert.rear_end_guard import RearEndGuardRegulator
from metadrive.exp_dataset.hierarchical_expert.roundabout_regulator import RoundaboutRegulator
from metadrive.exp_dataset.hierarchical_expert.safety_assessor import SafetyAssessor
from metadrive.exp_dataset.hierarchical_expert.trajectory_planner import QuinticLaneChangePlanner
from metadrive.exp_dataset.hierarchical_expert.trajectory_tracker import PurePursuitTracker


class HierarchicalExpertIDMPolicy(ExpertIDMPolicy):
    def __init__(
        self,
        control_object,
        random_seed=0,
        style_profile=None,
    ):
        super().__init__(control_object, random_seed)
        self.style = style_profile or DrivingStyleProfile()
        self.DISTANCE_WANTED = self.style.min_jam_distance
        self.TIME_WANTED = self.style.time_headway
        self.DELTA = self.style.velocity_exponent
        self.ACC_FACTOR = self.style.max_accel
        self.DEACC_FACTOR = -self.style.comfortable_decel
        self.target_speed = self.NORMAL_SPEED * self.style.desired_speed_ratio
        self.safety = SafetyAssessor(self.style)
        self.planner = QuinticLaneChangePlanner(self.style)
        self.manager = LaneChangeManager(self.safety, self.planner, self.style)
        self.trajectory_tracker = PurePursuitTracker()
        self.rear_end_guard = RearEndGuardRegulator()
        self.intersection_regulator = IntersectionConflictRegulator()
        self.roundabout_regulator = RoundaboutRegulator()
        self._last_maneuver_state = self.manager.state

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float(math.atan2(math.sin(angle), math.cos(angle)))

    def _compute_control_action(self, steering_target):
        if self.manager.state in (ManeuverState.EXECUTING, ManeuverState.ABORTING):
            steering = self.trajectory_tracker.compute_steering(
                ego_position=self.control_object.position,
                ego_heading=self.control_object.heading_theta,
                ego_speed=self.control_object.speed,
                trajectory=steering_target,
            )
            return float(steering)

        transitioning = self._last_maneuver_state in (ManeuverState.EXECUTING, ManeuverState.ABORTING)
        if transitioning:
            self.heading_pid.reset()
            self.lateral_pid.reset()
        steering = self.steering_control(steering_target)
        last_pp = getattr(self.trajectory_tracker, "_last_steering", None)
        max_step = getattr(self.trajectory_tracker, "MAX_STEER_DELTA_PER_STEP", None)
        if transitioning and isinstance(last_pp, (int, float)) and isinstance(max_step, (int, float)):
            max_delta = float(max_step) * 8
            steering = float(np.clip(steering, last_pp - max_delta, last_pp + max_delta))
        self.trajectory_tracker.reset()
        return float(steering)

    def act(self, *args, **kwargs):
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        current_lanes = self.control_object.navigation.current_ref_lanes
        next_lanes = self.control_object.navigation.next_ref_lanes
        try:
            if success:
                steering_target, acc_front_obj, acc_front_dist = self.manager.update(
                    ego=self.control_object,
                    all_objects=all_objects,
                    routing_target_lane=self.routing_target_lane,
                    current_lanes=current_lanes,
                    next_lanes=next_lanes,
                )
            else:
                steering_target = self.routing_target_lane
                acc_front_obj = None
                acc_front_dist = 5
        except Exception:
            steering_target = self.routing_target_lane
            acc_front_obj = None
            acc_front_dist = 5

        steering = self._compute_control_action(steering_target)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        idm_acc = acc
        acc = self.rear_end_guard.adjust_acceleration(
            ego=self.control_object,
            front_obj=acc_front_obj,
            front_dist=acc_front_dist,
            idm_acc=acc,
        )
        rear_end_diagnostics = getattr(self.rear_end_guard, "last_diagnostics", {})
        acc = self.roundabout_regulator.adjust_acceleration(
            ego=self.control_object,
            all_objects=all_objects,
            reference_target=steering_target,
            idm_acc=acc,
        )
        roundabout_diagnostics = getattr(self.roundabout_regulator, "last_diagnostics", {})
        acc = self.intersection_regulator.adjust_acceleration(
            ego=self.control_object,
            all_objects=all_objects,
            reference_target=steering_target,
            idm_acc=acc,
            maneuver_state=self.manager.state,
        )
        action = [steering, acc]
        diagnostics = getattr(self.intersection_regulator, "last_diagnostics", {})
        conflict_active = diagnostics.get("active", acc < idm_acc - 1e-6)
        self.action_info["action"] = action
        self.action_info["maneuver_state"] = self.manager.state.name
        self.action_info["style_aggression"] = self.style.aggression
        self.action_info["rear_end_guard_active"] = rear_end_diagnostics.get("active", acc < idm_acc - 1e-6)
        self.action_info["rear_end_guard_acc"] = rear_end_diagnostics.get("acc", idm_acc)
        self.action_info["rear_end_guard_ttc"] = rear_end_diagnostics.get("ttc")
        self.action_info["rear_end_guard_gap"] = rear_end_diagnostics.get("gap", acc_front_dist)
        self.action_info["roundabout_phase"] = roundabout_diagnostics.get("phase", "NONE")
        self.action_info["roundabout_active"] = roundabout_diagnostics.get("active", False)
        self.action_info["roundabout_acc"] = roundabout_diagnostics.get("acc", idm_acc)
        self.action_info["intersection_conflict_active"] = conflict_active
        self.action_info["intersection_conflict_count"] = diagnostics.get(
            "conflict_count",
            1 if conflict_active else 0,
        )
        self.action_info["intersection_conflict_acc"] = acc
        self._last_maneuver_state = self.manager.state
        return action

    def reset(self):
        super().reset()
        self.manager.state = ManeuverState.IDLE
        self.manager.active_trajectory = None
        self.manager.active_command = None
        self.manager.source_lane = None
        self.manager.cooldown_timer = 0
        self._last_maneuver_state = ManeuverState.IDLE
        self.trajectory_tracker.reset()
        self.rear_end_guard.reset()
        self.roundabout_regulator.reset()
