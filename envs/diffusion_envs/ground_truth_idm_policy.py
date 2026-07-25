"""IDM traffic policy using simulator object state instead of a range sensor."""

from __future__ import annotations

import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy


class GroundTruthIDMMixin:
    """Source nearby vehicles from engine state for an IDM-derived policy."""

    SURROUNDING_RADIUS_M = 50.0

    def _ground_truth_surrounding_vehicles(self) -> tuple[BaseVehicle, ...]:
        position = np.asarray(self.control_object.position[:2], dtype=np.float64)
        radius_squared = self.SURROUNDING_RADIUS_M**2
        objects = self.engine.get_objects(
            lambda obj: isinstance(obj, BaseVehicle) and obj is not self.control_object
        )
        nearby = []
        for vehicle in objects.values():
            other_position = np.asarray(vehicle.position[:2], dtype=np.float64)
            if float(np.sum((other_position - position) ** 2)) <= radius_squared:
                nearby.append(vehicle)
        return tuple(nearby)

    def act(self, *args, **kwargs):
        if not bool(
            self.engine.global_config.get("ground_truth_traffic_policy", False)
        ):
            return super().act(*args, **kwargs)
        success = self.move_to_next_road()
        all_objects = self._ground_truth_surrounding_vehicles()
        if success and self.enable_lane_change:
            acc_front_obj, acc_front_dist, steering_target_lane = (
                self.lane_change_policy(all_objects)
            )
        else:
            surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                all_objects,
                self.routing_target_lane,
                self.control_object.position,
                max_distance=self.MAX_LONG_DIST,
            )
            acc_front_obj = surrounding_objects.front_object()
            acc_front_dist = surrounding_objects.front_min_distance()
            steering_target_lane = self.routing_target_lane

        steering = self.steering_control(steering_target_lane)
        acceleration = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acceleration]
        self.action_info["action"] = action
        return action


class GroundTruthIDMPolicy(GroundTruthIDMMixin, IDMPolicy):
    """Standard IDM behavior without a range-sensor dependency."""


__all__ = ["GroundTruthIDMMixin", "GroundTruthIDMPolicy"]
