from __future__ import annotations

import math
from enum import IntEnum

import numpy as np


class RoundaboutPhase(IntEnum):
    NONE = 0
    APPROACHING = 1
    IN_ROUNDABOUT = 2
    EXITING = 3


class RoundaboutRegulator:
    def __init__(
        self,
        horizon: float = 4.0,
        dt: float = 0.2,
        candidate_radius: float = 35.0,
        min_entry_gap_time: float = 3.0,
        approach_yield_acc: float = -1.5,
        hard_yield_acc: float = -3.0,
        conflict_distance: float = 5.0,
        safe_time_gap: float = 1.5,
        max_braking: float = -4.5,
        accel_rate_limit: float = 1.0,
        roundabout_radius_threshold: float = 30.0,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.candidate_radius = float(candidate_radius)
        self.min_entry_gap_time = float(min_entry_gap_time)
        self.approach_yield_acc = float(approach_yield_acc)
        self.hard_yield_acc = float(hard_yield_acc)
        self.conflict_distance = float(conflict_distance)
        self.safe_time_gap = float(safe_time_gap)
        self.max_braking = float(max_braking)
        self.accel_rate_limit = float(accel_rate_limit)
        self.roundabout_radius_threshold = float(roundabout_radius_threshold)
        self._speed_epsilon = 1e-3
        self._committed_time_window = 0.8
        self._committed_speed_threshold = 3.0
        self._emergency_distance = 2.5
        self._last_accel = None
        self._last_phase = RoundaboutPhase.NONE
        self.last_diagnostics = {"phase": "NONE", "active": False, "acc": 0.0}

    def adjust_acceleration(self, ego, all_objects, reference_target, idm_acc) -> float:
        try:
            phase = self._detect_phase(ego, reference_target)
            if phase == RoundaboutPhase.NONE:
                return self._finalize(idm_acc, idm_acc, phase)
            if phase == RoundaboutPhase.APPROACHING:
                target_acc = self._gap_acceptance_acc(ego, all_objects, reference_target, idm_acc)
            elif phase == RoundaboutPhase.IN_ROUNDABOUT:
                target_acc = float(idm_acc)
            elif phase == RoundaboutPhase.EXITING:
                target_acc = float(idm_acc)
            else:
                target_acc = float(idm_acc)
            return self._finalize(idm_acc, target_acc, phase)
        except Exception:
            return float(idm_acc)

    def _detect_phase(self, ego, reference_target) -> RoundaboutPhase:
        navigation = getattr(ego, "navigation", None)
        current_ref_lanes = getattr(navigation, "current_ref_lanes", None) or []
        next_ref_lanes = getattr(navigation, "next_ref_lanes", None) or []
        in_roundabout_now = any(self._is_roundabout_lane(lane) for lane in current_ref_lanes)
        next_is_roundabout = any(self._is_roundabout_lane(lane) for lane in next_ref_lanes)

        if in_roundabout_now and not next_is_roundabout:
            return RoundaboutPhase.EXITING
        if in_roundabout_now:
            return RoundaboutPhase.IN_ROUNDABOUT
        if next_is_roundabout:
            try:
                current_lane = reference_target if reference_target is not None else getattr(ego, "lane", None)
                if current_lane is None or not hasattr(current_lane, "local_coordinates"):
                    return RoundaboutPhase.NONE
                long, _ = current_lane.local_coordinates(np.asarray(ego.position, dtype=np.float64))
                distance_to_entry = float(getattr(current_lane, "length", long) - long)
                if distance_to_entry <= 30.0:
                    return RoundaboutPhase.APPROACHING
            except Exception:
                return RoundaboutPhase.NONE
        return RoundaboutPhase.NONE

    def _is_roundabout_lane(self, lane) -> bool:
        if lane is None:
            return False
        lane_type_name = type(lane).__name__.lower()
        if "circular" in lane_type_name or "round" in lane_type_name:
            return True
        radius = getattr(lane, "radius", None)
        if radius is not None:
            try:
                radius = float(radius)
                if 0.0 < radius <= self.roundabout_radius_threshold:
                    return True
            except Exception:
                pass
        road_block_name = type(getattr(lane, "road_block", None)).__name__.lower()
        if "round" in road_block_name or "circle" in road_block_name:
            return True
        curvature = getattr(lane, "curvature", None)
        if curvature is not None:
            try:
                if float(curvature) > 1.0 / self.roundabout_radius_threshold:
                    return True
            except Exception:
                pass
        return False

    def _gap_acceptance_acc(self, ego, all_objects, reference_target, idm_acc) -> float:
        ego_path = self._predict_path_along_lane(ego, reference_target, self._object_speed(ego))
        ego_position = np.asarray(getattr(ego, "position", np.zeros(2)), dtype=np.float64)
        gap_times = []
        for obj in all_objects:
            if obj is ego:
                continue
            if self._object_speed(obj) <= 0.5:
                continue
            obj_position = np.asarray(getattr(obj, "position", np.zeros(2)), dtype=np.float64)
            if np.linalg.norm(obj_position - ego_position) > self.candidate_radius:
                continue
            if not self._is_roundabout_lane(getattr(obj, "lane", None)):
                continue
            ego_dir = self._object_direction(ego)
            obj_dir = self._object_direction(obj)
            cos_angle = float(np.dot(ego_dir, obj_dir))
            if cos_angle > 0.7:
                continue

            obj_path = self._predict_path_along_lane(obj, self._object_reference_lane(obj), self._object_speed(obj))
            dists = np.linalg.norm(ego_path - obj_path, axis=1)
            conflict_idx = np.where(dists < self.conflict_distance)[0]
            if conflict_idx.size == 0:
                continue
            first_idx = int(conflict_idx[0])
            t_ego = float(first_idx * self.dt)
            conflict_pos = ego_path[first_idx]
            obj_dists_to_conflict = np.linalg.norm(obj_path - conflict_pos, axis=1)
            t_obj_arrival = float(int(np.argmin(obj_dists_to_conflict)) * self.dt)
            gap_times.append(t_obj_arrival - t_ego)

        if not gap_times:
            return float(idm_acc)

        min_gap_time = float(min(gap_times))
        if min_gap_time <= 0.0:
            return float(idm_acc)
        if min_gap_time < self.min_entry_gap_time * 0.4:
            return float(max(self.hard_yield_acc, self.max_braking))
        if min_gap_time < self.min_entry_gap_time:
            return float(max(min(float(idm_acc), self.approach_yield_acc), self.max_braking))
        return float(idm_acc)

    def _circulating_conflict_acc(self, ego, all_objects, reference_target, idm_acc) -> float:
        ego_path = self._predict_path_along_lane(ego, reference_target, self._object_speed(ego))
        ego_position = np.asarray(getattr(ego, "position", np.zeros(2)), dtype=np.float64)
        conflicts = []
        for obj in all_objects:
            if obj is ego:
                continue
            obj_speed = self._object_speed(obj)
            if obj_speed <= 0.1:
                continue
            obj_position = np.asarray(getattr(obj, "position", np.zeros(2)), dtype=np.float64)
            if np.linalg.norm(obj_position - ego_position) > self.candidate_radius:
                continue
            obj_lane = self._object_reference_lane(obj)
            if not self._is_roundabout_lane(obj_lane) and not self._is_roundabout_lane(getattr(obj, "lane", None)):
                continue
            obj_path = self._predict_path_along_lane(obj, obj_lane, obj_speed)
            dists = np.linalg.norm(ego_path - obj_path, axis=1)
            if float(np.min(dists)) >= self.conflict_distance:
                continue
            conflict_idx = int(np.where(dists < self.conflict_distance)[0][0])
            conflict_time = float(conflict_idx * self.dt)
            conflicts.append(
                {
                    "time_gap": conflict_time,
                    "min_distance": float(np.min(dists)),
                    "s_ego": max(self._object_speed(ego), self._speed_epsilon) * conflict_time,
                    "s_obj": obj_speed * conflict_time,
                    "obj_speed": obj_speed,
                }
            )

        if not conflicts:
            return float(idm_acc)

        most_dangerous = min(conflicts, key=lambda item: (item["time_gap"], item["min_distance"]))
        if self._should_hold_course(ego, most_dangerous):
            return float(idm_acc)
        return self._compute_required_acceleration(
            s_ego=most_dangerous["s_ego"],
            v_ego=max(self._object_speed(ego), 0.0),
            s_obj=most_dangerous["s_obj"],
            v_obj=most_dangerous["obj_speed"],
            idm_acc=idm_acc,
        )

    def _predict_path_along_lane(self, obj, reference_lane, speed) -> np.ndarray:
        speed = max(float(speed), 0.5)
        if reference_lane is not None and hasattr(reference_lane, "local_coordinates") and hasattr(reference_lane, "position"):
            try:
                long, _ = reference_lane.local_coordinates(np.asarray(obj.position, dtype=np.float64))
                long = float(max(long, 0.0))
                lane_length = float(getattr(reference_lane, "length", long + self.horizon * max(speed, 1.0)))
                samples = []
                end_pos = None
                end_dir = None
                for t in self._time_samples():
                    sample_long = long + speed * t
                    if sample_long <= lane_length:
                        samples.append(np.asarray(reference_lane.position(float(sample_long), 0.0), dtype=np.float64)[:2])
                        continue
                    if end_pos is None:
                        end_pos = np.asarray(reference_lane.position(lane_length, 0.0), dtype=np.float64)[:2]
                        near_end_long = max(lane_length - 0.5, 0.0)
                        near_end_pos = np.asarray(reference_lane.position(near_end_long, 0.0), dtype=np.float64)[:2]
                        end_dir = end_pos - near_end_pos
                        norm = np.linalg.norm(end_dir)
                        if norm > self._speed_epsilon:
                            end_dir = end_dir / norm
                        else:
                            end_dir = self._object_direction(obj)
                    overshoot = float(sample_long - lane_length)
                    samples.append(end_pos + end_dir * overshoot)
                return np.stack(samples, axis=0)
            except Exception:
                pass
        origin = np.asarray(obj.position, dtype=np.float64)[:2]
        direction = self._object_direction(obj)
        return np.stack([origin + direction * speed * t for t in self._time_samples()], axis=0)

    def _object_reference_lane(self, obj):
        navigation = getattr(obj, "navigation", None)
        current_ref_lanes = getattr(navigation, "current_ref_lanes", None) or []
        if current_ref_lanes:
            lane_index = getattr(obj, "lane_index", None)
            if isinstance(lane_index, (tuple, list)) and lane_index:
                candidate_idx = lane_index[-1]
                if isinstance(candidate_idx, (int, np.integer)) and 0 <= int(candidate_idx) < len(current_ref_lanes):
                    return current_ref_lanes[int(candidate_idx)]
            current_lane = getattr(obj, "lane", None)
            if current_lane in current_ref_lanes:
                return current_lane
            return current_ref_lanes[0]
        return getattr(obj, "lane", None)

    def _object_speed(self, obj) -> float:
        if hasattr(obj, "speed"):
            return max(float(obj.speed), 0.0)
        if hasattr(obj, "speed_km_h"):
            return max(float(obj.speed_km_h) / 3.6, 0.0)
        velocity = getattr(obj, "velocity_km_h", None)
        if velocity is not None:
            return max(float(np.linalg.norm(np.asarray(velocity, dtype=np.float64))) / 3.6, 0.0)
        return 0.0

    def _object_direction(self, obj) -> np.ndarray:
        heading = getattr(obj, "heading", None)
        if heading is not None:
            direction = np.asarray(heading, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm > self._speed_epsilon:
                return direction / norm
        velocity = getattr(obj, "velocity_km_h", None)
        if velocity is not None:
            direction = np.asarray(velocity, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm > self._speed_epsilon:
                return direction / norm
        heading_theta = getattr(obj, "heading_theta", None)
        if heading_theta is not None:
            return np.asarray([math.cos(float(heading_theta)), math.sin(float(heading_theta))], dtype=np.float64)
        return np.asarray([1.0, 0.0], dtype=np.float64)

    def _time_samples(self) -> np.ndarray:
        return np.arange(0.0, self.horizon + self.dt * 0.5, self.dt, dtype=np.float64)

    def _limit_accel_change(self, accel_cmd) -> float:
        accel_cmd = float(accel_cmd)
        if self._last_accel is None:
            self._last_accel = accel_cmd
            return accel_cmd
        limited = float(
            np.clip(
                accel_cmd,
                self._last_accel - self.accel_rate_limit,
                self._last_accel + self.accel_rate_limit,
            )
        )
        self._last_accel = limited
        return limited

    def _compute_required_acceleration(self, s_ego, v_ego, s_obj, v_obj, idm_acc) -> float:
        eps = self._speed_epsilon
        t_obj = float(s_obj / max(v_obj, eps))
        t_target = max(t_obj + self.safe_time_gap, self.dt)
        a_conf = 2.0 * (float(s_ego) - float(v_ego) * t_target) / (t_target ** 2)
        a_conf = min(float(a_conf), float(idm_acc))
        return float(np.clip(a_conf, self.max_braking, np.inf))

    def _should_hold_course(self, ego, conflict: dict) -> bool:
        return (
            conflict["time_gap"] < self._committed_time_window
            and float(getattr(ego, "speed", 0.0)) > self._committed_speed_threshold
            and conflict["min_distance"] >= self._emergency_distance
        )

    def _finalize(self, idm_acc, accel_cmd, phase) -> float:
        self._last_phase = phase
        if float(accel_cmd) >= float(idm_acc) - 1e-6:
            self._last_accel = None
            self.last_diagnostics = {
                "phase": phase.name,
                "active": False,
                "acc": float(idm_acc),
            }
            return float(idm_acc)
        final_acc = self._limit_accel_change(float(accel_cmd))
        self.last_diagnostics = {
            "phase": phase.name,
            "active": True,
            "acc": float(final_acc),
        }
        return float(final_acc)

    def reset(self) -> None:
        self._last_accel = None
        self._last_phase = RoundaboutPhase.NONE
