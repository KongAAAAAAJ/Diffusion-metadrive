from __future__ import annotations

import math

import numpy as np

from expert_dataset.hierarchical_expert.lane_change_manager import ManeuverState


class IntersectionConflictRegulator:
    def __init__(
        self,
        horizon: float = 3.0,
        dt: float = 0.2,
        candidate_radius: float = 40.0,
        conflict_distance: float = 4.5,
        safe_time_gap: float = 1.5,
        max_braking: float = -4.5,
        accel_rate_limit: float = 1.0,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.candidate_radius = float(candidate_radius)
        self.conflict_distance = float(conflict_distance)
        self.safe_time_gap = float(safe_time_gap)
        self.max_braking = float(max_braking)
        self.accel_rate_limit = float(accel_rate_limit)
        self._speed_epsilon = 1e-3
        self._committed_time_window = 0.8
        self._committed_speed_threshold = 3.0
        self._emergency_distance = 2.5
        self._last_accel = None
        self.last_diagnostics = {
            "active": False,
            "conflict_count": 0,
            "conflict_acc": 0.0,
        }

    def adjust_acceleration(self, ego, all_objects, reference_target, idm_acc, maneuver_state=None) -> float:
        candidates = self._filter_candidate_vehicles(ego, all_objects)
        ego_path = self._predict_ego_path(ego, reference_target, maneuver_state)
        conflicts = []
        for obj in candidates:
            conflict = self._evaluate_conflict(ego, ego_path, obj)
            if conflict is not None:
                conflicts.append(conflict)

        target_acc = float(idm_acc)
        if conflicts:
            selected_conflicts = self._find_most_dangerous_conflicts(conflicts)
            target_acc = min(
                self._target_acc_for_conflict(ego, selected, idm_acc)
                for selected in selected_conflicts
            )

        not_intervening = target_acc >= float(idm_acc) - 1e-6
        if not_intervening:
            self._last_accel = None
            self.last_diagnostics = {
                "active": False,
                "conflict_count": len(conflicts),
                "conflict_acc": float(idm_acc),
            }
            return float(idm_acc)
        final_acc = self._limit_accel_change(target_acc)
        self.last_diagnostics = {
            "active": True,
            "conflict_count": len(conflicts),
            "conflict_acc": float(final_acc),
        }
        return float(final_acc)

    def _filter_candidate_vehicles(self, ego, all_objects) -> list:
        candidates = []
        ego_position = np.asarray(getattr(ego, "position", np.zeros(2)), dtype=np.float64)
        ego_dir = self._object_direction(ego)
        for obj in all_objects:
            if obj is ego:
                continue
            if not hasattr(obj, "position"):
                continue
            position = np.asarray(obj.position, dtype=np.float64)
            if not np.all(np.isfinite(position)):
                continue
            if np.linalg.norm(position - ego_position) > self.candidate_radius:
                continue
            # Skip same-direction vehicles — those are handled by IDM / rear-end guard
            obj_dir = self._object_direction(obj)
            cos_angle = float(np.dot(ego_dir, obj_dir))
            if cos_angle > 0.7:  # < ~45° heading difference
                continue
            candidates.append(obj)
        return candidates

    def _predict_ego_path(self, ego, reference_target, maneuver_state) -> np.ndarray:
        if reference_target is not None and hasattr(reference_target, "local_coordinates") and hasattr(reference_target, "position"):
            try:
                long, _ = reference_target.local_coordinates(np.asarray(ego.position, dtype=np.float64))
                long = float(max(long, 0.0))
                lane_length = float(getattr(reference_target, "length", long + self.horizon * max(float(ego.speed), 1.0)))
                samples = []
                for t in self._time_samples():
                    sample_long = float(np.clip(long + max(float(ego.speed), 0.5) * t, 0.0, lane_length))
                    samples.append(np.asarray(reference_target.position(sample_long, 0.0), dtype=np.float64))
                return np.stack(samples, axis=0)
            except Exception:
                pass

        heading = self._object_direction(ego)
        origin = np.asarray(ego.position, dtype=np.float64)
        return np.stack([origin + heading * max(float(getattr(ego, "speed", 0.0)), 0.5) * t for t in self._time_samples()], axis=0)

    def _predict_object_path(self, obj) -> np.ndarray:
        reference_lane = self._object_reference_lane(obj)
        speed = self._object_speed(obj)
        if reference_lane is not None and hasattr(reference_lane, "local_coordinates") and hasattr(reference_lane, "position"):
            try:
                long, _ = reference_lane.local_coordinates(np.asarray(obj.position, dtype=np.float64))
                long = float(max(long, 0.0))
                lane_length = float(getattr(reference_lane, "length", long + self.horizon * max(speed, 1.0)))
                samples = []
                for t in self._time_samples():
                    sample_long = float(np.clip(long + max(speed, 0.5) * t, 0.0, lane_length))
                    samples.append(np.asarray(reference_lane.position(sample_long, 0.0), dtype=np.float64))
                return np.stack(samples, axis=0)
            except Exception:
                pass

        origin = np.asarray(obj.position, dtype=np.float64)
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

    def _evaluate_conflict(self, ego, ego_path: np.ndarray, obj) -> dict | None:
        obj_path = self._predict_object_path(obj)
        speed_ego = max(float(getattr(ego, "speed", 0.0)), self._speed_epsilon)
        speed_obj = max(self._object_speed(obj), self._speed_epsilon)

        diff = ego_path[:, np.newaxis, :] - obj_path[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        min_distance = float(np.min(dist_matrix))
        if min_distance >= self.conflict_distance:
            return None

        conflict_mask = dist_matrix < self.conflict_distance
        ego_indices, obj_indices = np.where(conflict_mask)
        if ego_indices.size == 0:
            return None

        ego_times = ego_indices.astype(np.float64) * self.dt
        obj_times = obj_indices.astype(np.float64) * self.dt
        arrival_diff = np.abs(ego_times - obj_times)
        max_arrival_diff = self.safe_time_gap + 1.0
        temporal_mask = arrival_diff < max_arrival_diff
        if not np.any(temporal_mask):
            return None

        valid_indices = np.flatnonzero(temporal_mask)
        order = np.lexsort((arrival_diff[valid_indices], ego_times[valid_indices]))
        best_valid = int(valid_indices[int(order[0])])
        conflict_ego_time = float(ego_times[best_valid])
        conflict_obj_time = float(obj_times[best_valid])
        return {
            "time_gap": conflict_ego_time,
            "min_distance": min_distance,
            "s_ego": speed_ego * conflict_ego_time,
            "s_obj": speed_obj * conflict_obj_time,
            "obj_speed": speed_obj,
            "arrival_diff": float(arrival_diff[best_valid]),
        }

    def _find_most_dangerous_conflicts(self, conflicts: list[dict], top_k: int = 2) -> list[dict]:
        if top_k <= 0:
            return []
        ranked = sorted(conflicts, key=lambda item: (item["time_gap"], item["min_distance"]))
        return ranked[:top_k]

    def _target_acc_for_conflict(self, ego, conflict: dict, idm_acc: float) -> float:
        if self._should_hold_course(ego, conflict):
            return float(idm_acc)
        if conflict["time_gap"] < self._committed_time_window and conflict["min_distance"] < self._emergency_distance:
            return self.max_braking
        return self._compute_required_acceleration(
            s_ego=conflict["s_ego"],
            v_ego=max(float(getattr(ego, "speed", 0.0)), 0.0),
            s_obj=conflict["s_obj"],
            v_obj=conflict["obj_speed"],
            idm_acc=idm_acc,
        )

    def _compute_required_acceleration(self, s_ego, v_ego, s_obj, v_obj, idm_acc) -> float:
        eps = self._speed_epsilon
        t_obj = float(s_obj / max(v_obj, eps))
        t_target = max(t_obj + self.safe_time_gap, self.dt)
        a_conf = 2.0 * (float(s_ego) - float(v_ego) * t_target) / (t_target**2)
        a_conf = min(float(a_conf), float(idm_acc))
        return float(np.clip(a_conf, self.max_braking, np.inf))

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

    def _should_hold_course(self, ego, conflict: dict) -> bool:
        if conflict["time_gap"] >= self._committed_time_window:
            return False
        ego_speed = max(float(getattr(ego, "speed", 0.0)), self._speed_epsilon)
        if ego_speed <= self._committed_speed_threshold:
            return False
        ttc_to_closest = float(conflict["min_distance"]) / ego_speed
        return ttc_to_closest > self._committed_time_window

    def _time_samples(self) -> np.ndarray:
        return np.arange(0.0, self.horizon + self.dt * 0.5, self.dt, dtype=np.float64)
