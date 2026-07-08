from __future__ import annotations

import copy
import math
import numpy as np


class PlatoonNormalPlanner:
    """Deterministic platoon trajectory planner based on a small Frenet lattice."""

    def __init__(
        self,
        *,
        num_output_points: int = 8,
        sample_points: int = 32,
        lane_change_target_margin_m: float = 0.3,
        keep_lateral_margin_m: float = 0.5,
        safety_distance_m: float = 6.0,
        safety_weight: float = 8.0,
        ttc_threshold_s: float = 3.0,
        ttc_weight: float = 4.0,
        hard_collision_check_enabled: bool = True,
        collision_margin_m: float = 0.2,
    ) -> None:
        self.num_output_points = int(num_output_points)
        self.sample_points = int(sample_points)
        self.lane_change_target_margin_m = float(lane_change_target_margin_m)
        self.keep_lateral_margin_m = float(keep_lateral_margin_m)
        self.safety_distance_m = float(safety_distance_m)
        self.safety_weight = float(safety_weight)
        self.ttc_threshold_s = float(ttc_threshold_s)
        self.ttc_weight = float(ttc_weight)
        self.hard_collision_check_enabled = bool(hard_collision_check_enabled)
        self.collision_margin_m = float(collision_margin_m)
        self._last_debug: dict | None = None

    def plan(self, env, agent_decisions) -> dict[str, np.ndarray]:
        results: dict[str, np.ndarray] = {}
        debug: dict[str, dict] = {}
        agents = getattr(env, "agents", {}) or {}
        for agent_id, decision in (agent_decisions or {}).items():
            vehicle = agents.get(agent_id)
            if vehicle is None:
                continue
            action = int((decision or {}).get("action", 0))
            target_point = np.asarray((decision or {}).get("target_point", [15.0, 0.0]), dtype=np.float32).reshape(2)
            trajectory, agent_debug = self._plan_for_vehicle(env, vehicle, action, target_point)
            results[agent_id] = trajectory
            debug[agent_id] = agent_debug
        self._last_debug = debug
        return results

    def get_last_debug(self) -> dict | None:
        return copy.deepcopy(self._last_debug)

    def _plan_for_vehicle(self, env, vehicle, action: int, target_point: np.ndarray) -> tuple[np.ndarray, dict]:
                
        source_lane = getattr(vehicle, "lane", None)
        if source_lane is None:
            fallback = self._fallback_keep_trajectory(vehicle)
            return fallback, {
                "fallback_used": True,
                "fallback_reason": "missing_source_lane",
                "candidate_scores": [],
                "best_index": None,
                "candidate_count": 0,
            }

        target_world = self._ego_local_to_world(vehicle, target_point)
        start_s, start_d = source_lane.local_coordinates(np.asarray(vehicle.position[:2], dtype=np.float32))  # 车辆当前位置映射
        target_lane = self._resolve_target_lane(env, source_lane, action)
        effective_action = int(action) if target_lane is not None else 0
        if target_lane is None:
            target_lane = source_lane

        continuation_lane = self._get_continuation_lane(env, vehicle, source_lane)
        source_length = float(getattr(source_lane, "length", 0.0))

        # 车辆可能刚跨过路段边界但 vehicle.lane 还没更新，planner 会尝试切到 continuation_lane，然后重新计算start_s, start_d
        if start_s > source_length and continuation_lane is not None:
            cont_length_tmp = float(getattr(continuation_lane, "length", 0.0))
            try:
                stitch_pt = np.asarray(source_lane.position(source_length, 0.0)[:2], dtype=np.float32)
                _rs, _rd = continuation_lane.local_coordinates(stitch_pt)
                _s_base = float(np.clip(_rs, 0.0, cont_length_tmp))
                new_start_s = _s_base + (start_s - source_length)
                if 0.0 <= new_start_s <= cont_length_tmp:
                    new_start_d_pt = continuation_lane.position(new_start_s, 0.0)[:2]
                    _new_s, _new_d = continuation_lane.local_coordinates(
                        np.asarray(vehicle.position[:2], dtype=np.float32)
                    )
                    source_lane = continuation_lane
                    start_s = float(np.clip(_new_s, 0.0, cont_length_tmp))
                    start_d = float(_new_d)
                    source_length = cont_length_tmp
                    continuation_lane = self._get_continuation_lane(env, vehicle, source_lane)
            except Exception:
                pass

        cont_length = float(getattr(continuation_lane, "length", 0.0)) if continuation_lane else 0.0
        cont_s_base = 0.0
        cont_d_offset = 0.0
        if continuation_lane is not None:
            try:
                stitch_pt = np.asarray(source_lane.position(source_length, 0.0)[:2], dtype=np.float32)
                _raw_s, _raw_d = continuation_lane.local_coordinates(stitch_pt)
                cont_s_base = float(np.clip(_raw_s, 0.0, cont_length))
                cont_d_offset = float(_raw_d)
            except Exception:
                pass
        remaining_cont = max(cont_length - cont_s_base, 0.0)
        total_length = source_length + remaining_cont

        desired_end_s = float(source_lane.local_coordinates(target_world)[0])  # 取target_point作为desired_end_s
        desired_end_s = max(float(start_s) + 5.0, desired_end_s)  # 至少前进5m
        desired_end_s = min(desired_end_s, total_length) # 不超出total_length
        desired_end_d = self._desired_end_lateral(source_lane, target_lane, effective_action, desired_end_s, target_world, start_d)

        'lattice轨迹采样过程，从时间、纵向距离、横向距离三个维度采样，生成候选轨迹，并对每条候选轨迹进行打分，选择最优轨迹'
        candidates: list[np.ndarray] = []
        scores: list[float] = []
        # 1. 时间维度
        for duration in self._candidate_durations(): 
            # 2. 纵向距离维度
            for s_offset in self._candidate_longitudinal_offsets():
                s_end = float(np.clip(desired_end_s + s_offset, float(start_s) + 3.0, total_length))
                # 3. 横向距离维度
                for d_end in self._candidate_lateral_targets(effective_action, desired_end_d, source_lane):
                    candidate = self._build_frenet_candidate(
                        source_lane=source_lane,
                        start_s=float(start_s),
                        start_d=float(start_d),
                        ego_speed_mps=max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.1),
                        end_s=s_end,
                        end_d=float(d_end),
                        duration=float(duration),
                        continuation_lane=continuation_lane,
                        cont_s_base=cont_s_base,
                        cont_d_offset=cont_d_offset,
                    )

                    if candidate is None:
                        continue
                    # !碰撞检测!!!效果待验证
                    if self._candidate_collides_with_predicted_vehicles(
                        candidate,
                        env=env,
                        vehicle=vehicle,
                        duration=float(duration),
                    ):
                        candidate = None
                        continue
                    # !动力学检测!!!待添加
                    
                    score = self._score_candidate(
                        candidate,
                        target_world=target_world,
                        source_lane=source_lane,
                        target_lane=target_lane,
                        action=effective_action,
                        desired_end_d=float(desired_end_d),
                        env=env,
                        vehicle=vehicle,
                        duration=float(duration),
                    )
                    if np.isfinite(score):
                        candidates.append(candidate)
                        scores.append(float(score))

        if not candidates:
            fallback = self._fallback_keep_trajectory(vehicle, source_lane=source_lane)
            return fallback, {
                "fallback_used": True,
                "fallback_reason": "no_valid_candidates",
                "candidate_scores": [],
                "best_index": None,
                "candidate_count": 0,
            }

        best_index = int(np.argmin(np.asarray(scores, dtype=np.float64)))
        resampled_candidates = [self._resample_to_8x3(candidate) for candidate in candidates]
        best = resampled_candidates[best_index]

        debug = 0
        if debug:
            # 新建figure，绘制resampled_candidates中的所有轨迹，并将best轨迹高亮
            try:
                from models.platoon_planner.platoon_planner_helper import save_platoon_candidates_debug_plot

                save_platoon_candidates_debug_plot(vehicle, resampled_candidates, best_index)
            except Exception:
                pass


        return best, {
            "fallback_used": False,
            "fallback_reason": None,
            "candidate_scores": list(scores),
            "best_index": best_index,
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "score": float(score),
                    "selected": int(idx) == best_index,
                    "trajectory_world": resampled_candidates[idx].astype(np.float32, copy=False).tolist(),
                }
                for idx, score in enumerate(scores)
            ],
        }

    def _candidate_durations(self) -> tuple[float, ...]:
        return (3.0, 4.0, 5.0)

    def _candidate_longitudinal_offsets(self) -> tuple[float, ...]:
        # return (-8.0, -4.0, 0.0, 4.0, 8.0)
        return range(-60, 60, 2)  # (-60, -19, ..., 0, ..., 18, 59)

    def _candidate_lateral_targets(self, action: int, desired_end_d: float, source_lane) -> tuple[float, ...]:
        lane_half_width = 0.5 * float(getattr(source_lane, "width", 3.5) or 3.5)
        if action == 0:
            base = [
                np.clip(desired_end_d - self.keep_lateral_margin_m, -lane_half_width, lane_half_width),
                np.clip(desired_end_d, -lane_half_width, lane_half_width),
                np.clip(desired_end_d + self.keep_lateral_margin_m, -lane_half_width, lane_half_width),
            ]
        else:
            base = [
                desired_end_d - self.lane_change_target_margin_m,
                desired_end_d,
                desired_end_d + self.lane_change_target_margin_m,
            ]
        return tuple(float(x) for x in base)

    @staticmethod
    def _get_continuation_lane(env, vehicle, source_lane):
        """Return the next lane segment after source_lane, selected by geometric proximity.

        The candidate whose centerline START is closest to source_lane's END is selected.
        If all route-aware candidates are more than _MAX_STITCH_DIST away, the full road
        network is searched to find the physically adjacent lane (handles complex merge
        blocks where next_ref_lanes may point to upstream segments).
        """
        _MAX_STITCH_DIST = 10.0  # metres; candidates farther than this trigger full search

        lane_index = getattr(source_lane, "index", None)
        if lane_index is None or len(lane_index) < 3:
            return None

        source_length = float(getattr(source_lane, "length", 0.0))
        try:
            source_end = np.asarray(source_lane.position(source_length, 0.0)[:2], dtype=np.float32)
        except Exception:
            return None

        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        road_network = getattr(current_map, "road_network", None) if current_map is not None else None

        # --- Primary candidates: navigation (route-aware) + graph[end_node] ---
        candidates = []
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None:
            next_ref_lanes = getattr(navigation, "next_ref_lanes", None)
            if next_ref_lanes:
                candidates.extend(next_ref_lanes)
        if road_network is not None:
            end_node = lane_index[1]
            for next_lanes in (getattr(road_network, "graph", None) or {}).get(end_node, {}).values():
                candidates.extend(next_lanes)

        best, best_dist = None, float("inf")
        for lane in candidates:
            try:
                lane_start = np.asarray(lane.position(0.0, 0.0)[:2], dtype=np.float32)
                dist = float(np.linalg.norm(lane_start - source_end))
                if dist < best_dist:
                    best_dist, best = dist, lane
            except Exception:
                continue

        # --- Fallback: search all lanes when primary candidates are too far away ---
        # This handles complex merge blocks (e.g. DoubleRamp) where next_ref_lanes
        # points to a segment that starts far upstream of the physical stitch point.
        if best_dist > _MAX_STITCH_DIST and road_network is not None:
            for _, end_dict in (getattr(road_network, "graph", None) or {}).items():
                for _, lanes in end_dict.items():
                    for lane in lanes:
                        if lane is source_lane:
                            continue
                        try:
                            lane_start = np.asarray(lane.position(0.0, 0.0)[:2], dtype=np.float32)
                            dist = float(np.linalg.norm(lane_start - source_end))
                            if dist < best_dist:
                                best_dist, best = dist, lane
                        except Exception:
                            continue

        return best if best_dist <= _MAX_STITCH_DIST else None

    @staticmethod
    def _resolve_target_lane(env, source_lane, action: int):
        if action == 0 or source_lane is None:
            return source_lane
        lane_index = tuple(getattr(source_lane, "index", ()) or ())
        if len(lane_index) < 3:
            return None
        road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
        if road_network is None or not hasattr(road_network, "get_lane"):
            return None
        if lane_index == ("3C0_1_", "4G0_0_", 2) and int(action) == 1:
            try:
                return road_network.get_lane(("3C0_1_", "4G1_0_", 0))
            except Exception:
                return None
        target_index = (lane_index[0], lane_index[1], int(lane_index[2]) + int(action))
        try:
            return road_network.get_lane(target_index)
        except Exception:
            return None

    @staticmethod
    def _ego_local_to_world(vehicle, target_point: np.ndarray) -> np.ndarray:
        ego_pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        heading = float(getattr(vehicle, "heading_theta", 0.0))
        x_fwd, y_lat = float(target_point[0]), float(target_point[1])
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        world = np.asarray(
            [
                ego_pos[0] + cos_h * x_fwd - sin_h * y_lat,
                ego_pos[1] + sin_h * x_fwd + cos_h * y_lat,
            ],
            dtype=np.float32,
        )
        return world

    def _desired_end_lateral(self, source_lane, target_lane, action: int, desired_end_s: float, target_world: np.ndarray, start_d: float) -> float:
        if action == 0:
            _, target_d = source_lane.local_coordinates(target_world)
            lane_half_width = 0.2 * float(getattr(source_lane, "width", 3.5) or 3.5)
            return float(np.clip(target_d, -lane_half_width, lane_half_width))

        ref_world = np.asarray(target_lane.position(float(np.clip(desired_end_s, 0.0, float(getattr(target_lane, "length", desired_end_s)))), 0.0)[:2], dtype=np.float32)
        _, desired_d = source_lane.local_coordinates(ref_world)
        # if action < 0:
        #     desired_d = max(desired_d, start_d + 0.5)
        # else:
        #     desired_d = min(desired_d, start_d - 0.5)
        return float(desired_d)

    def _build_frenet_candidate(
        self,
        *,
        source_lane,
        start_s: float,
        start_d: float,
        ego_speed_mps: float,
        end_s: float,
        end_d: float,
        duration: float,
        continuation_lane=None,
        cont_s_base: float = 0.0,
        cont_d_offset: float = 0.0,
    ) -> np.ndarray | None:
        if duration <= 1e-3 or end_s <= start_s:
            return None
        source_length = float(getattr(source_lane, "length", max(end_s, start_s)))
        cont_length = float(getattr(continuation_lane, "length", 0.0)) if continuation_lane else 0.0
        # Effective remaining distance in continuation_lane after the stitch point.
        # cont_s_base is where source_lane's end projects onto continuation_lane; the
        # usable range from the stitch onward is [cont_s_base, cont_length].
        remaining_cont = max(cont_length - cont_s_base, 0.0)
        total_length = source_length + remaining_cont
        accel = 2.0 * (float(end_s) - float(start_s) - float(ego_speed_mps) * float(duration)) / max(float(duration) ** 2, 1e-6)
        coeffs = self._solve_quintic_coefficients(float(start_d), 0.0, 0.0, float(end_d), 0.0, 0.0, float(duration))
        times = np.linspace(0.0, float(duration), self.sample_points, dtype=np.float64)
        s_values = float(start_s) + float(ego_speed_mps) * times + 0.5 * float(accel) * times**2
        s_values = np.clip(s_values, 0.0, total_length)
        if np.any(np.diff(s_values) < -1e-5):
            return None
        points = []
        for t, s in zip(times, s_values):
            lateral, _, _ = self._evaluate_quintic(coeffs, float(t))
            if s <= source_length or continuation_lane is None:
                world = np.asarray(
                    source_lane.position(float(np.clip(s, 0.0, source_length)), float(lateral))[:2],
                    dtype=np.float64,
                )
            else:
                s_in_cont = float(cont_s_base + (s - source_length))
                world = np.asarray(
                    continuation_lane.position(
                        float(np.clip(s_in_cont, 0.0, cont_length)),
                        float(lateral + cont_d_offset),
                    )[:2],
                    dtype=np.float64,
                )
            points.append(world)
        points_arr = np.asarray(points, dtype=np.float64)
        if points_arr.shape[0] < 2 or not np.all(np.isfinite(points_arr)):
            return None
        return self._append_heading(points_arr)

    @staticmethod
    def _solve_quintic_coefficients(d0, d0_dot, d0_ddot, df, df_dot, df_ddot, T) -> np.ndarray:
        a0 = d0
        a1 = d0_dot
        a2 = d0_ddot / 2.0
        matrix = np.asarray(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ],
            dtype=np.float64,
        )
        rhs = np.asarray(
            [
                df - (a0 + a1 * T + a2 * T**2),
                df_dot - (a1 + 2 * a2 * T),
                df_ddot - (2 * a2),
            ],
            dtype=np.float64,
        )
        a3, a4, a5 = np.linalg.solve(matrix, rhs)
        return np.asarray([a0, a1, a2, a3, a4, a5], dtype=np.float64)

    @staticmethod
    def _evaluate_quintic(coeffs, t) -> tuple[float, float, float]:
        d = sum(coeff * (t**idx) for idx, coeff in enumerate(coeffs))
        d_dot = sum(idx * coeff * (t ** (idx - 1)) for idx, coeff in enumerate(coeffs) if idx >= 1)
        d_ddot = sum(idx * (idx - 1) * coeff * (t ** (idx - 2)) for idx, coeff in enumerate(coeffs) if idx >= 2)
        return float(d), float(d_dot), float(d_ddot)

    @staticmethod
    def _append_heading(points_xy: np.ndarray) -> np.ndarray:
        headings = np.zeros((points_xy.shape[0],), dtype=np.float64)
        if points_xy.shape[0] >= 2:
            deltas = np.diff(points_xy, axis=0)
            seg_headings = np.arctan2(deltas[:, 1], deltas[:, 0])
            headings[:-1] = seg_headings
            headings[-1] = seg_headings[-1]
        return np.concatenate([points_xy, headings[:, None]], axis=1)

    def _score_candidate(
        self,
        candidate: np.ndarray,
        *,
        target_world: np.ndarray,
        source_lane,
        target_lane,
        action: int,
        desired_end_d: float,
        env=None,
        vehicle=None,
        duration: float | None = None,
    ) -> float:
        if candidate.shape[0] < 2:
            return float("inf")
        endpoint = candidate[-1, :2]
        end_dist = float(np.linalg.norm(endpoint - np.asarray(target_world[:2], dtype=np.float64)))
        _, end_d = source_lane.local_coordinates(endpoint)
        end_d_error = abs(float(end_d) - float(desired_end_d))
        heading_jumps = np.abs(np.diff(candidate[:, 2]))
        heading_jump_penalty = float(np.max(heading_jumps)) if heading_jumps.size > 0 else 0.0
        seg_lengths = np.linalg.norm(np.diff(candidate[:, :2], axis=0), axis=1)
        if np.any(seg_lengths < 1e-6):
            return float("inf")
        curvature_proxy = (
            float(np.mean(np.abs(np.diff(candidate[:, 2])) / np.maximum(seg_lengths, 1e-6)))
            if candidate.shape[0] >= 3
            else 0.0
        )

        action_penalty = 0.0
        if action == 0:
            lane_half_width = 0.5 * float(getattr(source_lane, "width", 3.5) or 3.5)
            if abs(float(end_d)) > lane_half_width + 0.1:
                return float("inf")
        else:
            target_center_world = np.asarray(
                target_lane.position(float(np.clip(source_lane.local_coordinates(endpoint)[0], 0.0, float(getattr(target_lane, "length", 0.0) or 0.0))), 0.0)[:2],
                dtype=np.float64,
            )
            lane_sep = float(np.linalg.norm(target_center_world - np.asarray(source_lane.position(float(np.clip(source_lane.local_coordinates(endpoint)[0], 0.0, float(getattr(source_lane, "length", 0.0) or 0.0))), 0.0)[:2], dtype=np.float64)))
            if lane_sep < 1.0:
                action_penalty += 10.0
            if action < 0 and float(end_d) >= 0.0:
                action_penalty += 5.0
            if action > 0 and float(end_d) <= 0.0:
                action_penalty += 5.0

        safety_penalty = self._safety_distance_penalty(candidate, env=env, vehicle=vehicle)
        ttc_penalty = self._ttc_penalty(
            candidate,
            source_lane=source_lane,
            env=env,
            vehicle=vehicle,
            duration=duration,
        )

        return (
            end_dist
            + 1.5 * end_d_error
            + 2.0 * heading_jump_penalty
            + 1.5 * curvature_proxy
            + action_penalty
            + safety_penalty
            + ttc_penalty
        )

    def _safety_distance_penalty(self, candidate: np.ndarray, *, env=None, vehicle=None) -> float:
        if self.safety_weight <= 0.0 or self.safety_distance_m <= 1e-6 or env is None:
            return 0.0
        min_dist = self._min_distance_to_other_agents(candidate, env=env, vehicle=vehicle)
        if not np.isfinite(min_dist) or min_dist >= self.safety_distance_m:
            return 0.0
        normalized_shortfall = (self.safety_distance_m - min_dist) / max(self.safety_distance_m, 1e-6)
        return float(self.safety_weight * normalized_shortfall)

    def _ttc_penalty(self, candidate: np.ndarray, *, source_lane, env=None, vehicle=None, duration: float | None = None) -> float:
        if self.ttc_weight <= 0.0 or self.ttc_threshold_s <= 1e-6 or env is None or vehicle is None:
            return 0.0
        agents = getattr(env, "agents", {}) or {}
        ego_name = getattr(vehicle, "name", None)
        ego_speed = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
        if ego_speed <= 1e-3:
            return 0.0
        start_s = float(source_lane.local_coordinates(candidate[0, :2])[0])
        end_s = float(source_lane.local_coordinates(candidate[-1, :2])[0])
        horizon = float(duration) if duration is not None and duration > 1e-3 else max((end_s - start_s) / ego_speed, 1e-3)
        candidate_speed = max((end_s - start_s) / max(horizon, 1e-3), 0.0)
        worst_penalty = 0.0
        for other_id, other in agents.items():
            if other is vehicle or getattr(other, "name", None) == ego_name or other_id == ego_name:
                continue
            if not self._same_lane(other, source_lane):
                continue
            other_pos = getattr(other, "position", None)
            if other_pos is None:
                continue
            try:
                other_s, other_d = source_lane.local_coordinates(np.asarray(other_pos[:2], dtype=np.float64))
            except Exception:
                continue
            if abs(float(other_d)) > 0.5 * float(getattr(source_lane, "width", 3.5) or 3.5) + 0.5:
                continue
            gap = float(other_s) - start_s
            if gap <= 0.0:
                continue
            other_speed = max(float(getattr(other, "speed_km_h", 0.0) or 0.0) / 3.6, 0.0)
            closing_speed = candidate_speed - other_speed
            if closing_speed <= 1e-3:
                continue
            ttc = gap / closing_speed
            if ttc >= self.ttc_threshold_s:
                continue
            normalized = (self.ttc_threshold_s - ttc) / max(self.ttc_threshold_s, 1e-6)
            worst_penalty = max(worst_penalty, float(self.ttc_weight * normalized))
        return float(worst_penalty)

    def _min_distance_to_other_agents(self, candidate: np.ndarray, *, env=None, vehicle=None) -> float:
        agents = getattr(env, "agents", {}) or {}
        ego_name = getattr(vehicle, "name", None)
        min_dist = float("inf")
        for other_id, other in agents.items():
            if other is vehicle or getattr(other, "name", None) == ego_name or other_id == ego_name:
                continue
            other_pos = getattr(other, "position", None)
            if other_pos is None:
                continue
            try:
                other_xy = np.asarray(other_pos[:2], dtype=np.float64)
            except Exception:
                continue
            distances = np.linalg.norm(candidate[:, :2] - other_xy[None, :], axis=1)
            if distances.size:
                min_dist = min(min_dist, float(np.min(distances)))
        return float(min_dist)

    def _candidate_collides_with_predicted_vehicles(
        self,
        candidate: np.ndarray,
        *,
        env=None,
        vehicle=None,
        duration: float | None = None,
    ) -> bool:
        if not self.hard_collision_check_enabled or env is None or vehicle is None:
            return False
        if candidate is None or candidate.shape[0] == 0:
            return False

        ego_length, ego_width = self._vehicle_dimensions(vehicle)
        times = np.linspace(
            0.0,
            max(float(duration or 0.0), 0.0),
            int(candidate.shape[0]),
            dtype=np.float64,
        )
        margin = max(float(self.collision_margin_m), 0.0)
        ego_name = getattr(vehicle, "name", None)

        for other_id, other in self._surrounding_vehicles(env):
            if other is vehicle or getattr(other, "name", None) == ego_name or other_id == ego_name:
                continue
            other_pos = getattr(other, "position", None)
            if other_pos is None:
                continue
            try:
                other_xy0 = np.asarray(other_pos[:2], dtype=np.float64)
            except Exception:
                continue
            if other_xy0.shape[0] < 2 or not np.all(np.isfinite(other_xy0[:2])):
                continue

            other_velocity = self._vehicle_velocity_xy(other)
            other_length, other_width = self._vehicle_dimensions(other)
            half_length_sum = 0.5 * (ego_length + other_length) + margin
            half_width_sum = 0.5 * (ego_width + other_width) + margin
            predicted_xy = other_xy0[None, :2] + times[:, None] * other_velocity[None, :]
            deltas = np.asarray(candidate[:, :2], dtype=np.float64) - predicted_xy
            if np.any((np.abs(deltas[:, 0]) <= half_length_sum) & (np.abs(deltas[:, 1]) <= half_width_sum)):
                return True
        return False

    @staticmethod
    def _surrounding_vehicles(env) -> list[tuple[object, object]]:
        vehicles: list[tuple[object, object]] = []
        seen: set[int] = set()

        agents = getattr(env, "agents", {}) or {}
        if isinstance(agents, dict):
            iterable = agents.items()
        else:
            iterable = enumerate(agents)
        for vehicle_id, other in iterable:
            if other is None or id(other) in seen:
                continue
            seen.add(id(other))
            vehicles.append((vehicle_id, other))

        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        if traffic_manager is None:
            return vehicles
        traffic_vehicles = getattr(traffic_manager, "traffic_vehicles", None)
        if traffic_vehicles is None:
            traffic_vehicles = getattr(traffic_manager, "_traffic_vehicles", []) or []
        try:
            traffic_iterable = list(traffic_vehicles)
        except TypeError:
            traffic_iterable = []
        for idx, other in enumerate(traffic_iterable):
            if other is None or id(other) in seen:
                continue
            seen.add(id(other))
            vehicles.append((getattr(other, "name", f"traffic_{idx}"), other))
        return vehicles

    @staticmethod
    def _vehicle_dimensions(vehicle) -> tuple[float, float]:
        length = getattr(vehicle, "LENGTH", None)
        if length is None:
            length = getattr(vehicle, "length", 5.74)
        width = getattr(vehicle, "WIDTH", None)
        if width is None:
            width = getattr(vehicle, "width", 2.3)
        try:
            length = float(length)
        except (TypeError, ValueError):
            length = 5.74
        try:
            width = float(width)
        except (TypeError, ValueError):
            width = 2.3
        if not np.isfinite(length) or length <= 0.0:
            length = 5.74
        if not np.isfinite(width) or width <= 0.0:
            width = 2.3
        return float(length), float(width)

    @staticmethod
    def _vehicle_velocity_xy(vehicle) -> np.ndarray:
        velocity = getattr(vehicle, "velocity", None)
        if velocity is None:
            return np.zeros((2,), dtype=np.float64)
        try:
            velocity_xy = np.asarray(velocity[:2], dtype=np.float64)
        except Exception:
            return np.zeros((2,), dtype=np.float64)
        if velocity_xy.shape[0] < 2 or not np.all(np.isfinite(velocity_xy[:2])):
            return np.zeros((2,), dtype=np.float64)
        return velocity_xy[:2]

    @staticmethod
    def _same_lane(other, source_lane) -> bool:
        source_index = tuple(getattr(source_lane, "index", ()) or ())
        other_lane_index = getattr(other, "lane_index", None)
        if other_lane_index is not None and tuple(other_lane_index) == source_index:
            return True
        other_lane = getattr(other, "lane", None)
        return tuple(getattr(other_lane, "index", ()) or ()) == source_index

    def _resample_to_8x3(self, candidate: np.ndarray) -> np.ndarray:
        if candidate.shape[0] == self.num_output_points:
            return candidate.astype(np.float32, copy=False)
        indices = np.linspace(0, candidate.shape[0] - 1, self.num_output_points)
        x = np.interp(indices, np.arange(candidate.shape[0]), candidate[:, 0])
        y = np.interp(indices, np.arange(candidate.shape[0]), candidate[:, 1])
        h = np.interp(indices, np.arange(candidate.shape[0]), np.unwrap(candidate[:, 2]))
        return np.stack([x, y, h], axis=1).astype(np.float32, copy=False)

    def _fallback_keep_trajectory(self, vehicle, source_lane=None) -> np.ndarray:
        lane = source_lane if source_lane is not None else getattr(vehicle, "lane", None)
        start_pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        heading = float(getattr(vehicle, "heading_theta", 0.0))
        if lane is None:
            step = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6, 3.0)
            points = []
            for idx in range(self.num_output_points):
                dist = idx * step
                points.append(
                    [
                        float(start_pos[0] + math.cos(heading) * dist),
                        float(start_pos[1] + math.sin(heading) * dist),
                        heading,
                    ]
                )
            return np.asarray(points, dtype=np.float32)

        start_s, start_d = lane.local_coordinates(start_pos)
        step = max(float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6 * 0.6, 2.0)
        points = []
        for idx in range(self.num_output_points):
            s = float(np.clip(float(start_s) + idx * step, 0.0, float(getattr(lane, "length", 0.0) or 0.0)))
            world = np.asarray(lane.position(s, float(start_d))[:2], dtype=np.float32)
            lane_heading = float(lane.heading_theta_at(s)) if hasattr(lane, "heading_theta_at") else heading
            points.append([float(world[0]), float(world[1]), lane_heading])
        return np.asarray(points, dtype=np.float32)
