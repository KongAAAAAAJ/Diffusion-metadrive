"""Simulator-ground-truth trajectory anchors for the BEV-only planner.

The generator is intentionally independent of RuleMaker.  It reads only the
current simulator lane graph and ego kinematics, then emits the frozen ten
semantic modes in the ego coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from models.bev_planner.mode_contract import (
    ModeIndex,
    ModeTopology,
    NUM_MODES,
    TRAJECTORY_DIM,
    TRAJECTORY_SHAPE,
    TRAJECTORY_STEPS,
    HardModeMaskConfig,
    validate_trajectory_kinematics,
)


class DynamicAnchorError(ValueError):
    """Raised when simulator topology cannot produce a valid anchor set."""


@dataclass(frozen=True)
class DynamicAnchorConfig:
    dt_s: float = 0.5
    high_accel_mps2: float = 1.5
    medium_accel_mps2: float = 0.0
    low_accel_mps2: float = -2.0
    stop_accel_mps2: float = -4.5
    max_speed_mps: float = 100.0 / 3.6
    max_lateral_lane_distance_m: float = 12.0
    max_lane_heading_error_rad: float = 0.35

    def __post_init__(self) -> None:
        numeric = (
            "dt_s",
            "high_accel_mps2",
            "medium_accel_mps2",
            "low_accel_mps2",
            "stop_accel_mps2",
            "max_speed_mps",
            "max_lateral_lane_distance_m",
            "max_lane_heading_error_rad",
        )
        for name in numeric:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise DynamicAnchorError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.dt_s <= 0.0 or self.max_speed_mps <= 0.0:
            raise DynamicAnchorError("dt_s and max_speed_mps must be positive")
        if self.max_lateral_lane_distance_m <= 0.0 or self.max_lane_heading_error_rad <= 0.0:
            raise DynamicAnchorError("lane search limits must be positive")


@dataclass(frozen=True)
class DynamicAnchorOutput:
    coarse_trajectories: np.ndarray
    topology: ModeTopology

    def __post_init__(self) -> None:
        trajectories = np.asarray(self.coarse_trajectories)
        if trajectories.shape != TRAJECTORY_SHAPE:
            raise DynamicAnchorError(
                f"coarse_trajectories must have shape {TRAJECTORY_SHAPE}, got {trajectories.shape}"
            )
        if trajectories.dtype != np.float32:
            raise DynamicAnchorError(
                f"coarse_trajectories must have dtype float32, got {trajectories.dtype}"
            )
        if not np.isfinite(trajectories).all():
            raise DynamicAnchorError("coarse_trajectories contains non-finite values")
        value = np.ascontiguousarray(trajectories).copy()
        value.setflags(write=False)
        object.__setattr__(self, "coarse_trajectories", value)
        if not isinstance(self.topology, ModeTopology):
            raise DynamicAnchorError("topology must be a ModeTopology instance")


def _wrap_to_pi(angle: float | np.ndarray) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _unique_lanes(lanes: Iterable[object]) -> list[object]:
    result: list[object] = []
    seen: set[int] = set()
    for lane in lanes:
        if lane is None or id(lane) in seen:
            continue
        seen.add(id(lane))
        result.append(lane)
    return result


class SimulatorDynamicAnchorGenerator:
    """Build fixed-mode anchors from the active MetaDrive lane graph."""

    def __init__(self, config: DynamicAnchorConfig | None = None) -> None:
        self.config = config or DynamicAnchorConfig()

    @staticmethod
    def _road_network(env: object) -> object:
        current_map = getattr(env, "current_map", None)
        if current_map is None:
            current_map = getattr(getattr(env, "engine", None), "current_map", None)
        network = getattr(current_map, "road_network", None)
        if network is None:
            raise DynamicAnchorError("environment has no active road network")
        return network

    @staticmethod
    def _lane_position(lane: object, longitudinal: float) -> np.ndarray:
        try:
            return np.asarray(lane.position(float(longitudinal), 0.0), dtype=np.float64).reshape(-1)[:2]
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicAnchorError("lane does not provide finite position(s, 0)") from exc

    @staticmethod
    def _lane_heading(lane: object, longitudinal: float) -> float:
        heading_at = getattr(lane, "heading_theta_at", None)
        if callable(heading_at):
            try:
                return float(heading_at(float(longitudinal)))
            except (TypeError, ValueError):
                pass
        length = max(float(getattr(lane, "length", 0.0) or 0.0), 1e-3)
        s0 = float(np.clip(longitudinal - 0.25, 0.0, length))
        s1 = float(np.clip(longitudinal + 0.25, 0.0, length))
        if abs(s1 - s0) < 1e-6:
            s0, s1 = 0.0, length
        delta = SimulatorDynamicAnchorGenerator._lane_position(lane, s1) - SimulatorDynamicAnchorGenerator._lane_position(lane, s0)
        if float(np.linalg.norm(delta)) < 1e-6:
            raise DynamicAnchorError("lane has no measurable heading")
        return float(np.arctan2(delta[1], delta[0]))

    @staticmethod
    def _world_to_ego(points_world: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
        delta = np.asarray(points_world, dtype=np.float64) - ego_pose[None, :2]
        cos_h = float(np.cos(ego_pose[2]))
        sin_h = float(np.sin(ego_pose[2]))
        rotation = np.asarray([[cos_h, sin_h], [-sin_h, cos_h]], dtype=np.float64)
        return delta @ rotation.T

    @staticmethod
    def _project(lane: object, position: np.ndarray) -> tuple[float, float]:
        try:
            longitudinal, lateral = lane.local_coordinates(np.asarray(position[:2], dtype=np.float64))
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicAnchorError("lane does not provide local_coordinates(position)") from exc
        values = np.asarray([longitudinal, lateral], dtype=np.float64)
        if not np.isfinite(values).all():
            raise DynamicAnchorError("lane projection returned non-finite values")
        return float(values[0]), float(values[1])

    def _direct_peer(self, road_network: object, source_lane: object, direction: int) -> object | None:
        index = tuple(getattr(source_lane, "index", ()) or ())
        if len(index) < 3 or not isinstance(index[-1], (int, np.integer)):
            return None
        # MetaDrive lane ids grow from left to right, so left is id-1.
        target_lane_id = int(index[-1]) - direction
        # Some road-network implementations use Python list indexing; without
        # this guard, -1 would silently select the opposite edge lane.
        if target_lane_id < 0:
            return None
        target_index = (*index[:-1], target_lane_id)
        get_lane = getattr(road_network, "get_lane", None)
        if not callable(get_lane):
            return None
        try:
            lane = get_lane(target_index)
        except (KeyError, IndexError, TypeError):
            return None
        return lane if lane is not source_lane else None

    def _search_parallel_lane(
        self,
        road_network: object,
        source_lane: object,
        ego_pose: np.ndarray,
        direction: int,
    ) -> object | None:
        get_all_lanes = getattr(road_network, "get_all_lanes", None)
        candidates = list(get_all_lanes()) if callable(get_all_lanes) else []
        source_index = tuple(getattr(source_lane, "index", ()) or ())
        source_nodes = set(source_index[:2])
        best: tuple[float, object] | None = None
        for candidate in _unique_lanes(candidates):
            if candidate is source_lane:
                continue
            candidate_index = tuple(getattr(candidate, "index", ()) or ())
            if source_nodes and not source_nodes.intersection(candidate_index[:2]):
                continue
            try:
                s, _ = self._project(candidate, ego_pose[:2])
                length = float(getattr(candidate, "length"))
                s = float(np.clip(s, 0.0, max(length, 0.0)))
                center_world = self._lane_position(candidate, s)
                center_local = self._world_to_ego(center_world[None], ego_pose)[0]
                lateral = float(center_local[1])
                heading_error = abs(float(_wrap_to_pi(self._lane_heading(candidate, s) - ego_pose[2])))
            except (DynamicAnchorError, TypeError, ValueError):
                continue
            if direction * lateral <= 0.5:
                continue
            if abs(lateral) > self.config.max_lateral_lane_distance_m:
                continue
            if heading_error > self.config.max_lane_heading_error_rad:
                continue
            cost = abs(lateral) + 2.0 * heading_error + 0.05 * abs(float(center_local[0]))
            if best is None or cost < best[0]:
                best = (cost, candidate)
        return None if best is None else best[1]

    def _lateral_lane(
        self,
        road_network: object,
        source_lane: object,
        ego_pose: np.ndarray,
        direction: int,
    ) -> object | None:
        direct = self._direct_peer(road_network, source_lane, direction)
        if direct is not None:
            return direct
        return self._search_parallel_lane(road_network, source_lane, ego_pose, direction)

    def _successors(self, road_network: object, lane: object) -> list[object]:
        index = tuple(getattr(lane, "index", ()) or ())
        graph = getattr(road_network, "graph", {})
        candidates: list[object] = []
        if len(index) >= 2:
            try:
                outgoing = graph[index[1]]
            except (KeyError, TypeError):
                outgoing = {}
            if isinstance(outgoing, dict):
                for lane_group in outgoing.values():
                    if isinstance(lane_group, (list, tuple)):
                        candidates.extend(lane_group)

        lane_end = self._lane_position(lane, float(getattr(lane, "length", 0.0) or 0.0))
        lane_heading = self._lane_heading(lane, float(getattr(lane, "length", 0.0) or 0.0))
        source_lane_id = index[-1] if index and isinstance(index[-1], (int, np.integer)) else None

        def cost(candidate: object) -> float:
            start = self._lane_position(candidate, 0.0)
            heading = self._lane_heading(candidate, 0.0)
            candidate_index = tuple(getattr(candidate, "index", ()) or ())
            candidate_lane_id = candidate_index[-1] if candidate_index and isinstance(candidate_index[-1], (int, np.integer)) else None
            lane_id_cost = 0.0 if source_lane_id is None or candidate_lane_id is None else abs(int(candidate_lane_id) - int(source_lane_id))
            return float(np.linalg.norm(start - lane_end)) + 5.0 * abs(float(_wrap_to_pi(heading - lane_heading))) + lane_id_cost

        valid: list[object] = []
        for candidate in _unique_lanes(candidates):
            try:
                if cost(candidate) <= 12.0:
                    valid.append(candidate)
            except (DynamicAnchorError, TypeError, ValueError):
                continue
        return sorted(valid, key=cost)

    def _sample_lane_path(
        self,
        road_network: object,
        lane: object,
        start_s: float,
        distances: np.ndarray,
    ) -> np.ndarray:
        chain = [lane]
        starts = [float(np.clip(start_s, 0.0, max(float(getattr(lane, "length", 0.0) or 0.0), 0.0)))]
        covered = max(float(getattr(lane, "length", 0.0) or 0.0) - starts[0], 0.0)
        while covered < float(np.max(distances)) and len(chain) < 6:
            successors = self._successors(road_network, chain[-1])
            if not successors or successors[0] in chain:
                break
            chain.append(successors[0])
            starts.append(0.0)
            covered += max(float(getattr(chain[-1], "length", 0.0) or 0.0), 0.0)

        result = []
        for distance in np.asarray(distances, dtype=np.float64):
            remaining = max(float(distance), 0.0)
            point = None
            for current_lane, lane_start in zip(chain, starts):
                length = max(float(getattr(current_lane, "length", 0.0) or 0.0), 0.0)
                available = max(length - lane_start, 0.0)
                if remaining <= available + 1e-6:
                    point = self._lane_position(current_lane, min(lane_start + remaining, length))
                    break
                remaining -= available
            if point is None:
                last = chain[-1]
                length = max(float(getattr(last, "length", 0.0) or 0.0), 0.0)
                end = self._lane_position(last, length)
                heading = self._lane_heading(last, length)
                point = end + remaining * np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
            result.append(point)
        return np.asarray(result, dtype=np.float64)

    def _travel_distances(self, speed_mps: float, accel_mps2: float) -> np.ndarray:
        speed = max(float(speed_mps), 0.0)
        distance = 0.0
        values = []
        for _ in range(TRAJECTORY_STEPS):
            next_speed = float(np.clip(speed + accel_mps2 * self.config.dt_s, 0.0, self.config.max_speed_mps))
            distance += 0.5 * (speed + next_speed) * self.config.dt_s
            values.append(distance)
            speed = next_speed
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _freeze_stationary_tail(
        distances: np.ndarray,
        *,
        movement_epsilon_m: float,
    ) -> np.ndarray:
        """Make a stopped anchor exactly stationary after its last moving step."""

        values = np.asarray(distances, dtype=np.float64).copy()
        increments = np.diff(np.concatenate(([0.0], values)))
        stationary = np.flatnonzero(increments <= float(movement_epsilon_m))
        if stationary.size:
            first = int(stationary[0])
            frozen_distance = 0.0 if first == 0 else float(values[first - 1])
            values[first:] = frozen_distance
        return values

    @staticmethod
    def _headings_from_xy(xy: np.ndarray) -> np.ndarray:
        points = np.concatenate([np.zeros((1, 2), dtype=np.float64), xy], axis=0)
        delta = np.diff(points, axis=0)
        headings = np.zeros((xy.shape[0],), dtype=np.float64)
        previous = 0.0
        for index, segment in enumerate(delta):
            if float(np.linalg.norm(segment)) > 1e-5:
                previous = float(np.arctan2(segment[1], segment[0]))
            headings[index] = previous
        return _wrap_to_pi(headings)

    @staticmethod
    def _rate_limit_headings(
        headings: np.ndarray,
        *,
        max_delta_rad: float,
    ) -> np.ndarray:
        limited = np.asarray(headings, dtype=np.float64).copy()
        previous = 0.0
        for index in range(len(limited)):
            delta = float(_wrap_to_pi(limited[index] - previous))
            previous = float(
                _wrap_to_pi(
                    previous
                    + np.clip(delta, -max_delta_rad, max_delta_rad)
                )
            )
            limited[index] = previous
        return limited

    def _trajectory(
        self,
        road_network: object,
        source_lane: object,
        target_lane: object | None,
        source_s: float,
        target_s: float,
        ego_pose: np.ndarray,
        speed_mps: float,
        accel_mps2: float,
    ) -> np.ndarray:
        distances = self._travel_distances(speed_mps, accel_mps2)
        source_world = self._sample_lane_path(road_network, source_lane, source_s, distances)
        source_local = self._world_to_ego(source_world, ego_pose)
        if target_lane is None or target_lane is source_lane:
            xy = source_local
        else:
            target_world = self._sample_lane_path(road_network, target_lane, target_s, distances)
            target_local = self._world_to_ego(target_world, ego_pose)
            progress = np.linspace(1.0 / TRAJECTORY_STEPS, 1.0, TRAJECTORY_STEPS)
            blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            xy = source_local * (1.0 - blend[:, None]) + target_local * blend[:, None]
        heading = self._headings_from_xy(xy)
        return np.column_stack([xy, heading]).astype(np.float32)

    def _stop_trajectory(
        self,
        road_network: object,
        source_lane: object,
        source_s: float,
        ego_pose: np.ndarray,
        speed_mps: float,
    ) -> np.ndarray:
        if speed_mps <= 1.0e-9:
            return np.zeros(
                (TRAJECTORY_STEPS, TRAJECTORY_DIM), dtype=np.float32
            )
        contract = HardModeMaskConfig(dt_s=self.config.dt_s)
        distances = self._freeze_stationary_tail(
            self._travel_distances(
                speed_mps, self.config.stop_accel_mps2
            ),
            movement_epsilon_m=contract.movement_epsilon_m,
        )
        world = self._sample_lane_path(
            road_network, source_lane, source_s, distances
        )
        xy = self._world_to_ego(world, ego_pose)
        increments = np.diff(np.concatenate(([0.0], distances)))
        stationary = np.flatnonzero(
            increments <= contract.movement_epsilon_m
        )
        if stationary.size:
            first = int(stationary[0])
            frozen_xy = (
                np.zeros((2,), dtype=np.float64)
                if first == 0
                else xy[first - 1].copy()
            )
            xy[first:] = frozen_xy
        heading = self._headings_from_xy(xy)
        heading = self._rate_limit_headings(
            heading,
            max_delta_rad=(
                contract.max_yaw_rate_rad_s * contract.dt_s
            ),
        )
        if stationary.size:
            first = int(stationary[0])
            frozen_heading = 0.0 if first == 0 else float(heading[first - 1])
            heading[first:] = frozen_heading
        result = np.column_stack([xy, heading]).astype(np.float32)
        audit = validate_trajectory_kinematics(
            result,
            speed_mps,
            np.zeros((TRAJECTORY_DIM,), dtype=np.float64),
            contract,
        )
        if not audit.valid:
            raise DynamicAnchorError(
                "STOP anchor violates fixed-time kinematics: "
                + ",".join(audit.violations)
            )
        return result

    def generate(self, env: object, ego_id: str) -> DynamicAnchorOutput:
        agents = getattr(env, "agents", None)
        if not isinstance(agents, dict) or ego_id not in agents:
            raise DynamicAnchorError(f"active ego agent {ego_id!r} is required")
        vehicle = agents[ego_id]
        source_lane = getattr(vehicle, "lane", None)
        if source_lane is None:
            raise DynamicAnchorError(f"agent {ego_id!r} has no current lane")
        position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float64).reshape(-1)
        if position.size < 2:
            raise DynamicAnchorError(f"agent {ego_id!r} has no finite position")
        heading = float(getattr(vehicle, "heading_theta"))
        speed_mps = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6
        ego_pose = np.asarray([position[0], position[1], heading], dtype=np.float64)
        if not np.isfinite(ego_pose).all() or not np.isfinite(speed_mps) or speed_mps < 0.0:
            raise DynamicAnchorError(f"agent {ego_id!r} has invalid kinematics")

        road_network = self._road_network(env)
        source_s, _ = self._project(source_lane, position[:2])
        left_lane = self._lateral_lane(road_network, source_lane, ego_pose, direction=1)
        right_lane = self._lateral_lane(road_network, source_lane, ego_pose, direction=-1)
        left_s = self._project(left_lane, position[:2])[0] if left_lane is not None else 0.0
        right_s = self._project(right_lane, position[:2])[0] if right_lane is not None else 0.0

        accelerations = (
            self.config.high_accel_mps2,
            self.config.medium_accel_mps2,
            self.config.low_accel_mps2,
        )
        trajectories = np.zeros(TRAJECTORY_SHAPE, dtype=np.float32)
        for slot, accel in zip(
            (ModeIndex.KEEP_HIGH, ModeIndex.KEEP_MEDIUM, ModeIndex.KEEP_LOW), accelerations
        ):
            trajectories[slot] = self._trajectory(
                road_network, source_lane, None, source_s, 0.0, ego_pose, speed_mps, accel
            )
        if left_lane is not None:
            for slot, accel in zip(
                (ModeIndex.LEFT_HIGH, ModeIndex.LEFT_MEDIUM, ModeIndex.LEFT_LOW), accelerations
            ):
                trajectories[slot] = self._trajectory(
                    road_network, source_lane, left_lane, source_s, left_s, ego_pose, speed_mps, accel
                )
        if right_lane is not None:
            for slot, accel in zip(
                (ModeIndex.RIGHT_HIGH, ModeIndex.RIGHT_MEDIUM, ModeIndex.RIGHT_LOW), accelerations
            ):
                trajectories[slot] = self._trajectory(
                    road_network, source_lane, right_lane, source_s, right_s, ego_pose, speed_mps, accel
                )
        trajectories[ModeIndex.STOP] = self._stop_trajectory(
            road_network,
            source_lane,
            source_s,
            ego_pose,
            speed_mps,
        )
        return DynamicAnchorOutput(
            coarse_trajectories=np.ascontiguousarray(trajectories),
            topology=ModeTopology(
                left_reachable=left_lane is not None,
                right_reachable=right_lane is not None,
            ),
        )


__all__ = [
    "DynamicAnchorConfig",
    "DynamicAnchorError",
    "DynamicAnchorOutput",
    "SimulatorDynamicAnchorGenerator",
]
