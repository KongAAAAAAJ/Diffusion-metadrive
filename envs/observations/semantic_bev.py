"""Eight-channel simulator ground-truth BEV rasterization.

The rasterizer is deliberately independent from Panda3D and pygame.  It consumes
ground-truth lane polygons and actor footprints, producing an ego-centric uint8
tensor with shape ``[8, 256, 256]``.  In image coordinates, ego-forward points
up and ego-left points left.

The navigation channel contains only the task route exposed by MetaDrive's
navigation module.  RuleMaker decisions and locally selected manoeuvres are not
accepted by this interface, preventing expert-label leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


class BEVChannel(IntEnum):
    """Frozen channel order for the BEV-only planner."""

    DRIVABLE = 0
    LANE_GEOMETRY = 1
    NAVIGATION_ROUTE = 2
    EGO_HISTORY = 3
    PLATOON_VEHICLES = 4
    BACKGROUND_T0 = 5
    BACKGROUND_T_MINUS_0_5 = 6
    BACKGROUND_T_MINUS_1_0 = 7


BEV_CHANNEL_NAMES = (
    "drivable",
    "lane_geometry",
    "navigation_route",
    "ego_history",
    "platoon_vehicles",
    "background_t0",
    "background_t_minus_0_5",
    "background_t_minus_1_0",
)


@dataclass(frozen=True)
class SemanticBEVConfig:
    """Frozen metric and pixel contract for semantic BEV tensors.

    All empty pixels are 0.  Binary semantic foreground is 255.  Lane
    boundaries use 128 and lane centerlines use 255.  Ego footprints at
    t/t-0.5/t-1.0 use 255/170/85 so one uint8 channel retains temporal order.
    """

    height: int = 256
    width: int = 256
    x_min_m: float = -12.0
    x_max_m: float = 52.0
    y_min_m: float = -32.0
    y_max_m: float = 32.0
    lane_center_value: int = 255
    lane_boundary_value: int = 128
    foreground_value: int = 255
    ego_history_values: tuple[int, int, int] = (255, 170, 85)
    line_width_m: float = 0.25
    history_offsets_s: tuple[float, float, float] = (0.0, 0.5, 1.0)
    history_tolerance_s: float = 0.051

    def __post_init__(self) -> None:
        if (self.height, self.width) != (256, 256):
            raise ValueError("Semantic BEV resolution is frozen at 256x256.")
        if (self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m) != (-12.0, 52.0, -32.0, 32.0):
            raise ValueError("Semantic BEV metric range is frozen at x=[-12,52], y=[-32,32].")
        if self.history_offsets_s != (0.0, 0.5, 1.0):
            raise ValueError("Semantic BEV history offsets are frozen at (0.0, 0.5, 1.0) seconds.")
        if self.ego_history_values != (255, 170, 85):
            raise ValueError("Semantic BEV ego history encoding is frozen at (255, 170, 85).")
        if not 0.0 < self.history_tolerance_s <= 0.1:
            raise ValueError("history_tolerance_s must be in (0, 0.1].")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(BEVChannel), self.height, self.width)

    @property
    def pixels_per_meter(self) -> float:
        return (self.height - 1) / (self.x_max_m - self.x_min_m)


def _xy_array(value: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [N,2+], got {array.shape}")
    if not np.isfinite(array[:, :2]).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return np.ascontiguousarray(array[:, :2]).copy()


@dataclass(frozen=True)
class OrientedBoxState:
    """Ground-truth 2D actor footprint in world coordinates."""

    center_xy: np.ndarray
    heading_rad: float
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xy, dtype=np.float32).reshape(-1)
        if center.size < 2 or not np.isfinite(center[:2]).all():
            raise ValueError("center_xy must contain two finite values")
        if not np.isfinite(self.heading_rad):
            raise ValueError("heading_rad must be finite")
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("box length and width must be positive")
        # A simulator mutates vehicle.position in place.  Snapshots must own
        # their coordinates or all history channels collapse to the latest pose.
        object.__setattr__(self, "center_xy", np.ascontiguousarray(center[:2]).copy())
        object.__setattr__(self, "heading_rad", float(self.heading_rad))
        object.__setattr__(self, "length_m", float(self.length_m))
        object.__setattr__(self, "width_m", float(self.width_m))

    def corners_world(self) -> np.ndarray:
        half_length = self.length_m / 2.0
        half_width = self.width_m / 2.0
        local = np.asarray(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ],
            dtype=np.float32,
        )
        cos_h = float(np.cos(self.heading_rad))
        sin_h = float(np.sin(self.heading_rad))
        rotation = np.asarray([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        return local @ rotation.T + self.center_xy[None, :]


@dataclass(frozen=True)
class SimulatorSnapshot:
    """Actor-only ground-truth snapshot captured at one simulator time."""

    timestamp_s: float
    platoon: Mapping[str, OrientedBoxState]
    background: tuple[OrientedBoxState, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("snapshot timestamp must be finite")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "platoon", dict(self.platoon))
        object.__setattr__(self, "background", tuple(self.background))


@dataclass(frozen=True)
class SemanticBEVScene:
    """Renderer-ready scene with no learned or expert-decision fields."""

    ego_pose: np.ndarray
    drivable_polygons: tuple[np.ndarray, ...] = ()
    lane_centerlines: tuple[np.ndarray, ...] = ()
    lane_boundaries: tuple[np.ndarray, ...] = ()
    navigation_route_polygons: tuple[np.ndarray, ...] = ()
    ego_history: tuple[OrientedBoxState, ...] = ()
    platoon_vehicles: tuple[OrientedBoxState, ...] = ()
    background_history: tuple[tuple[OrientedBoxState, ...], ...] = field(
        default_factory=lambda: ((), (), ())
    )

    def __post_init__(self) -> None:
        pose = np.asarray(self.ego_pose, dtype=np.float32).reshape(-1)
        if pose.size != 3 or not np.isfinite(pose).all():
            raise ValueError("ego_pose must be finite [world_x, world_y, heading_rad]")
        object.__setattr__(self, "ego_pose", np.ascontiguousarray(pose))
        for field_name in (
            "drivable_polygons",
            "lane_centerlines",
            "lane_boundaries",
            "navigation_route_polygons",
        ):
            values = tuple(
                _xy_array(value, name=field_name)
                for value in getattr(self, field_name)
            )
            object.__setattr__(self, field_name, values)
        object.__setattr__(self, "ego_history", tuple(self.ego_history))
        object.__setattr__(self, "platoon_vehicles", tuple(self.platoon_vehicles))
        histories = tuple(tuple(boxes) for boxes in self.background_history)
        if len(histories) != 3:
            raise ValueError("background_history must contain t, t-0.5s and t-1.0s")
        object.__setattr__(self, "background_history", histories)


class SemanticBEVRasterizer:
    """Rasterize :class:`SemanticBEVScene` without a rendering engine."""

    def __init__(self, config: SemanticBEVConfig | None = None) -> None:
        self.config = config or SemanticBEVConfig()

    def world_to_ego(self, points_world: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
        points = _xy_array(points_world, name="points_world")
        pose = np.asarray(ego_pose, dtype=np.float32).reshape(3)
        delta = points - pose[None, :2]
        cos_h = float(np.cos(pose[2]))
        sin_h = float(np.sin(pose[2]))
        rotation_world_to_ego = np.asarray([[cos_h, sin_h], [-sin_h, cos_h]], dtype=np.float32)
        return delta @ rotation_world_to_ego.T

    def ego_to_pixel(self, points_ego: np.ndarray) -> np.ndarray:
        points = _xy_array(points_ego, name="points_ego")
        cfg = self.config
        col = (cfg.y_max_m - points[:, 1]) / (cfg.y_max_m - cfg.y_min_m) * (cfg.width - 1)
        row = (cfg.x_max_m - points[:, 0]) / (cfg.x_max_m - cfg.x_min_m) * (cfg.height - 1)
        return np.stack([col, row], axis=-1).astype(np.float32, copy=False)

    def world_to_pixel(self, points_world: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
        return self.ego_to_pixel(self.world_to_ego(points_world, ego_pose))

    @staticmethod
    def _opencv_points(points: np.ndarray) -> np.ndarray:
        rounded = np.rint(points).astype(np.int64)
        rounded = np.clip(rounded, -32768, 32767).astype(np.int32)
        return rounded.reshape((-1, 1, 2))

    def _fill_world_polygon(self, layer: np.ndarray, polygon: np.ndarray, ego_pose: np.ndarray, value: int) -> None:
        if polygon.shape[0] < 3:
            return
        pixels = self.world_to_pixel(polygon, ego_pose)
        cv2.fillPoly(layer, [self._opencv_points(pixels)], color=int(value), lineType=cv2.LINE_8)

    def _draw_world_line(self, layer: np.ndarray, line: np.ndarray, ego_pose: np.ndarray, value: int) -> None:
        if line.shape[0] < 2:
            return
        pixels = self.world_to_pixel(line, ego_pose)
        thickness = max(1, int(round(self.config.line_width_m * self.config.pixels_per_meter)))
        cv2.polylines(
            layer,
            [self._opencv_points(pixels)],
            isClosed=False,
            color=int(value),
            thickness=thickness,
            lineType=cv2.LINE_8,
        )

    def _fill_box(self, layer: np.ndarray, box: OrientedBoxState, ego_pose: np.ndarray, value: int) -> None:
        self._fill_world_polygon(layer, box.corners_world(), ego_pose, value)

    def rasterize(self, scene: SemanticBEVScene) -> np.ndarray:
        cfg = self.config
        bev = np.zeros(cfg.shape, dtype=np.uint8)

        for polygon in scene.drivable_polygons:
            self._fill_world_polygon(
                bev[BEVChannel.DRIVABLE], polygon, scene.ego_pose, cfg.foreground_value
            )

        for boundary in scene.lane_boundaries:
            self._draw_world_line(
                bev[BEVChannel.LANE_GEOMETRY],
                boundary,
                scene.ego_pose,
                cfg.lane_boundary_value,
            )
        for centerline in scene.lane_centerlines:
            self._draw_world_line(
                bev[BEVChannel.LANE_GEOMETRY],
                centerline,
                scene.ego_pose,
                cfg.lane_center_value,
            )

        for polygon in scene.navigation_route_polygons:
            self._fill_world_polygon(
                bev[BEVChannel.NAVIGATION_ROUTE], polygon, scene.ego_pose, cfg.foreground_value
            )

        for box, value in reversed(tuple(zip(scene.ego_history[:3], cfg.ego_history_values))):
            self._fill_box(bev[BEVChannel.EGO_HISTORY], box, scene.ego_pose, value)

        for box in scene.platoon_vehicles:
            self._fill_box(
                bev[BEVChannel.PLATOON_VEHICLES], box, scene.ego_pose, cfg.foreground_value
            )

        background_channels = (
            BEVChannel.BACKGROUND_T0,
            BEVChannel.BACKGROUND_T_MINUS_0_5,
            BEVChannel.BACKGROUND_T_MINUS_1_0,
        )
        for channel, boxes in zip(background_channels, scene.background_history):
            for box in boxes:
                self._fill_box(bev[channel], box, scene.ego_pose, cfg.foreground_value)

        return np.ascontiguousarray(bev)


def _read_dimension(obj: object, names: Sequence[str], fallback: float) -> float:
    for name in names:
        try:
            value = getattr(obj, name)
            if callable(value):
                value = value()
            value = float(value)
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value
    return float(fallback)


def _object_to_box(obj: object) -> OrientedBoxState | None:
    try:
        position = np.asarray(getattr(obj, "position"), dtype=np.float32).reshape(-1)
        heading = float(getattr(obj, "heading_theta"))
    except (AttributeError, TypeError, ValueError):
        return None
    if position.size < 2 or not np.isfinite(position[:2]).all() or not np.isfinite(heading):
        return None
    length = _read_dimension(obj, ("top_down_length", "LENGTH", "length"), 4.5)
    width = _read_dimension(obj, ("top_down_width", "WIDTH", "width"), 1.8)
    return OrientedBoxState(position[:2], heading, length, width)


def _deduplicate_objects(objects: Iterable[object]) -> list[object]:
    seen: set[int] = set()
    result = []
    for obj in objects:
        marker = id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(obj)
    return result


class MetaDriveSceneAdapter:
    """Extract semantic inputs directly from live MetaDrive ground truth."""

    def __init__(self, config: SemanticBEVConfig | None = None, lane_sample_interval_m: float = 1.0) -> None:
        self.config = config or SemanticBEVConfig()
        self.lane_sample_interval_m = float(lane_sample_interval_m)
        if self.lane_sample_interval_m <= 0.0:
            raise ValueError("lane_sample_interval_m must be positive")
        self._cached_network_id: int | None = None
        self._cached_drivable: tuple[np.ndarray, ...] = ()
        self._cached_centers: tuple[np.ndarray, ...] = ()
        self._cached_boundaries: tuple[np.ndarray, ...] = ()

    @staticmethod
    def _road_network(env: object) -> object:
        current_map = getattr(env, "current_map", None)
        if current_map is None:
            current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if current_map is None:
            current_map = getattr(getattr(getattr(env, "engine", None), "map_manager", None), "current_map", None)
        network = getattr(current_map, "road_network", None)
        if network is None:
            raise RuntimeError("MetaDrive environment has no active road network")
        return network

    @staticmethod
    def _agents(env: object) -> Mapping[str, object]:
        agents = getattr(env, "agents", None)
        if agents is None:
            agents = getattr(getattr(env, "engine", None), "agents", None)
        if not isinstance(agents, Mapping):
            raise RuntimeError("MetaDrive environment does not expose an agent mapping")
        return agents

    def capture_snapshot(self, env: object, timestamp_s: float) -> SimulatorSnapshot:
        agents = self._agents(env)
        platoon = {}
        agent_markers = set()
        for agent_id, vehicle in agents.items():
            box = _object_to_box(vehicle)
            if box is None:
                raise RuntimeError(f"Unable to read ground-truth footprint for platoon agent {agent_id!r}")
            platoon[str(agent_id)] = box
            agent_markers.add(id(vehicle))

        engine = getattr(env, "engine", None)
        get_objects = getattr(engine, "get_objects", None)
        objects = get_objects() if callable(get_objects) else {}
        object_values = objects.values() if isinstance(objects, Mapping) else objects
        background = []
        for obj in _deduplicate_objects(object_values or ()):
            if id(obj) in agent_markers:
                continue
            # MetaDrive's object registry also contains signs and static props.
            # Dynamic vehicles expose speed in addition to a pose and footprint.
            if not hasattr(obj, "speed") and not hasattr(obj, "speed_km_h"):
                continue
            box = _object_to_box(obj)
            if box is not None:
                background.append(box)
        return SimulatorSnapshot(timestamp_s=timestamp_s, platoon=platoon, background=tuple(background))

    def _map_geometry(self, road_network: object) -> tuple[tuple[np.ndarray, ...], ...]:
        network_id = id(road_network)
        if self._cached_network_id == network_id:
            return self._cached_drivable, self._cached_centers, self._cached_boundaries

        get_all_lanes = getattr(road_network, "get_all_lanes", None)
        if not callable(get_all_lanes):
            raise RuntimeError("Road network does not provide get_all_lanes()")
        drivable = []
        centers = []
        boundaries = []
        for lane in _deduplicate_objects(get_all_lanes()):
            try:
                polygon = _xy_array(getattr(lane, "polygon"), name="lane.polygon")
                length = float(getattr(lane, "length"))
            except (AttributeError, TypeError, ValueError):
                continue
            if polygon.shape[0] >= 3:
                drivable.append(polygon)
            sample_count = max(2, int(np.ceil(max(length, 0.0) / self.lane_sample_interval_m)) + 1)
            longitudinal = np.linspace(0.0, max(length, 0.0), sample_count)
            center = []
            left = []
            right = []
            try:
                for distance in longitudinal:
                    width = float(lane.width_at(float(distance)))
                    center.append(lane.position(float(distance), 0.0))
                    left.append(lane.position(float(distance), width / 2.0))
                    right.append(lane.position(float(distance), -width / 2.0))
            except (AttributeError, TypeError, ValueError):
                continue
            centers.append(_xy_array(center, name="lane.centerline"))
            boundaries.append(_xy_array(left, name="lane.left_boundary"))
            boundaries.append(_xy_array(right, name="lane.right_boundary"))

        self._cached_network_id = network_id
        self._cached_drivable = tuple(drivable)
        self._cached_centers = tuple(centers)
        self._cached_boundaries = tuple(boundaries)
        return self._cached_drivable, self._cached_centers, self._cached_boundaries

    @staticmethod
    def _route_lanes(navigation: object, road_network: object) -> list[object]:
        route_lanes: list[object] = []
        checkpoints = list(getattr(navigation, "checkpoints", None) or ())
        target_indices = list(getattr(navigation, "_target_checkpoints_index", None) or (0,))
        start_index = int(target_indices[0]) if target_indices else 0
        graph = getattr(road_network, "graph", {})

        if len(checkpoints) >= 2:
            # NodeRoadNetwork: checkpoints are nodes and graph[a][b] is a lane list.
            for index in range(max(0, start_index), len(checkpoints) - 1):
                first, second = checkpoints[index], checkpoints[index + 1]
                try:
                    lanes = graph[first][second]
                except (KeyError, TypeError):
                    lanes = None
                if isinstance(lanes, (list, tuple)):
                    route_lanes.extend(lanes)

            # EdgeRoadNetwork: checkpoints themselves are lane indices.
            get_peer_lanes = getattr(road_network, "get_peer_lanes_from_index", None)
            if callable(get_peer_lanes):
                for checkpoint in checkpoints[max(0, start_index):]:
                    try:
                        route_lanes.extend(get_peer_lanes(checkpoint))
                    except (KeyError, TypeError):
                        continue

        if not route_lanes:
            route_lanes.extend(getattr(navigation, "current_ref_lanes", None) or ())
            route_lanes.extend(getattr(navigation, "next_ref_lanes", None) or ())
        return _deduplicate_objects(route_lanes)

    def _snapshot_for_offset(
        self,
        snapshots: Sequence[SimulatorSnapshot], current_time_s: float, offset_s: float
    ) -> SimulatorSnapshot:
        past = [snapshot for snapshot in snapshots if snapshot.timestamp_s <= current_time_s + 1e-6]
        if not past:
            raise ValueError("At least one current or past simulator snapshot is required")
        target = current_time_s - offset_s
        selected = min(past, key=lambda snapshot: abs(snapshot.timestamp_s - target))
        error = abs(selected.timestamp_s - target)
        if error > self.config.history_tolerance_s:
            raise ValueError(
                f"No simulator snapshot within {self.config.history_tolerance_s:.3f}s "
                f"of required history time {target:.3f}s"
            )
        return selected

    def build_scene(
        self,
        env: object,
        ego_id: str,
        snapshots: Sequence[SimulatorSnapshot],
    ) -> SemanticBEVScene:
        if not snapshots:
            raise ValueError("snapshots must contain the current simulator state")
        current_time = max(snapshot.timestamp_s for snapshot in snapshots)
        selected = tuple(
            self._snapshot_for_offset(snapshots, current_time, offset)
            for offset in self.config.history_offsets_s
        )
        current = selected[0]
        if ego_id not in current.platoon:
            raise KeyError(f"Ego agent {ego_id!r} is missing from the current snapshot")
        ego_box = current.platoon[ego_id]
        ego_pose = np.asarray([*ego_box.center_xy, ego_box.heading_rad], dtype=np.float32)

        road_network = self._road_network(env)
        drivable, centers, boundaries = self._map_geometry(road_network)
        agents = self._agents(env)
        ego_vehicle = agents.get(ego_id)
        navigation = getattr(ego_vehicle, "navigation", None)
        route_polygons = []
        if navigation is not None:
            for lane in self._route_lanes(navigation, road_network):
                try:
                    polygon = _xy_array(getattr(lane, "polygon"), name="route_lane.polygon")
                except (AttributeError, TypeError, ValueError):
                    continue
                if polygon.shape[0] >= 3:
                    route_polygons.append(polygon)

        ego_history = tuple(snapshot.platoon[ego_id] for snapshot in selected if ego_id in snapshot.platoon)
        platoon_vehicles = tuple(box for agent_id, box in current.platoon.items() if agent_id != ego_id)
        background_history = tuple(snapshot.background for snapshot in selected)
        return SemanticBEVScene(
            ego_pose=ego_pose,
            drivable_polygons=drivable,
            lane_centerlines=centers,
            lane_boundaries=boundaries,
            navigation_route_polygons=tuple(route_polygons),
            ego_history=ego_history,
            platoon_vehicles=platoon_vehicles,
            background_history=background_history,
        )

    def rasterize(
        self,
        env: object,
        ego_id: str,
        snapshots: Sequence[SimulatorSnapshot],
    ) -> np.ndarray:
        scene = self.build_scene(env, ego_id, snapshots)
        return SemanticBEVRasterizer(self.config).rasterize(scene)


def semantic_bev_to_rgb(bev: np.ndarray) -> np.ndarray:
    """Create a deterministic RGB debug view from an eight-channel BEV tensor."""

    array = np.asarray(bev)
    expected = (len(BEVChannel), 256, 256)
    if array.shape != expected:
        raise ValueError(f"bev must have shape {expected}, got {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"bev must be uint8, got {array.dtype}")

    rgb = np.zeros((256, 256, 3), dtype=np.float32)

    def blend(channel: BEVChannel, color: tuple[int, int, int], alpha: float) -> None:
        mask = array[channel].astype(np.float32) / 255.0
        weight = (mask * alpha)[..., None]
        color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        rgb[:] = rgb * (1.0 - weight) + color_array * weight

    blend(BEVChannel.DRIVABLE, (70, 70, 70), 0.85)
    blend(BEVChannel.NAVIGATION_ROUTE, (40, 190, 80), 0.55)
    blend(BEVChannel.LANE_GEOMETRY, (245, 245, 245), 0.9)
    blend(BEVChannel.BACKGROUND_T_MINUS_1_0, (135, 85, 35), 0.45)
    blend(BEVChannel.BACKGROUND_T_MINUS_0_5, (220, 145, 35), 0.6)
    blend(BEVChannel.BACKGROUND_T0, (255, 205, 55), 0.9)
    blend(BEVChannel.PLATOON_VEHICLES, (45, 125, 255), 0.95)
    blend(BEVChannel.EGO_HISTORY, (235, 55, 65), 0.95)
    return np.ascontiguousarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


def save_semantic_bev_png(path: str | Path, bev: np.ndarray) -> Path:
    """Save an RGB debug view.  Dataset collection must store channels, not this image."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = semantic_bev_to_rgb(bev)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write semantic BEV visualization: {output_path}")
    return output_path
