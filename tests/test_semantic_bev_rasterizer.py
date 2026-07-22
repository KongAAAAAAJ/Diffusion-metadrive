from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from envs.observations.semantic_bev import (
    BEV_CHANNEL_NAMES,
    BEVChannel,
    MetaDriveSceneAdapter,
    OrientedBoxState,
    SemanticBEVConfig,
    SemanticBEVRasterizer,
    SemanticBEVScene,
    SimulatorSnapshot,
    save_semantic_bev_png,
)


def _rectangle(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    return np.asarray([[x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=np.float32)


def _box(x: float, y: float, heading: float = 0.0, length: float = 4.5, width: float = 1.8):
    return OrientedBoxState(np.asarray([x, y], dtype=np.float32), heading, length, width)


def _pixel(rasterizer: SemanticBEVRasterizer, x: float, y: float) -> tuple[int, int]:
    col, row = rasterizer.ego_to_pixel(np.asarray([[x, y]], dtype=np.float32))[0]
    return int(round(row)), int(round(col))


def _straight_scene() -> SemanticBEVScene:
    return SemanticBEVScene(
        ego_pose=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        drivable_polygons=(_rectangle(-20.0, 65.0, -7.0, 7.0),),
        lane_centerlines=(np.asarray([[-20.0, 0.0], [65.0, 0.0]], dtype=np.float32),),
        lane_boundaries=(
            np.asarray([[-20.0, -3.5], [65.0, -3.5]], dtype=np.float32),
            np.asarray([[-20.0, 3.5], [65.0, 3.5]], dtype=np.float32),
        ),
        navigation_route_polygons=(_rectangle(-20.0, 65.0, -3.5, 3.5),),
        ego_history=(_box(0.0, 0.0), _box(-4.0, 0.0), _box(-8.0, 0.0)),
        platoon_vehicles=(_box(10.0, 0.0), _box(-8.0, 0.0)),
        background_history=(
            (_box(22.0, -4.5),),
            (_box(20.0, -4.5),),
            (_box(18.0, -4.5),),
        ),
    )


def _lane_change_scene() -> SemanticBEVScene:
    scene = _straight_scene()
    route = np.asarray(
        [
            [-10.0, -1.75],
            [10.0, -1.75],
            [35.0, 1.75],
            [60.0, 1.75],
            [60.0, 5.25],
            [35.0, 5.25],
            [10.0, 1.75],
            [-10.0, 1.75],
        ],
        dtype=np.float32,
    )
    return replace(scene, navigation_route_polygons=(route,))


def _merge_scene() -> SemanticBEVScene:
    main = _rectangle(-20.0, 65.0, -3.5, 3.5)
    ramp = np.asarray(
        [[-10.0, -14.0], [35.0, -3.5], [45.0, -3.5], [-8.0, -18.0]],
        dtype=np.float32,
    )
    route = np.asarray(
        [[-8.0, -17.0], [36.0, -6.0], [55.0, -2.5], [55.0, 0.5], [35.0, -2.5], [-10.0, -13.0]],
        dtype=np.float32,
    )
    return SemanticBEVScene(
        ego_pose=np.asarray([0.0, -14.5, 0.22], dtype=np.float32),
        drivable_polygons=(main, ramp),
        lane_centerlines=(
            np.asarray([[-20.0, 0.0], [65.0, 0.0]], dtype=np.float32),
            np.asarray([[-10.0, -16.0], [40.0, -3.5]], dtype=np.float32),
        ),
        lane_boundaries=(),
        navigation_route_polygons=(route,),
        ego_history=(_box(0.0, -14.5, 0.22),),
        platoon_vehicles=(_box(-7.0, -16.2, 0.22),),
        background_history=((_box(25.0, 0.0),), (), ()),
    )


def test_semantic_bev_contract_is_frozen() -> None:
    config = SemanticBEVConfig()
    assert config.shape == (8, 256, 256)
    assert [channel.value for channel in BEVChannel] == list(range(8))
    assert BEV_CHANNEL_NAMES == (
        "drivable",
        "lane_geometry",
        "navigation_route",
        "ego_history",
        "platoon_vehicles",
        "background_t0",
        "background_t_minus_0_5",
        "background_t_minus_1_0",
    )
    assert config.foreground_value == 255
    assert config.ego_history_values == (255, 170, 85)
    with pytest.raises(ValueError, match="256x256"):
        SemanticBEVConfig(height=128)
    with pytest.raises(ValueError, match="metric range"):
        SemanticBEVConfig(x_max_m=48.0)


def test_world_to_pixel_has_forward_up_and_left_left() -> None:
    rasterizer = SemanticBEVRasterizer()
    ego_pixel = rasterizer.ego_to_pixel(np.asarray([[0.0, 0.0]], dtype=np.float32))[0]
    forward_pixel = rasterizer.ego_to_pixel(np.asarray([[10.0, 0.0]], dtype=np.float32))[0]
    left_pixel = rasterizer.ego_to_pixel(np.asarray([[0.0, 10.0]], dtype=np.float32))[0]
    assert forward_pixel[1] < ego_pixel[1]
    assert left_pixel[0] < ego_pixel[0]

    world_forward = rasterizer.world_to_ego(
        np.asarray([[0.0, 10.0]], dtype=np.float32),
        np.asarray([0.0, 0.0, np.pi / 2], dtype=np.float32),
    )
    np.testing.assert_allclose(world_forward, [[10.0, 0.0]], atol=1e-5)


def test_history_requires_snapshots_at_frozen_offsets() -> None:
    adapter = MetaDriveSceneAdapter()
    snapshot = SimulatorSnapshot(timestamp_s=1.0, platoon={"agent0": _box(0.0, 0.0)})
    with pytest.raises(ValueError, match="required history time 0.000s"):
        adapter._snapshot_for_offset([snapshot], current_time_s=1.0, offset_s=1.0)


def test_straight_scene_populates_all_semantic_groups() -> None:
    rasterizer = SemanticBEVRasterizer()
    bev = rasterizer.rasterize(_straight_scene())
    assert bev.shape == (8, 256, 256)
    assert bev.dtype == np.uint8
    assert bev.flags.c_contiguous

    road_px = _pixel(rasterizer, 30.0, 2.0)
    lane_px = _pixel(rasterizer, 30.0, 0.0)
    boundary_px = _pixel(rasterizer, 30.0, 3.5)
    ego_px = _pixel(rasterizer, 0.0, 0.0)
    ego_half_second_px = _pixel(rasterizer, -4.0, 0.0)
    ego_one_second_px = _pixel(rasterizer, -8.0, 0.0)
    platoon_px = _pixel(rasterizer, 10.0, 0.0)
    bg_now_px = _pixel(rasterizer, 22.0, -4.5)
    bg_old_px = _pixel(rasterizer, 18.0, -4.5)
    assert bev[BEVChannel.DRIVABLE][road_px] == 255
    assert bev[BEVChannel.LANE_GEOMETRY][lane_px] == 255
    assert bev[BEVChannel.LANE_GEOMETRY][boundary_px] == 128
    assert bev[BEVChannel.NAVIGATION_ROUTE][road_px] == 255
    assert bev[BEVChannel.EGO_HISTORY][ego_px] == 255
    assert bev[BEVChannel.EGO_HISTORY][ego_half_second_px] == 170
    assert bev[BEVChannel.EGO_HISTORY][ego_one_second_px] == 85
    assert bev[BEVChannel.PLATOON_VEHICLES][platoon_px] == 255
    assert bev[BEVChannel.BACKGROUND_T0][bg_now_px] == 255
    assert bev[BEVChannel.BACKGROUND_T_MINUS_1_0][bg_old_px] == 255


@pytest.mark.parametrize(
    ("name", "scene"),
    (
        ("straight", _straight_scene()),
        ("lane_change", _lane_change_scene()),
        ("merge", _merge_scene()),
    ),
)
def test_debug_visualizations_for_required_scene_types(
    tmp_path: Path, name: str, scene: SemanticBEVScene
) -> None:
    bev = SemanticBEVRasterizer().rasterize(scene)
    output = save_semantic_bev_png(tmp_path / f"{name}.png", bev)
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (256, 256, 3)
    assert int(image.max()) > 0


class _FakeLane:
    def __init__(self, y: float, index: tuple[str, str, int]) -> None:
        self.y = float(y)
        self.index = index
        self.length = 80.0
        self._polygon = _rectangle(-20.0, 80.0, self.y - 1.75, self.y + 1.75)

    @property
    def polygon(self) -> np.ndarray:
        return self._polygon

    def width_at(self, longitudinal: float) -> float:
        return 3.5

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return np.asarray([longitudinal - 20.0, self.y + lateral], dtype=np.float32)


class _FakeRoadNetwork:
    def __init__(self, lanes: list[_FakeLane]) -> None:
        self._lanes = lanes
        self.graph = {"A": {"B": lanes}}

    def get_all_lanes(self) -> list[_FakeLane]:
        return self._lanes


def _fake_vehicle(name: str, x: float, y: float, navigation=None):
    return SimpleNamespace(
        name=name,
        position=np.asarray([x, y], dtype=np.float32),
        heading_theta=0.0,
        top_down_length=4.5,
        top_down_width=1.8,
        speed_km_h=30.0,
        navigation=navigation,
    )


def test_metadrive_adapter_uses_navigation_not_rule_maker_output() -> None:
    route_lane = _FakeLane(0.0, ("A", "B", 0))
    other_lane = _FakeLane(3.5, ("A", "B", 1))
    network = _FakeRoadNetwork([route_lane, other_lane])
    navigation = SimpleNamespace(
        checkpoints=["A", "B"],
        _target_checkpoints_index=[0, 0],
        current_ref_lanes=[route_lane, other_lane],
        next_ref_lanes=None,
    )
    leader = _fake_vehicle("agent0", 0.0, 0.0, navigation=navigation)
    follower = _fake_vehicle("agent1", -8.0, 0.0, navigation=navigation)
    rear = _fake_vehicle("agent2", -16.0, 0.0, navigation=navigation)
    background = _fake_vehicle("traffic", 20.0, 3.5)
    objects = {vehicle.name: vehicle for vehicle in (leader, follower, rear, background)}
    env = SimpleNamespace(
        current_map=SimpleNamespace(road_network=network),
        agents={"agent0": leader, "agent1": follower, "agent2": rear},
        engine=SimpleNamespace(get_objects=lambda: objects),
        rule_maker_selected_trajectory=np.full((8, 3), 999.0, dtype=np.float32),
    )

    adapter = MetaDriveSceneAdapter()
    snapshots = []
    for timestamp, leader_x, traffic_x in ((0.0, -8.0, 16.0), (0.5, -4.0, 18.0), (1.0, 0.0, 20.0)):
        leader.position[0] = leader_x
        background.position[0] = traffic_x
        snapshots.append(adapter.capture_snapshot(env, timestamp))

    bev_before = adapter.rasterize(env, "agent0", snapshots)
    env.rule_maker_selected_trajectory[:] = -999.0
    bev_after = adapter.rasterize(env, "agent0", snapshots)
    np.testing.assert_array_equal(bev_before, bev_after)

    assert bev_before[BEVChannel.NAVIGATION_ROUTE].max() == 255
    assert bev_before[BEVChannel.PLATOON_VEHICLES].max() == 255
    assert bev_before[BEVChannel.BACKGROUND_T0].max() == 255
    assert bev_before[BEVChannel.BACKGROUND_T_MINUS_0_5].max() == 255
    assert bev_before[BEVChannel.BACKGROUND_T_MINUS_1_0].max() == 255

    current_px = _pixel(SemanticBEVRasterizer(), 20.0, 3.5)
    oldest_px = _pixel(SemanticBEVRasterizer(), 16.0, 3.5)
    assert bev_before[BEVChannel.BACKGROUND_T0][current_px] == 255
    assert bev_before[BEVChannel.BACKGROUND_T_MINUS_1_0][oldest_px] == 255
