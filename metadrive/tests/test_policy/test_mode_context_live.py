from __future__ import annotations

import numpy as np

from models.diffusion.mode_context import build_mode_context_from_vehicle


class _FakeLane:
    def __init__(
        self,
        lane_id: int,
        lateral_offset: float,
        length: float = 60.0,
        width: float = 4.0,
        start_x: float = 0.0,
    ):
        self.index = ("A", "B", lane_id)
        self._lateral_offset = float(lateral_offset)
        self.length = float(length)
        self.width = float(width)
        self._start_x = float(start_x)

    def position(self, longitudinal, lateral):
        return np.asarray([self._start_x + float(longitudinal), self._lateral_offset + float(lateral)], dtype=np.float32)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def heading_theta_at(self, longitudinal: float) -> float:
        return 0.0

    def local_coordinates(self, position):
        return float(position[0] - self._start_x), float(position[1] - self._lateral_offset)


class _FakeNavigation:
    def __init__(self, lane, current_ref_lanes=None, next_ref_lanes=None):
        self.current_ref_lanes = list(current_ref_lanes or [lane])
        self.next_ref_lanes = list(next_ref_lanes) if next_ref_lanes is not None else None

    def get_current_lane_width(self):
        return 4.0


class _FakeRoad:
    def __init__(self, start: str, end: str):
        self.start_node = start
        self.end_node = end


class _FakeSocket:
    def __init__(self, road):
        self.positive_road = road


class _FakeBlock:
    def __init__(self, graph_block_id: str, sockets):
        self.graph_block_id = graph_block_id
        self._sockets = {idx: socket for idx, socket in enumerate(sockets)}

    def get_socket_list(self):
        return list(self._sockets.values())


class _FakeMap:
    def __init__(self, graph, blocks=None):
        self.road_network = type("RoadNetwork", (), {"graph": graph})()
        self.blocks = list(blocks or [])


class _FakeVehicle:
    def __init__(self, lane, current_map, current_ref_lanes=None, next_ref_lanes=None):
        self.position = np.asarray(lane.position(0.0, 0.0), dtype=np.float32)
        self.heading_theta = 0.0
        self.speed_km_h = 36.0
        self.navigation = _FakeNavigation(lane, current_ref_lanes=current_ref_lanes, next_ref_lanes=next_ref_lanes)
        self.lane = lane
        self.lane_index = lane.index
        self.lidar = None
        self.engine = type("Engine", (), {"current_map": current_map})()


def test_build_mode_context_from_vehicle_extracts_adjacent_lanes():
    lanes = [_FakeLane(0, 4.0), _FakeLane(1, 0.0), _FakeLane(2, -4.0)]
    graph = {"A": {"B": lanes}}
    current_map = _FakeMap(graph)
    vehicle = _FakeVehicle(lanes[1], current_map)

    ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)

    assert ctx.left_lane_polyline is not None
    assert ctx.right_lane_polyline is not None
    assert ctx.has_left_adjacent is True
    assert ctx.has_right_adjacent is True
    assert float(ctx.left_lane_polyline[-1, 1]) > 0.0
    assert float(ctx.right_lane_polyline[-1, 1]) < 0.0
    assert np.allclose(ctx.current_lane_polyline[0], np.zeros((2,), dtype=np.float32), atol=1e-4)
    assert np.all(np.diff(ctx.current_lane_polyline[:, 0]) >= -1e-4)
    assert np.allclose(ctx.current_lane_polyline[:, 1], 0.0, atol=1e-4)


def test_build_mode_context_from_vehicle_returns_none_for_missing_branches():
    lanes = [_FakeLane(0, 0.0)]
    graph = {"A": {"B": lanes}}
    current_map = _FakeMap(graph)
    vehicle = _FakeVehicle(lanes[0], current_map)

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        scenario_id="S1_free_cruise_straight",
        local_route="R1_entry_straight",
        ego_main_route_block_ids=("s0",),
    )

    assert ctx.left_branch_polyline is None
    assert ctx.right_branch_polyline is None
    assert ctx.has_left_branch is False
    assert ctx.has_right_branch is False


def test_build_mode_context_from_vehicle_uses_route_blocks_to_resolve_branch_polyline():
    current_lane = _FakeLane(0, 0.0)
    branch_lane = _FakeLane(0, -6.0)
    graph = {
        "A": {"B": [current_lane]},
        "R": {"S": [branch_lane]},
    }
    branch_block = _FakeBlock("s_ramp0", [_FakeSocket(_FakeRoad("R", "S"))])
    current_map = _FakeMap(graph, blocks=[branch_block])
    vehicle = _FakeVehicle(current_lane, current_map)

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        scenario_id="S8_ego_exit_to_ramp",
        local_route="R6_exit_to_ramp",
        ego_main_route_block_ids=("g0", "s_ramp0", "c0_ramp0"),
    )

    assert ctx.right_branch_polyline is not None
    assert ctx.has_right_branch is True


def test_build_mode_context_from_vehicle_extends_route_into_next_road_for_s8_exit_like_case():
    current_lane = _FakeLane(1, 0.0, length=8.0)
    next_lane = _FakeLane(1, -1.5, length=40.0, start_x=8.0)
    graph = {
        "A": {"B": [current_lane]},
        "B": {"C": [next_lane]},
    }
    current_map = _FakeMap(graph)
    vehicle = _FakeVehicle(current_lane, current_map, next_ref_lanes=[next_lane])
    vehicle.position = np.asarray(current_lane.position(6.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)

    assert float(ctx.current_lane_polyline[-1, 1]) < -1.0


def test_build_mode_context_from_vehicle_does_not_jump_to_next_road_before_current_horizon_is_exhausted():
    current_lane = _FakeLane(1, 0.0, length=120.0)
    next_lane = _FakeLane(1, -20.0, length=60.0, start_x=120.0)
    graph = {
        "A": {"B": [current_lane]},
        "B": {"C": [next_lane]},
    }
    current_map = _FakeMap(graph)
    vehicle = _FakeVehicle(current_lane, current_map, next_ref_lanes=[next_lane])
    vehicle.position = np.asarray(current_lane.position(0.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)

    assert np.all(np.diff(ctx.current_lane_polyline[:, 0]) >= -1e-4)
    assert np.allclose(ctx.current_lane_polyline[:, 1], 0.0, atol=1e-4)
    assert float(ctx.current_lane_polyline[-1, 0]) <= 58.0 + 1e-4


def test_build_mode_context_from_vehicle_connects_into_next_road_without_large_gap():
    current_lane = _FakeLane(1, 0.0, length=10.0)
    next_lane = _FakeLane(1, 0.0, length=60.0, start_x=10.0)
    graph = {
        "A": {"B": [current_lane]},
        "B": {"C": [next_lane]},
    }
    current_map = _FakeMap(graph)
    vehicle = _FakeVehicle(current_lane, current_map, next_ref_lanes=[next_lane])
    vehicle.position = np.asarray(current_lane.position(6.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)

    deltas = np.linalg.norm(np.diff(ctx.current_lane_polyline, axis=0), axis=1)
    assert float(np.max(deltas)) <= 2.01
    assert np.all(np.diff(ctx.current_lane_polyline[:, 0]) >= -1e-4)


def test_build_mode_context_from_vehicle_extends_terminal_local_route_into_map_successor():
    current_lane = _FakeLane(1, 0.0, length=10.0)
    next_lane = _FakeLane(1, 0.0, length=40.0, start_x=10.0)
    graph = {
        "A": {"B": [current_lane]},
        "B": {"C": [next_lane]},
    }
    final_block = _FakeBlock("c1", [_FakeSocket(_FakeRoad("A", "B"))])
    current_map = _FakeMap(graph, blocks=[final_block])
    vehicle = _FakeVehicle(current_lane, current_map, next_ref_lanes=None)
    vehicle.position = np.asarray(current_lane.position(6.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        local_route="R2_entry_curve",
        ego_main_route_block_ids=("c1",),
    )

    assert float(ctx.current_lane_polyline[-1, 0]) > 20.0
    assert np.allclose(ctx.current_lane_polyline[:, 1], 0.0, atol=1e-4)


def test_build_mode_context_from_vehicle_prefers_same_lane_id_for_next_segment():
    current_lanes = [_FakeLane(0, 4.0, length=10.0), _FakeLane(1, 0.0, length=10.0), _FakeLane(2, -4.0, length=10.0)]
    next_lanes = [
        _FakeLane(0, 4.0, length=40.0, start_x=10.0),
        _FakeLane(1, 0.0, length=40.0, start_x=10.0),
        _FakeLane(2, -4.0, length=40.0, start_x=10.0),
    ]
    graph = {
        "A": {"B": current_lanes},
        "B": {"C": next_lanes},
    }
    current_map = _FakeMap(graph)
    vehicle = _FakeVehicle(current_lanes[1], current_map, next_ref_lanes=next_lanes)
    vehicle.position = np.asarray(current_lanes[1].position(6.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)

    assert ctx.left_lane_polyline is not None
    assert ctx.right_lane_polyline is not None
    assert float(ctx.left_lane_polyline[-1, 1]) > 3.0
    assert float(ctx.right_lane_polyline[-1, 1]) < -3.0


def test_build_mode_context_from_vehicle_prefers_physical_lane_when_it_is_in_current_ref_lanes():
    physical_lane = _FakeLane(2, -4.0)
    reference_lane = _FakeLane(1, 0.0)
    graph = {"A": {"B": [physical_lane]}}
    current_map = _FakeMap(graph)
    vehicle = _FakeVehicle(physical_lane, current_map, current_ref_lanes=[reference_lane])

    ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)

    assert np.allclose(ctx.current_lane_polyline[0], np.zeros((2,), dtype=np.float32), atol=1e-4)
    assert np.allclose(ctx.current_lane_polyline[:, 1], 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests for _get_all_positive_roads_from_block and G-block deacc lane fix
# ---------------------------------------------------------------------------

class _FakeLaneXY:
    """Fake lane with explicit position support for G-block simulation tests."""

    def __init__(self, start_node, end_node, lane_id, start_xy, end_xy, width=4.0):
        self.index = (start_node, end_node, lane_id)
        self._start_xy = np.asarray(start_xy, dtype=np.float64)
        self._end_xy = np.asarray(end_xy, dtype=np.float64)
        diff = self._end_xy - self._start_xy
        self.length = float(np.linalg.norm(diff))
        self.width = float(width)
        self._direction = diff / max(self.length, 1e-9)

    def position(self, longitudinal, lateral):
        pt = self._start_xy + self._direction * float(longitudinal)
        normal = np.asarray([-self._direction[1], self._direction[0]], dtype=np.float64)
        pt = pt + normal * float(lateral)
        return np.asarray(pt, dtype=np.float32)

    def heading_theta_at(self, longitudinal):
        import math
        return math.atan2(float(self._direction[1]), float(self._direction[0]))

    def local_coordinates(self, position):
        pos = np.asarray(position[:2], dtype=np.float64)
        delta = pos - self._start_xy
        long = float(np.dot(delta, self._direction))
        normal = np.asarray([-self._direction[1], self._direction[0]], dtype=np.float64)
        lat = float(np.dot(delta, normal))
        return long, lat

    def width_at(self, longitudinal):
        return self.width


class _FakeBlockNetwork:
    """Fake block_network with get_positive_lanes support."""

    def __init__(self, lane_groups):
        # lane_groups: list of lists of _FakeLaneXY
        self._groups = lane_groups

    def get_positive_lanes(self):
        return [list(g) for g in self._groups]


class _FakeGBlock:
    """Simulates a G-block (FreeOutRampOnStraight) with two roads:
    - Main highway dec_road (A→B), 3 parallel lanes at lat 0, -4, -8
    - Deceleration/service lane dec_lane (N1→B), single lane at lat -12
    """

    def __init__(self, graph_block_id="g0"):
        self.graph_block_id = graph_block_id
        # Main highway: 3 lanes A→B, y=0, -4, -8
        hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
        # Deceleration lane N1→B, parallel but 1 more lane to the right (y=-12)
        deacc_lane = _FakeLaneXY("N1", "B", 0, (15.0, -12.0), (60.0, -12.0))
        self.block_network = _FakeBlockNetwork([hw_lanes, [deacc_lane]])

    def get_socket_list(self):
        return []


def test_get_all_positive_roads_from_block_returns_all_roads():
    """_get_all_positive_roads_from_block should return all roads in a G-block."""
    from models.diffusion.mode_context import _get_all_positive_roads_from_block

    block = _FakeGBlock()
    roads = _get_all_positive_roads_from_block(block)

    # Expect both A→B (main highway) and N1→B (deacc lane)
    road_keys = {(str(r.start_node), str(r.end_node)) for r in roads}
    assert ("A", "B") in road_keys, f"Expected (A,B) in roads, got {road_keys}"
    assert ("N1", "B") in road_keys, f"Expected (N1,B) in roads, got {road_keys}"
    assert len(roads) >= 2


def test_resolve_branch_polylines_finds_deacc_lane_in_g_block():
    """When vehicle is on main highway of G-block, the deceleration lane
    (a parallel road in the same block) should be identified as right_branch."""
    # G-block: main highway A→B (3 lanes, lat 0,-4,-8), deacc lane N1→B (lat -12)
    hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
    deacc_lane = _FakeLaneXY("N1", "B", 0, (15.0, -12.0), (60.0, -12.0))

    graph = {
        "A": {"B": hw_lanes},
        "N1": {"B": [deacc_lane]},
    }
    g0_block = _FakeGBlock("g0")
    current_map = _FakeMap(graph, blocks=[g0_block])

    # Vehicle is on the rightmost main highway lane (lane 2, lat -8)
    vehicle = _FakeVehicle(hw_lanes[2], current_map)
    vehicle.position = np.asarray(hw_lanes[2].position(10.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        scenario_id="S8_ego_exit_to_ramp",
        local_route="R6_exit_to_ramp",
        ego_main_route_block_ids=("g0", "s_ramp0"),
    )

    assert ctx.right_branch_polyline is not None, (
        "Expected right_branch_polyline to be set (deacc lane should be found)"
    )
    assert ctx.has_right_branch is True
    # Deacc lane is to the right (negative y in local frame)
    assert float(ctx.right_branch_polyline[-1, 1]) < -1.0


# ---------------------------------------------------------------------------
# Tests for S8 Problems 1-4: branch chaining, physical-lane index, ramp filter
# ---------------------------------------------------------------------------

class _FakeFullGBlock:
    """Simulates a full G-block: dec_road (A→B, 3 lanes), extend_road (B→C, 3
    lanes), deacc_lane (N1→B), and bend_1_road (B→B1).  Used to test the
    predecessor / sibling filter and the branch-polyline chain fix."""

    def __init__(self, graph_block_id: str = "g0"):
        self.graph_block_id = graph_block_id
        hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
        ext_lanes = [_FakeLaneXY("B", "C", i, (60.0, -i * 4.0), (120.0, -i * 4.0)) for i in range(3)]
        deacc_lane = _FakeLaneXY("N1", "B", 0, (30.0, -12.0), (60.0, -12.0))
        # bend_1_road: slight right diverge from (60,-12) to (80,-16)
        bend_1 = _FakeLaneXY("B", "B1", 0, (60.0, -12.0), (80.0, -16.0))
        self.block_network = _FakeBlockNetwork([hw_lanes, ext_lanes, [deacc_lane], [bend_1]])


def test_branch_polyline_stays_on_lane_centre_without_chaining():
    """P1 fix: right-branch polyline for deacc_lane should NOT chain into
    bend_1_road.  Instead it extrapolates linearly from the deacc_lane's exit
    heading, keeping the lateral offset constant.  This prevents the
    LC trajectory endpoint from following the ramp curve off the lane centre.

    Setup: vehicle at x=40 on dec_road lane 2 (y=-8).  deacc_lane spans x=30–60
    at y=-12.  bend_1_road continues from (60,-12) to (80,-16).

    The branch polyline represents only the deacc_lane.  Linear extrapolation
    from the lane end keeps last_y ≈ first_y (constant lateral offset).
    """
    hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
    deacc_lane = _FakeLaneXY("N1", "B", 0, (30.0, -12.0), (60.0, -12.0))
    bend_1 = _FakeLaneXY("B", "B1", 0, (60.0, -12.0), (80.0, -16.0))

    graph = {
        "A": {"B": hw_lanes},
        "N1": {"B": [deacc_lane]},
        "B": {"B1": [bend_1]},
    }
    g0_block = _FakeFullGBlock("g0")
    current_map = _FakeMap(graph, blocks=[g0_block])

    # Vehicle on dec_road lane 2 (y=-8) at x=40
    vehicle = _FakeVehicle(hw_lanes[2], current_map)
    vehicle.position = np.asarray(hw_lanes[2].position(40.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        ego_main_route_block_ids=("g0",),
    )

    assert ctx.right_branch_polyline is not None, "right_branch_polyline should be set"
    # First point should be near deacc_lane (y ≈ -4 in local from y=-8 on main road)
    assert float(ctx.right_branch_polyline[0, 1]) < -0.5, "first branch point should be right of vehicle"
    # Without chaining, the lateral offset stays constant (no bend_1 curve).
    first_y = float(ctx.right_branch_polyline[0, 1])
    last_y = float(ctx.right_branch_polyline[-1, 1])
    assert abs(last_y - first_y) < 1.0, (
        f"last branch point (y={last_y:.2f}) should stay at ≈same lateral offset "
        f"as first point (y={first_y:.2f}), not follow ramp curve"
    )


def test_adjacent_lane_uses_physical_lane_index_not_stale_nav_index():
    """P2: When vehicle.lane_index (navigation) lags behind vehicle.lane (physical),
    the physical lane's index must be used for adjacent-lane lookup.

    Setup: physical lane = deacc_lane ("N1","B",0) — single lane, no left adjacent.
    vehicle.lane_index is deliberately set to ("A","B",2) — stale navigation index.

    Expected after fix:
    - left_lane_polyline = None  (deacc_lane has no left adjacent)
    - left_branch target = dec_road lane 2 (positive y in local from deacc_lane)
    """
    hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
    deacc_lane = _FakeLaneXY("N1", "B", 0, (30.0, -12.0), (60.0, -12.0))

    graph = {
        "A": {"B": hw_lanes},
        "N1": {"B": [deacc_lane]},
    }
    g0_block = _FakeGBlock("g0")
    current_map = _FakeMap(graph, blocks=[g0_block])

    # Physical lane = deacc_lane; stale nav lane_index = ("A","B",2)
    vehicle = _FakeVehicle(deacc_lane, current_map)
    vehicle.position = np.asarray(deacc_lane.position(10.0, 0.0), dtype=np.float32)
    vehicle.lane_index = ("A", "B", 2)  # stale navigation index

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        ego_main_route_block_ids=("g0",),
    )

    # With fix: physical lane ("N1","B",0) is used → no left adjacent on deacc_lane
    assert ctx.left_lane_polyline is None, (
        "left_lane_polyline must be None when physical lane is the single-lane deacc_lane "
        "(fix: use current_lane.index not vehicle.lane_index)"
    )
    # With fix: left_branch_polyline targets dec_road lane 2 (to the left of deacc_lane)
    assert ctx.left_branch_polyline is not None, (
        "left_branch_polyline should target dec_road lane 2 when on deacc_lane"
    )
    assert float(ctx.left_branch_polyline[-1, 1]) > 0.5, (
        "left_branch target should be to the LEFT (+y) of the deacc_lane vehicle"
    )


def test_no_branch_after_entering_ramp_junction():
    """P3: Once the vehicle has entered the ramp (on bend_1_road, departed junction B),
    both left and right branch polylines must be None so that all LC modes become
    invalid immediately.

    The predecessor filter removes dec_road/deacc_lane (end at B = cur_start).
    The sibling filter removes extend_road (starts at B = cur_start).
    """
    hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
    ext_lanes = [_FakeLaneXY("B", "C", i, (60.0, -i * 4.0), (120.0, -i * 4.0)) for i in range(3)]
    deacc_lane = _FakeLaneXY("N1", "B", 0, (30.0, -12.0), (60.0, -12.0))
    bend_1 = _FakeLaneXY("B", "B1", 0, (60.0, -12.0), (80.0, -16.0))

    graph = {
        "A": {"B": hw_lanes},
        "B": {"C": ext_lanes, "B1": [bend_1]},
        "N1": {"B": [deacc_lane]},
    }
    g0_block = _FakeFullGBlock("g0")
    current_map = _FakeMap(graph, blocks=[g0_block])

    # Vehicle has just entered bend_1_road (just past junction B)
    vehicle = _FakeVehicle(bend_1, current_map)
    vehicle.position = np.asarray(bend_1.position(2.0, 0.0), dtype=np.float32)
    vehicle.heading_theta = float(bend_1.heading_theta_at(2.0))

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        ego_main_route_block_ids=("g0",),
    )

    # Predecessor filter suppresses dec_road and deacc_lane (both end at B).
    # Sibling filter suppresses extend_road (starts at B, same as bend_1).
    assert ctx.left_branch_polyline is None, (
        "left_branch_polyline must be None on ramp (predecessor/sibling filter)"
    )
    assert ctx.right_branch_polyline is None, (
        "right_branch_polyline must be None on ramp (predecessor/sibling filter)"
    )
    assert ctx.has_left_branch is False
    assert ctx.has_right_branch is False


def test_branch_endpoint_extrapolates_at_constant_lateral_offset():
    """P1/P4 fix: When deacc_lane is short (only 10 m remaining), the
    right-branch endpoint should linearly extrapolate from the lane's exit
    heading, keeping the same lateral offset instead of following bend_1_road.

    The endpoint stays at world_y ≈ -12 (the deacc lane y-level), NOT shifting
    to y < -12 (which would mean bend_1 was chained).
    """
    hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
    deacc_lane = _FakeLaneXY("N1", "B", 0, (30.0, -12.0), (60.0, -12.0))
    bend_1 = _FakeLaneXY("B", "B1", 0, (60.0, -12.0), (80.0, -16.0))

    graph = {
        "A": {"B": hw_lanes},
        "N1": {"B": [deacc_lane]},
        "B": {"B1": [bend_1]},
    }
    g0_block = _FakeFullGBlock("g0")
    current_map = _FakeMap(graph, blocks=[g0_block])

    # Vehicle near junction B — only 10 m of deacc_lane remain from projection
    vehicle = _FakeVehicle(hw_lanes[2], current_map)
    vehicle.position = np.asarray(hw_lanes[2].position(50.0, 0.0), dtype=np.float32)

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        ego_main_route_block_ids=("g0",),
    )

    assert ctx.right_branch_polyline is not None, "right_branch_polyline must be set"

    # Compute world-space endpoint
    veh_x, veh_y = float(vehicle.position[0]), float(vehicle.position[1])
    end_local = ctx.right_branch_polyline[-1]  # (local_x, local_y)
    end_world_x = veh_x + float(end_local[0])
    end_world_y = veh_y + float(end_local[1])

    # Without chaining: straight-ahead extrapolation from deacc_lane gives
    # world_y ≈ -12 (constant lateral offset).  The endpoint should NOT be
    # below -12 (which would mean following bend_1 curve).
    assert end_world_y >= -13.0, (
        f"endpoint world_y={end_world_y:.2f} should stay near deacc lane level "
        f"(-12), not follow bend_1 curve further down"
    )
    # And the endpoint must be ahead of the vehicle (positive local_x)
    assert float(end_local[0]) > 5.0, "branch endpoint should be well ahead of vehicle"


def test_no_branch_on_ramp_after_highway_divergence():
    """When the vehicle is on s_ramp0 (a continuation road beyond the
    g0 block's ramp exit), the highway continuation road (extend_road
    B→C and s_main0 C→D) must NOT appear as branch candidates, even
    though they are physically nearby.

    Topology (simplified G-block + successors):
      A →B (highway dec_road, 3 lanes)
      N1→B (deacc_lane, 1 lane)
      B →C (extend_road, 3 lanes)  ← highway continuation
      B →B1 (bend_1, ramp entry)
      B1→B2 (connect)
      B2→R  (straight_part)
      R →S  (s_ramp0, 1 lane)      ← vehicle is here
      C →D  (s_main0, 3 lanes)     ← must NOT be a branch

    The diverged-branch filter detects that C is a diverged successor
    of ancestor B (B has successors {C, B1}; the vehicle took B→B1,
    so C is diverged).  extend_road B→C is also filtered because
    B is an ancestor and C is not an ancestor.
    """
    # g0 block roads
    hw_lanes = [_FakeLaneXY("A", "B", i, (0.0, -i * 4.0), (60.0, -i * 4.0)) for i in range(3)]
    deacc_lane = _FakeLaneXY("N1", "B", 0, (30.0, -12.0), (60.0, -12.0))
    ext_lanes = [_FakeLaneXY("B", "C", i, (60.0, -i * 4.0), (120.0, -i * 4.0)) for i in range(3)]
    bend_1 = _FakeLaneXY("B", "B1", 0, (60.0, -12.0), (70.0, -14.0))
    connect = _FakeLaneXY("B1", "B2", 0, (70.0, -14.0), (90.0, -14.0))
    straight_part = _FakeLaneXY("B2", "R", 0, (90.0, -14.0), (105.0, -14.0))

    # s_ramp0 block road
    s_ramp0_lane = _FakeLaneXY("R", "S", 0, (105.0, -14.0), (165.0, -14.0))

    # s_main0 block road (highway continuation, physically nearby at y=0..-8)
    s_main0_lanes = [_FakeLaneXY("C", "D", i, (120.0, -i * 4.0), (320.0, -i * 4.0)) for i in range(3)]

    graph = {
        "A": {"B": hw_lanes},
        "N1": {"B": [deacc_lane]},
        "B": {"C": ext_lanes, "B1": [bend_1]},
        "B1": {"B2": [connect]},
        "B2": {"R": [straight_part]},
        "R": {"S": [s_ramp0_lane]},
        "C": {"D": s_main0_lanes},
    }

    g0_block = type("Block", (), {
        "graph_block_id": "g0",
        "block_network": _FakeBlockNetwork([hw_lanes, [deacc_lane], ext_lanes,
                                            [bend_1], [connect], [straight_part]]),
        "get_socket_list": lambda self: [],
    })()
    s_ramp0_block = type("Block", (), {
        "graph_block_id": "s_ramp0",
        "block_network": _FakeBlockNetwork([[s_ramp0_lane]]),
        "get_socket_list": lambda self: [],
    })()
    s_main0_block = type("Block", (), {
        "graph_block_id": "s_main0",
        "block_network": _FakeBlockNetwork([s_main0_lanes]),
        "get_socket_list": lambda self: [],
    })()

    current_map = _FakeMap(graph, blocks=[g0_block, s_ramp0_block, s_main0_block])

    # Vehicle is on s_ramp0 near start
    vehicle = _FakeVehicle(s_ramp0_lane, current_map)
    vehicle.position = np.asarray(s_ramp0_lane.position(10.0, 0.0), dtype=np.float32)
    vehicle.heading_theta = float(s_ramp0_lane.heading_theta_at(10.0))

    ctx = build_mode_context_from_vehicle(
        vehicle,
        current_map=current_map,
        ego_main_route_block_ids=("g0", "s_ramp0"),
    )

    assert ctx.left_branch_polyline is None, (
        "left_branch must be None on s_ramp0 (highway extension is a diverged branch)"
    )
    assert ctx.right_branch_polyline is None, (
        "right_branch must be None on s_ramp0"
    )
    assert ctx.has_left_branch is False
    assert ctx.has_right_branch is False
