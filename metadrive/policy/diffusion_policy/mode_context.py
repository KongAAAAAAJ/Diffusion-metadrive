from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy


def _as_numpy_2d(points: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return None
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected polyline with shape (N, 2), got {tuple(array.shape)}")
    return array


def _as_scalar(sample: Dict[str, Any], key: str, default: float) -> float:
    value = sample.get(key, default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _as_int(sample: Dict[str, Any], key: str, default: int) -> int:
    value = sample.get(key, default)
    array = np.asarray(value)
    if array.size == 0:
        return int(default)
    return int(array.reshape(-1)[0])


def _as_bool(sample: Dict[str, Any], key: str, default: bool) -> bool:
    value = sample.get(key, default)
    array = np.asarray(value)
    if array.size == 0:
        return bool(default)
    return bool(array.reshape(-1)[0])


def _world_to_local_xy(current_pose: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32)
    world_points = np.asarray(world_points, dtype=np.float32)
    dx = world_points[:, 0] - current_pose[0]
    dy = world_points[:, 1] - current_pose[1]
    heading = float(current_pose[2])
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    return np.stack([local_x, local_y], axis=1).astype(np.float32, copy=False)


def _build_fallback_current_lane_polyline(sample: Dict[str, Any]) -> np.ndarray:
    current_pose = np.asarray(sample["reference_pose_world"], dtype=np.float32)
    future_reference = np.asarray(sample["future_reference_pose_world"], dtype=np.float32)
    if future_reference.ndim != 2 or future_reference.shape[0] == 0:
        return np.zeros((2, 2), dtype=np.float32)
    local_points = _world_to_local_xy(current_pose, future_reference)
    return np.concatenate([np.zeros((1, 2), dtype=np.float32), local_points], axis=0)


def _sample_lane_polyline(
    lane: Any,
    num_points: int = 30,
    longitudinal_step_m: float = 2.0,
    start_longitudinal: float = 0.0,
) -> np.ndarray:
    lane_length = float(getattr(lane, "length", 0.0))
    if lane_length <= 0.0:
        return np.zeros((2, 2), dtype=np.float32)
    start_longitudinal = float(np.clip(float(start_longitudinal), 0.0, lane_length))
    distances = start_longitudinal + np.arange(0.0, longitudinal_step_m * num_points, longitudinal_step_m, dtype=np.float32)

    # For distances within the lane, sample from the lane directly.
    # For distances beyond the lane end, extrapolate linearly from the
    # lane endpoint using the exit heading.  This avoids clamped duplicate
    # points that create false kink angles in downstream polyline checks.
    points: list[np.ndarray] = []
    end_pos: np.ndarray | None = None
    end_heading: float | None = None
    end_dir: np.ndarray | None = None

    for d in distances:
        d_f = float(d)
        if d_f <= lane_length:
            points.append(np.asarray(lane.position(d_f, 0.0), dtype=np.float32))
        else:
            # Lazy-init end state
            if end_pos is None:
                end_pos = np.asarray(lane.position(lane_length, 0.0), dtype=np.float32)
                try:
                    end_heading = float(lane.heading_theta_at(lane_length))
                except Exception:
                    end_heading = float(lane.heading_theta_at(0.0))
                end_dir = np.array([math.cos(end_heading), math.sin(end_heading)],
                                   dtype=np.float32)
            overshoot = d_f - lane_length
            points.append(end_pos + end_dir * overshoot)

    return np.asarray(points, dtype=np.float32)


def _sample_lane_polyline_from_vehicle_position(
    lane: Any,
    vehicle_position: np.ndarray,
    num_points: int = 30,
    longitudinal_step_m: float = 2.0,
) -> np.ndarray:
    start_longitudinal = 0.0
    if hasattr(lane, "local_coordinates"):
        try:
            start_longitudinal = float(lane.local_coordinates(vehicle_position)[0])
        except Exception:
            start_longitudinal = 0.0
    return _sample_lane_polyline(
        lane,
        num_points=num_points,
        longitudinal_step_m=longitudinal_step_m,
        start_longitudinal=start_longitudinal,
    )


def _connection_score(current_lane: Any, candidate_lane: Any) -> float:
    try:
        current_end = np.asarray(
            current_lane.position(float(getattr(current_lane, "length", 0.0)), 0.0),
            dtype=np.float32,
        )
        lane_start = np.asarray(candidate_lane.position(0.0, 0.0), dtype=np.float32)
        current_heading = float(current_lane.heading_theta_at(float(getattr(current_lane, "length", 0.0)))) \
            if hasattr(current_lane, "heading_theta_at") else 0.0
        lane_heading = float(candidate_lane.heading_theta_at(0.0)) if hasattr(candidate_lane, "heading_theta_at") else current_heading
    except Exception:
        return float("inf")

    position_cost = float(np.linalg.norm(lane_start - current_end))
    heading_delta = math.atan2(math.sin(lane_heading - current_heading), math.cos(lane_heading - current_heading))
    current_lane_index = getattr(current_lane, "index", None)
    candidate_lane_index = getattr(candidate_lane, "index", None)
    lane_id_cost = 0.0
    if current_lane_index is not None and candidate_lane_index is not None:
        try:
            lane_id_cost = abs(int(candidate_lane_index[2]) - int(current_lane_index[2])) * 1.5
        except Exception:
            lane_id_cost = 0.0
        if str(candidate_lane_index[1]) == str(current_lane_index[0]):
            lane_id_cost += 1000.0
    return position_cost + abs(float(heading_delta)) * 2.0 + lane_id_cost


def _resolve_best_connected_lane(current_lane: Any, candidate_lanes: Sequence[Any] | None) -> Any | None:
    """Pick the best continuation lane using a combined score.

    The score combines geometric distance (endpoint gap + heading delta) with
    a lane-id proximity bonus.  Geometry is the primary signal so that
    topology-changing transitions (e.g. single-lane ramp → 3-lane mainline)
    pick the physically nearest lane rather than the one whose lane index
    happens to match.  The lane-id bonus (1.5 per index difference) is large
    enough to break ties on same-road continuations where geometric distances
    are nearly identical.

    Backtrack candidates — lanes whose end_node equals the current lane's
    start_node — are excluded outright.
    """
    if not candidate_lanes:
        return None

    current_index = getattr(current_lane, "index", None)
    try:
        current_id: int | None = int(current_index[2]) if current_index is not None else None
    except Exception:
        current_id = None

    best_lane = None
    best_score: float = float("inf")

    for lane in candidate_lanes:
        lane_index = getattr(lane, "index", None)

        # Backtrack check: skip lanes that loop back to current lane's source node.
        if current_index is not None and lane_index is not None:
            if str(lane_index[1]) == str(current_index[0]):
                continue

        # Geometric score: endpoint gap + heading delta.
        try:
            current_end = np.asarray(
                current_lane.position(float(getattr(current_lane, "length", 0.0)), 0.0),
                dtype=np.float32,
            )
            lane_start = np.asarray(lane.position(0.0, 0.0), dtype=np.float32)
            current_heading = (
                float(current_lane.heading_theta_at(float(getattr(current_lane, "length", 0.0))))
                if hasattr(current_lane, "heading_theta_at")
                else 0.0
            )
            lane_heading = float(lane.heading_theta_at(0.0)) if hasattr(lane, "heading_theta_at") else current_heading
        except Exception:
            continue

        position_cost = float(np.linalg.norm(lane_start - current_end))
        heading_delta = math.atan2(math.sin(lane_heading - current_heading), math.cos(lane_heading - current_heading))
        geom = position_cost + abs(float(heading_delta)) * 2.0
        if not np.isfinite(geom):
            continue

        # Lane-ID bonus: 1.5 per lane-index difference.  On same-road
        # continuations (position_cost ≈ 0), this dominates and preserves
        # lane identity.  On topology changes (position_cost >> 1.5), geometry
        # naturally selects the physically nearest lane.
        lane_id_cost: float = 0.0
        if current_id is not None and lane_index is not None:
            try:
                lane_id_cost = float(abs(int(lane_index[2]) - current_id)) * 1.5
            except Exception:
                lane_id_cost = 0.0

        score = geom + lane_id_cost
        if score < best_score:
            best_score = score
            best_lane = lane

    return best_lane


def _lane_start_longitudinal(lane: Any, vehicle_position: np.ndarray) -> float:
    start_longitudinal = 0.0
    if hasattr(lane, "local_coordinates"):
        try:
            start_longitudinal = float(lane.local_coordinates(vehicle_position)[0])
        except Exception:
            start_longitudinal = 0.0
    lane_length = float(getattr(lane, "length", 0.0))
    return float(np.clip(start_longitudinal, 0.0, lane_length))


def _sample_route_ahead_on_lane_chain(
    lane_chain: Sequence[tuple[Any, float]],
    num_points: int = 30,
    longitudinal_step_m: float = 2.0,
) -> np.ndarray:
    if not lane_chain:
        return np.zeros((2, 2), dtype=np.float32)

    segments: list[tuple[Any, float, float]] = []
    for lane, start_longitudinal in lane_chain:
        lane_length = float(getattr(lane, "length", 0.0))
        if lane_length <= 0.0:
            continue
        start_longitudinal = float(np.clip(float(start_longitudinal), 0.0, lane_length))
        traversable_length = max(lane_length - start_longitudinal, 0.0)
        segments.append((lane, start_longitudinal, traversable_length))
    if not segments:
        return np.zeros((2, 2), dtype=np.float32)

    cumulative_lengths: list[float] = []
    total_length = 0.0
    for _, _, traversable_length in segments:
        total_length += traversable_length
        cumulative_lengths.append(total_length)

    points: list[np.ndarray] = []
    for step_idx in range(num_points):
        route_distance = float(step_idx) * float(longitudinal_step_m)
        remaining_distance = min(route_distance, total_length)
        lane_idx = 0
        previous_cumulative = 0.0
        for idx, cumulative in enumerate(cumulative_lengths):
            if remaining_distance <= cumulative + 1e-6:
                lane_idx = idx
                break
            previous_cumulative = cumulative
        lane, start_longitudinal, _ = segments[lane_idx]
        local_distance = remaining_distance - previous_cumulative
        sample_longitudinal = start_longitudinal + local_distance
        points.append(np.asarray(lane.position(float(sample_longitudinal), 0.0), dtype=np.float32))

    return np.asarray(points, dtype=np.float32)


def _sample_route_ahead_polyline_from_vehicle_position(
    lane: Any,
    vehicle_position: np.ndarray,
    next_ref_lanes: Sequence[Any] | None,
    num_points: int = 30,
    longitudinal_step_m: float = 2.0,
    current_map: Any = None,
) -> np.ndarray:
    start_longitudinal = _lane_start_longitudinal(lane, vehicle_position)
    lane_chain: list[tuple[Any, float]] = [(lane, start_longitudinal)]
    lane_length = float(getattr(lane, "length", 0.0))
    remaining_length = max(lane_length - start_longitudinal, 0.0)
    required_length = max(float(num_points - 1), 0.0) * float(longitudinal_step_m)

    # Chain into successor lanes until the required polyline length is covered.
    # At each level, verify that the geometric gap is small (≤ 2 m) to avoid
    # chaining across lateral offsets that would produce heading kinks.
    chain_lane = lane
    chain_remaining = remaining_length
    chain_next_refs = next_ref_lanes
    for _ in range(4):  # up to 4 successor levels
        if chain_remaining + 1e-6 >= required_length:
            break
        next_lane = _resolve_best_connected_lane(chain_lane, chain_next_refs)
        if next_lane is None:
            break
        try:
            current_end = np.asarray(
                chain_lane.position(float(getattr(chain_lane, "length", 0.0)), 0.0),
                dtype=np.float32)
            next_start = np.asarray(
                next_lane.position(0.0, 0.0), dtype=np.float32)
            gap = float(np.linalg.norm(next_start - current_end))
        except Exception:
            gap = 0.0
        if gap > 2.0:
            break
        lane_chain.append((next_lane, 0.0))
        next_lane_length = float(getattr(next_lane, "length", 0.0))
        chain_remaining += next_lane_length
        chain_lane = next_lane
        # For subsequent levels, resolve successor lanes from the road graph.
        chain_next_refs = _resolve_terminal_route_successor_lanes(
            chain_lane, current_map, ()
        ) if current_map is not None else None

    return _sample_route_ahead_on_lane_chain(
        lane_chain,
        num_points=num_points,
        longitudinal_step_m=longitudinal_step_m,
    )


def _road_key(road: Any) -> tuple[str, str]:
    return (str(getattr(road, "start_node", "")), str(getattr(road, "end_node", "")))


def _get_route_blocks(current_map: Any, block_ids: Sequence[str] | None) -> list[Any]:
    if current_map is None or not block_ids:
        return []
    blocks_by_id = {
        str(getattr(block, "graph_block_id", "")): block
        for block in getattr(current_map, "blocks", []) or []
        if getattr(block, "graph_block_id", None) is not None
    }
    return [blocks_by_id[block_id] for block_id in block_ids if block_id in blocks_by_id]


def _get_route_roads(current_map: Any, block_ids: Sequence[str] | None) -> list[Any]:
    route_blocks = _get_route_blocks(current_map, block_ids)
    route_roads: list[Any] = []
    for block in route_blocks:
        road = _get_first_positive_road_from_block(block)
        if road is not None:
            route_roads.append(road)
    return route_roads


def _lane_matches_road(lane: Any, road: Any) -> bool:
    lane_index = getattr(lane, "index", None)
    if lane_index is None or road is None:
        return False
    return str(lane_index[0]) == str(getattr(road, "start_node", "")) and str(lane_index[1]) == str(getattr(road, "end_node", ""))


def _resolve_terminal_route_successor_lanes(
    lane: Any,
    current_map: Any,
    route_roads: Sequence[Any] = (),
) -> Sequence[Any] | None:
    """Search the map's road graph for successor lanes when navigation has no ``next_ref_lanes``.

    ``route_roads`` is accepted for API compatibility but is **not** used as a gate: the
    function activates whenever the lane has a valid index and the map exposes a road network.
    The caller (``_resolve_candidate_next_lanes``) already ensures this is only invoked when
    ``next_ref_lanes`` is None — i.e. the vehicle is on the terminal route segment.
    """
    lane_index = getattr(lane, "index", None)
    road_network = getattr(current_map, "road_network", None) if current_map is not None else None
    if lane_index is None or road_network is None:
        return None
    successors = getattr(road_network, "graph", {}).get(lane_index[1], {})
    if not successors:
        return None

    best_successor_lanes: Sequence[Any] | None = None
    best_score: float | None = None
    for end_node, successor_lanes in successors.items():
        # Backtrack: skip roads that terminate at the current lane's source node.
        if str(end_node) == str(lane_index[0]) or not successor_lanes:
            continue
        connected_lane = _resolve_best_connected_lane(lane, successor_lanes)
        if connected_lane is None:
            continue
        score = _connection_score(lane, connected_lane)
        if best_score is None or score < best_score:
            best_score = score
            best_successor_lanes = successor_lanes
    return best_successor_lanes


def _resolve_candidate_next_lanes(
    lane: Any,
    current_map: Any,
    next_ref_lanes: Sequence[Any] | None,
    route_roads: Sequence[Any],
) -> Sequence[Any] | None:
    topology_successors = _resolve_terminal_route_successor_lanes(lane, current_map, route_roads)
    if not next_ref_lanes:
        return topology_successors
    if not topology_successors:
        return next_ref_lanes

    merged_lanes: list[Any] = []
    seen_indices: set[Any] = set()
    for candidate_lane in list(next_ref_lanes) + list(topology_successors):
        lane_index = getattr(candidate_lane, "index", None)
        key = tuple(lane_index) if lane_index is not None else id(candidate_lane)
        if key in seen_indices:
            continue
        seen_indices.add(key)
        merged_lanes.append(candidate_lane)
    return merged_lanes


def _get_first_positive_road_from_block(block: Any) -> Any | None:
    if block is None:
        return None
    lane_groups = getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []
    for lane_group in lane_groups:
        if not lane_group:
            continue
        lane = lane_group[0]
        lane_index = getattr(lane, "index", None)
        if lane_index is None:
            continue
        return type("RoadRef", (), {"start_node": lane_index[0], "end_node": lane_index[1]})()
    roads = []
    for socket in getattr(block, "get_socket_list", lambda: [])() or []:
        positive_road = getattr(socket, "positive_road", None)
        if positive_road is not None:
            roads.append(positive_road)
    for socket in (getattr(block, "_sockets", {}) or {}).values():
        positive_road = getattr(socket, "positive_road", None)
        if positive_road is not None:
            roads.append(positive_road)
    if not roads:
        return None
    roads = sorted(roads, key=_road_key)
    return roads[0]


def _get_all_positive_roads_from_block(block: Any) -> list[Any]:
    """Return all positive road references from a block's network.

    Unlike ``_get_first_positive_road_from_block``, this function enumerates
    *every* positive road inside the block's ``block_network``.  This is
    necessary for blocks like the G-block (FreeOutRampOnStraight) whose
    internal roads include, in addition to the through-lanes, a parallel
    deceleration / service lane that forms the ramp-exit path.  When only the
    first road is inspected the deceleration lane is never considered as a
    branch candidate.
    """
    if block is None:
        return []
    roads: list[Any] = []
    seen: set[tuple[str, str]] = set()

    # Primary: iterate all positive lane-groups in the block network
    lane_groups = getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []
    for lane_group in lane_groups:
        if not lane_group:
            continue
        lane = lane_group[0]
        lane_index = getattr(lane, "index", None)
        if lane_index is None:
            continue
        key = (str(lane_index[0]), str(lane_index[1]))
        if key not in seen:
            seen.add(key)
            roads.append(type("RoadRef", (), {"start_node": lane_index[0], "end_node": lane_index[1]})())

    if roads:
        return roads

    # Fallback: read from sockets (used by mock objects in tests)
    for socket in getattr(block, "get_socket_list", lambda: [])() or []:
        positive_road = getattr(socket, "positive_road", None)
        if positive_road is not None:
            key = (str(getattr(positive_road, "start_node", "")), str(getattr(positive_road, "end_node", "")))
            if key not in seen:
                seen.add(key)
                roads.append(positive_road)
    for socket in (getattr(block, "_sockets", {}) or {}).values():
        positive_road = getattr(socket, "positive_road", None)
        if positive_road is not None:
            key = (str(getattr(positive_road, "start_node", "")), str(getattr(positive_road, "end_node", "")))
            if key not in seen:
                seen.add(key)
                roads.append(positive_road)
    return roads


def _sample_road_polyline(road_network: Any, road: Any) -> np.ndarray | None:
    if road_network is None or road is None:
        return None
    try:
        lanes = road_network.graph[road.start_node][road.end_node]
    except Exception:
        return None
    if not lanes:
        return None
    return _sample_lane_polyline(lanes[0])


def _extract_front_and_lane_gaps(vehicle: Any, ref_lane: Any) -> tuple[float, float, float, float]:
    if ref_lane is None or not hasattr(vehicle, "lidar"):
        return -1.0, -1.0, -1.0, -1.0
    try:
        current_ref_lanes = getattr(vehicle.navigation, "current_ref_lanes", None)
        all_objects = vehicle.lidar.get_surrounding_objects(vehicle)
        surrounding = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            ref_lane,
            vehicle.position,
            max_distance=IDMPolicy.MAX_LONG_DIST,
            ref_lanes=current_ref_lanes if current_ref_lanes and ref_lane in current_ref_lanes else None,
        )
    except Exception:
        return -1.0, -1.0, -1.0, -1.0

    front_object = surrounding.front_object()
    front_distance = surrounding.front_min_distance()
    front_speed = None if front_object is None else float(getattr(front_object, "speed_km_h", 0.0)) / 3.6

    def _gap(lane_exists_attr: str, front_attr: str, back_attr: str) -> float:
        lane_exists = getattr(surrounding, lane_exists_attr)() if hasattr(surrounding, lane_exists_attr) else False
        if not lane_exists:
            return -1.0
        front_gap = getattr(surrounding, front_attr)() if hasattr(surrounding, front_attr) else None
        back_gap = getattr(surrounding, back_attr)() if hasattr(surrounding, back_attr) else None
        candidates = [float(value) for value in (front_gap, back_gap) if value is not None]
        return min(candidates) if candidates else -1.0

    left_gap = _gap("left_lane_exist", "left_front_min_distance", "left_back_min_distance")
    right_gap = _gap("right_lane_exist", "right_front_min_distance", "right_back_min_distance")
    return (
        -1.0 if front_distance is None else float(front_distance),
        -1.0 if front_speed is None else float(front_speed),
        float(left_gap),
        float(right_gap),
    )


# Maximum forward distance (m) for the first polyline point of a branch road.
# Branches whose road start is farther than this are premature (vehicle not yet
# near the junction) and are suppressed to prevent very long lane-change arcs.
_MAX_BRANCH_FIRST_POINT_X: float = 40.0
# Roads that start more than this many metres BEHIND the vehicle are also
# discarded — they are either continuations that turned (false positives due
# to curve heading) or roads that the vehicle has already passed.
_MIN_BRANCH_FIRST_POINT_X: float = -10.0
# Maximum lateral distance (m) for a branch road to be considered a valid
# lane-change target.  Roads farther than this are on entirely different parts
# of the map (e.g. a remote curve segment appearing to the side due to map
# topology) and cannot be reached by a single lane change.
_MAX_BRANCH_LATERAL_DISTANCE: float = 12.0
# Maximum consecutive heading change (rad) allowed in a branch polyline.
# Polylines with sharper kinks contain U-turns or wrapping artefacts.
_MAX_BRANCH_POLYLINE_KINK_RAD: float = math.pi / 3.0  # 60 degrees


def _build_reachable_nodes(graph: dict, start_node: str, max_hops: int = 5) -> set[str]:
    """BFS forward from *start_node*, returning all reachable node IDs."""
    visited: set[str] = set()
    frontier: set[str] = {start_node}
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for end_node in graph.get(node, {}):
                s_end = str(end_node)
                if s_end not in visited and s_end not in frontier:
                    next_frontier.add(s_end)
        visited.update(frontier)
        frontier = next_frontier
    visited.update(frontier)
    return visited


def _is_grand_predecessor(graph: dict, road_start: str, road_end: str,
                          cur_start: str, cur_end: str) -> bool:
    """Return True if the candidate road should be filtered as a
    "grand-predecessor" — i.e. it originates from a node that has a direct
    edge to our current lane's start node (the vehicle has already passed
    that junction), UNLESS the candidate road converges to the same end node
    as our current lane (parallel roads merging at the same junction, which
    are valid lane-change targets).

    Example (G-block topology):
      - extend_road (4G0_0_→4G0_1_) when on connect (4G1_1_→4G1_2_):
        4G0_0_ has edge to 4G1_1_ via bend_1, road_end 4G0_1_ ≠ cur_end 4G1_2_
        → SKIP (vehicle passed the junction, chose ramp, can't go back)
      - dec_road (3C0_1_→4G0_0_) when on deacc_lane (4G1_0_→4G0_0_):
        3C0_1_ has edge to 4G1_0_ via merge_part, road_end 4G0_0_ == cur_end 4G0_0_
        → NOT skipped (parallel road converging to same end, valid LC target)
    """
    for succ_node in graph.get(road_start, {}):
        if str(succ_node) == cur_start:
            # road_start is a 1-hop predecessor of cur_start.
            # Allow if the candidate road CONVERGES to the same end node.
            return road_end != cur_end
    return False


def _build_diverged_branch_starts(
    graph: dict, cur_start: str, max_ancestor_hops: int = 6,
) -> tuple[set[str], set[str]]:
    """Return ``(ancestors, diverged)`` for the current lane's start node.

    **ancestors**: nodes from which *cur_start* is reachable via forward edges.
    **diverged**: direct successors of ancestors that are NOT ancestors
    themselves and are NOT *cur_start* — they lie on forks that diverged from
    the vehicle's historical path.

    The caller uses both sets to filter branch candidates:
    1. Roads whose ``start_node`` is in *diverged* → skip (the vehicle chose
       a different fork at the shared ancestor junction).
    2. Roads whose ``start_node`` is in *ancestors* **and** whose
       ``end_node`` is **not** in *ancestors* and ≠ cur_start/cur_end → skip
       (the road leaves the ancestral path toward a diverged branch).

    Example (G-block, vehicle on s_ramp0: 4G1_4_→6s0_0_):
      Ancestors of 4G1_4_: {4G1_3_, 4G1_2_, 4G1_1_, 4G0_0_, 3C0_1_, 4G1_0_}
      At ancestor 4G0_0_: successors {4G0_1_, 4G1_1_}.
        4G1_1_ ∈ ancestors → OK.
        4G0_1_ ∉ ancestors → diverged.
      ⇒ diverged = {4G0_1_}
      extend_road (4G0_0_→4G0_1_): start ∈ ancestors, end ∉ ancestors,
        end ≠ cur_start, end ≠ cur_end → SKIP.
      s_main0 (4G0_1_→5S0_0_): start ∈ diverged → SKIP.
    """
    # Build reverse adjacency: reverse[dst] = {src, …}
    reverse: dict[str, set[str]] = {}
    for src in graph:
        for dst in graph[src]:
            reverse.setdefault(str(dst), set()).add(str(src))

    s_cur = str(cur_start)
    ancestors: set[str] = set()
    frontier: set[str] = {s_cur}
    for _ in range(max_ancestor_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for parent in reverse.get(node, set()):
                if parent not in ancestors and parent != s_cur:
                    ancestors.add(parent)
                    next_frontier.add(parent)
        frontier = next_frontier
        if not frontier:
            break

    diverged: set[str] = set()
    for anc in ancestors:
        for succ in graph.get(anc, {}):
            s_succ = str(succ)
            if s_succ not in ancestors and s_succ != s_cur:
                diverged.add(s_succ)
    return ancestors, diverged


def _resolve_branch_polylines_from_route(
    current_pose: np.ndarray,
    current_map: Any,
    route_blocks: Sequence[Any],
    current_lane: Any = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build left/right branch polylines from the route blocks.

    Only roads that are *different* from the current lane's road and are
    laterally separated (>=0.5 m) are considered as branches.  Roads whose
    sampled first point is outside the window
    [_MIN_BRANCH_FIRST_POINT_X, _MAX_BRANCH_FIRST_POINT_X] are skipped:
    too-far-ahead roads produce premature overlong arcs; too-far-behind roads
    are either already-passed segments or curve-heading false positives.  For
    each qualifying road, the lane nearest to the vehicle (smallest absolute
    lateral offset) is selected as the branch target.

    In addition to the immediate predecessor / sibling / forward checks, a
    *reachability filter* excludes roads whose start node is reachable from
    the current lane's end node via forward traversal (downstream continuation)
    or whose start node can reach the current lane's start node via backward
    traversal (upstream ancestor).  This prevents downstream ramp segments or
    the main-highway extension from appearing as lane-change targets once the
    vehicle has committed to a ramp branch.
    """
    if current_map is None or len(route_blocks) < 1:
        return None, None
    road_network = getattr(current_map, "road_network", None)
    if road_network is None:
        return None, None

    # Current lane road identification — used to skip the road the vehicle is on.
    cur_lane_idx = getattr(current_lane, "index", None) if current_lane is not None else None
    cur_start = str(cur_lane_idx[0]) if cur_lane_idx is not None else None
    cur_end = str(cur_lane_idx[1]) if cur_lane_idx is not None else None

    # Build reachability sets for topological filtering.
    graph = getattr(road_network, "graph", {}) or {}
    forward_reachable: set[str] = set()
    if cur_end is not None:
        forward_reachable = _build_reachable_nodes(graph, cur_end, max_hops=6)
    diverged_starts: set[str] = set()
    ancestor_nodes: set[str] = set()
    if cur_start is not None:
        ancestor_nodes, diverged_starts = _build_diverged_branch_starts(graph, cur_start, max_ancestor_hops=6)

    vehicle_position = current_pose[:2]
    left_branch: Optional[np.ndarray] = None
    right_branch: Optional[np.ndarray] = None

    for block in route_blocks:
        block_roads = _get_all_positive_roads_from_block(block)
        for road in block_roads:
            road_start = str(getattr(road, "start_node", ""))
            road_end = str(getattr(road, "end_node", ""))

            # Skip the road the vehicle is currently on.
            if cur_start is not None and road_start == cur_start and road_end == cur_end:
                continue

            # Skip forward continuation.
            if cur_end is not None and road_start == cur_end:
                continue

            # Skip predecessor roads.
            if cur_start is not None and road_end == cur_start:
                continue

            # Skip sibling roads.
            if cur_start is not None and road_start == cur_start:
                continue

            # Skip downstream-reachable roads.
            if road_start in forward_reachable:
                continue

            # Grand-predecessor filter: skip roads originating from a node
            # that has a direct edge to our start node (already-passed
            # junction), UNLESS the road converges to the same end node as
            # the current lane (parallel roads are valid LC targets).
            if cur_start is not None and cur_end is not None:
                if _is_grand_predecessor(graph, road_start, road_end,
                                         cur_start, cur_end):
                    continue

            # Diverged-branch filter: skip roads whose start node lies on a
            # fork that diverged from the vehicle's historical path, or roads
            # that originate from an ancestor node but leave toward a diverged
            # branch (end not on the ancestral path).
            if road_start in diverged_starts:
                continue
            if road_start in ancestor_nodes and road_end not in ancestor_nodes \
                    and road_end != cur_start and road_end != cur_end:
                continue

            # Skip roads entirely within the ancestral (already-passed) path.
            if road_start in ancestor_nodes and road_end in ancestor_nodes:
                continue

            try:
                lanes = road_network.graph[road_start][road_end]
            except Exception:
                continue
            if not lanes:
                continue

            # Find the lane of this road nearest to the vehicle laterally.
            # Use the current lane's heading (not the vehicle's) to avoid
            # misselection when the vehicle heading is transiently misaligned
            # (e.g. just after a lane change onto a ramp).
            if current_lane is not None and hasattr(current_lane, 'heading_theta_at'):
                try:
                    _lon = float(current_lane.local_coordinates(vehicle_position[:2])[0])
                    _lane_hdg = float(current_lane.heading_theta_at(_lon))
                except Exception:
                    _lane_hdg = current_pose[2]
            else:
                _lane_hdg = current_pose[2]
            _selection_pose = np.array([current_pose[0], current_pose[1], _lane_hdg],
                                       dtype=np.float64)

            best_lane = None
            best_abs_lateral = float("inf")
            for lane in lanes:
                poly_w = _sample_lane_polyline(lane, num_points=5, longitudinal_step_m=2.0)
                poly_l = _world_to_local_xy(_selection_pose, poly_w)
                lat = float(np.mean(poly_l[:, 1]))
                if abs(lat) < best_abs_lateral:
                    best_abs_lateral = abs(lat)
                    best_lane = lane

            # Road must be laterally separated from the vehicle (not co-directional).
            if best_lane is None or best_abs_lateral < 0.5:
                continue

            # Heading compatibility: the branch lane direction at the point
            # closest to the vehicle must be roughly aligned with the ego
            # lane heading.  Ramp internal roads (diagonal connectors, bends)
            # have significant heading mismatch and are not viable LC targets.
            _ego_lane_heading = float(_selection_pose[2])
            _bl_lon, _ = best_lane.local_coordinates(vehicle_position)
            _bl_lon = max(0.0, min(float(_bl_lon), float(best_lane.length) - 0.1))
            _bl_heading = float(best_lane.heading_theta_at(_bl_lon))
            _heading_diff = abs(_bl_heading - _ego_lane_heading)
            if _heading_diff > math.pi:
                _heading_diff = 2.0 * math.pi - _heading_diff
            if _heading_diff > 0.12:  # ~7 degrees
                continue

            poly_world = _sample_lane_polyline_from_vehicle_position(
                best_lane, vehicle_position, num_points=30, longitudinal_step_m=2.0)
            # The output polyline remains in vehicle-local coordinates for the
            # trajectory generator.  However, geometric FILTER checks (first-pt,
            # backward, monotonicity, divergence, lateral classification) use
            # the lane-heading-aligned frame so they are not distorted by
            # transient vehicle heading misalignment.
            poly_local = _world_to_local_xy(current_pose, poly_world)
            poly_lane = _world_to_local_xy(_selection_pose, poly_world)

            # Skip branches too far ahead (premature) or behind (passed / false-positive).
            if poly_lane[0, 0] > _MAX_BRANCH_FIRST_POINT_X or poly_lane[0, 0] < _MIN_BRANCH_FIRST_POINT_X:
                continue

            # Reject branches whose polyline curves backward: if the last point is
            # behind the vehicle, the branch has already been passed or the road
            # geometry wraps around the vehicle's heading.
            if poly_lane[-1, 0] < 0.0:
                continue

            # Ensure the polyline has broadly forward monotonicity.
            x_vals = poly_lane[:, 0]
            x_cummax = np.maximum.accumulate(x_vals)
            if np.sum(x_vals < x_cummax - 2.0) > len(x_vals) // 2:
                continue

            lateral_hint = float(np.mean(poly_lane[:, 1]))

            if abs(lateral_hint) > _MAX_BRANCH_LATERAL_DISTANCE:
                continue

            # Divergence check: skip branches whose lateral distance from
            # the ego *increases* along the polyline.  Such roads diverge
            # away from the ego's road (e.g. a merge-out / exit ramp viewed
            # from the mainline) and are not reachable by a lane change.
            # Converging or parallel branches have stable or decreasing
            # |lateral|.
            _lats = np.abs(poly_lane[:, 1])
            _quarter = max(len(_lats) // 4, 1)
            _front_lat = float(np.mean(_lats[-_quarter:]))
            _rear_lat = float(np.mean(_lats[:_quarter]))
            if _front_lat > _rear_lat + 2.0:
                continue

            if poly_lane.shape[0] >= 3:
                _diffs = np.diff(poly_lane, axis=0)
                _angles = np.arctan2(_diffs[:, 1], _diffs[:, 0])
                _delta = np.abs(np.diff(_angles))
                _delta = np.minimum(_delta, 2.0 * np.pi - _delta)
                if float(np.max(_delta)) > _MAX_BRANCH_POLYLINE_KINK_RAD:
                    continue

            if lateral_hint >= 0.5 and left_branch is None:
                left_branch = poly_local
            elif lateral_hint <= -0.5 and right_branch is None:
                right_branch = poly_local

            if left_branch is not None and right_branch is not None:
                break  # both found within this block; stop scanning block roads

        if left_branch is not None and right_branch is not None:
            break

    return left_branch, right_branch


@dataclass(frozen=True)
class DynamicObstacle:
    initial_position_xy: np.ndarray
    velocity_xy: np.ndarray
    radius_m: float


def _extract_dynamic_obstacles(vehicle: Any) -> tuple[DynamicObstacle, ...]:
    lidar = getattr(vehicle, "lidar", None)
    if lidar is None:
        return ()
    try:
        objects = lidar.get_surrounding_objects(vehicle)
    except Exception:
        return ()

    obstacles: list[DynamicObstacle] = []
    for obj in objects:
        position = getattr(obj, "position", None)
        if position is None:
            continue
        try:
            local_position = np.asarray(
                vehicle.convert_to_local_coordinates(position, vehicle.position),
                dtype=np.float32,
            )
        except Exception:
            continue
        velocity_world = getattr(obj, "velocity", None)
        if velocity_world is None:
            speed_km_h = getattr(obj, "speed_km_h", 0.0)
            heading_theta = float(getattr(obj, "heading_theta", 0.0))
            speed_mps = float(speed_km_h) / 3.6
            velocity_world = np.asarray(
                [math.cos(heading_theta) * speed_mps, math.sin(heading_theta) * speed_mps],
                dtype=np.float32,
            )
        else:
            velocity_world = np.asarray(velocity_world, dtype=np.float32)
        try:
            local_velocity = np.asarray(
                vehicle.convert_to_local_coordinates(velocity_world, np.asarray([0.0, 0.0], dtype=np.float32)),
                dtype=np.float32,
            )
        except Exception:
            local_velocity = velocity_world.astype(np.float32, copy=False)
        radius_m = 0.5 * math.hypot(
            float(getattr(obj, "LENGTH", getattr(vehicle, "LENGTH", 4.8))),
            float(getattr(obj, "WIDTH", getattr(vehicle, "WIDTH", 2.0))),
        )
        obstacles.append(
            DynamicObstacle(
                initial_position_xy=local_position[:2].astype(np.float32, copy=False),
                velocity_xy=local_velocity[:2].astype(np.float32, copy=False),
                radius_m=radius_m,
            )
        )
    return tuple(obstacles)


@dataclass
class ModeContext:
    ego_speed_mps: float
    ego_heading: float
    current_lane_polyline: np.ndarray
    current_lane_width: float
    left_lane_polyline: Optional[np.ndarray]
    right_lane_polyline: Optional[np.ndarray]
    left_branch_polyline: Optional[np.ndarray]
    right_branch_polyline: Optional[np.ndarray]
    front_object_distance: float
    front_object_speed_mps: float
    left_lane_gap: float
    right_lane_gap: float
    current_ref_lane_count: int
    next_ref_lane_count: int
    has_left_adjacent: bool
    has_right_adjacent: bool
    has_left_branch: bool
    has_right_branch: bool
    dynamic_obstacles: tuple[DynamicObstacle, ...] = ()

    def __post_init__(self) -> None:
        self.current_lane_polyline = _as_numpy_2d(self.current_lane_polyline)
        if self.current_lane_polyline is None:
            raise ValueError("current_lane_polyline must not be None")
        self.left_lane_polyline = _as_numpy_2d(self.left_lane_polyline)
        self.right_lane_polyline = _as_numpy_2d(self.right_lane_polyline)
        self.left_branch_polyline = _as_numpy_2d(self.left_branch_polyline)
        self.right_branch_polyline = _as_numpy_2d(self.right_branch_polyline)

        self.has_left_adjacent = bool(self.has_left_adjacent or self.left_lane_polyline is not None)
        self.has_right_adjacent = bool(self.has_right_adjacent or self.right_lane_polyline is not None)
        self.has_left_branch = bool(self.has_left_branch or self.left_branch_polyline is not None)
        self.has_right_branch = bool(self.has_right_branch or self.right_branch_polyline is not None)


def _none_if_zero(pl: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return None for all-zero polylines (Phase-3 sentinel for absent lane)."""
    if pl is None:
        return None
    if not np.any(pl):
        return None
    return pl


def build_mode_context_from_sample(sample: Dict[str, Any]) -> ModeContext:
    current_lane_polyline = _as_numpy_2d(sample.get("current_lane_polyline"))
    if current_lane_polyline is None:
        current_lane_polyline = _build_fallback_current_lane_polyline(sample)

    left_lane_polyline = _none_if_zero(_as_numpy_2d(sample.get("left_lane_polyline")))
    right_lane_polyline = _none_if_zero(_as_numpy_2d(sample.get("right_lane_polyline")))
    left_branch_polyline = _none_if_zero(_as_numpy_2d(sample.get("left_branch_polyline")))
    right_branch_polyline = _none_if_zero(_as_numpy_2d(sample.get("right_branch_polyline")))

    return ModeContext(
        ego_speed_mps=_as_scalar(sample, "ego_speed_mps", _as_scalar(sample, "ego_speed_km_h", 0.0) / 3.6),
        ego_heading=_as_scalar(sample, "ego_heading", 0.0),
        current_lane_polyline=current_lane_polyline,
        current_lane_width=_as_scalar(sample, "current_lane_width", _as_scalar(sample, "lane_width", 4.0)),
        left_lane_polyline=left_lane_polyline,
        right_lane_polyline=right_lane_polyline,
        left_branch_polyline=left_branch_polyline,
        right_branch_polyline=right_branch_polyline,
        front_object_distance=_as_scalar(sample, "front_object_distance", -1.0),
        front_object_speed_mps=_as_scalar(
            sample,
            "front_object_speed_mps",
            _as_scalar(sample, "front_object_speed_km_h", 0.0) / 3.6,
        ),
        left_lane_gap=_as_scalar(sample, "left_lane_gap", -1.0),
        right_lane_gap=_as_scalar(sample, "right_lane_gap", -1.0),
        current_ref_lane_count=_as_int(sample, "current_ref_lane_count", 1),
        next_ref_lane_count=_as_int(sample, "next_ref_lane_count", 1),
        has_left_adjacent=_as_bool(sample, "has_left_adjacent", left_lane_polyline is not None),
        has_right_adjacent=_as_bool(sample, "has_right_adjacent", right_lane_polyline is not None),
        has_left_branch=_as_bool(sample, "has_left_branch", left_branch_polyline is not None),
        has_right_branch=_as_bool(sample, "has_right_branch", right_branch_polyline is not None),
    )


def build_mode_context_from_vehicle(
    vehicle: Any,
    current_map: Any = None,
    scenario_id: str | None = None,
    local_route: str | None = None,
    ego_main_route_block_ids: Sequence[str] | None = None,
) -> ModeContext:
    navigation = getattr(vehicle, "navigation", None)
    current_ref_lanes = getattr(navigation, "current_ref_lanes", None) if navigation is not None else None
    next_ref_lanes = getattr(navigation, "next_ref_lanes", None) if navigation is not None else None
    if not current_ref_lanes:
        raise ValueError("vehicle.navigation.current_ref_lanes is required to build a ModeContext")
    physical_lane = getattr(vehicle, "lane", None)
    if physical_lane is not None:
        current_lane = physical_lane
    else:
        current_lane = current_ref_lanes[0]
    route_roads = _get_route_roads(current_map, ego_main_route_block_ids)
    current_pose = np.asarray(
        [float(vehicle.position[0]), float(vehicle.position[1]), float(getattr(vehicle, "heading_theta", 0.0))],
        dtype=np.float32,
    )
    candidate_next_lanes = _resolve_candidate_next_lanes(current_lane, current_map, next_ref_lanes, route_roads)
    current_world_polyline = _sample_route_ahead_polyline_from_vehicle_position(
        current_lane,
        np.asarray(vehicle.position, dtype=np.float32),
        candidate_next_lanes,
        current_map=current_map,
    )
    current_lane_polyline = _world_to_local_xy(current_pose, current_world_polyline)
    current_lane_width = float(current_lane.width_at(0.0)) if hasattr(current_lane, "width_at") else float(
        getattr(current_lane, "width", 4.0)
    )

    left_lane_polyline = None
    right_lane_polyline = None
    has_left_adjacent = False
    has_right_adjacent = False

    # Prefer the physical lane's index for road-graph lookups so that when
    # the vehicle has crossed into a service/deceleration lane (whose road key
    # differs from the navigation ref lane), adjacent polylines are built from
    # the correct road rather than from the stale navigation lane_index.
    lane_index = getattr(current_lane, "index", None) or getattr(vehicle, "lane_index", None)
    road_network = getattr(current_map, "road_network", None) if current_map is not None else None
    if lane_index is not None and road_network is not None:
        try:
            road_lanes = road_network.graph[lane_index[0]][lane_index[1]]
            lane_id = int(lane_index[2])
            if lane_id - 1 >= 0:
                left_candidate_next_lanes = _resolve_candidate_next_lanes(
                    road_lanes[lane_id - 1],
                    current_map,
                    next_ref_lanes,
                    route_roads,
                )
                left_lane_polyline = _world_to_local_xy(
                    current_pose,
                    _sample_route_ahead_polyline_from_vehicle_position(
                        road_lanes[lane_id - 1],
                        np.asarray(vehicle.position, dtype=np.float32),
                        left_candidate_next_lanes,
                        current_map=current_map,
                    ),
                )
                has_left_adjacent = True
            if lane_id + 1 < len(road_lanes):
                right_candidate_next_lanes = _resolve_candidate_next_lanes(
                    road_lanes[lane_id + 1],
                    current_map,
                    next_ref_lanes,
                    route_roads,
                )
                right_lane_polyline = _world_to_local_xy(
                    current_pose,
                    _sample_route_ahead_polyline_from_vehicle_position(
                        road_lanes[lane_id + 1],
                        np.asarray(vehicle.position, dtype=np.float32),
                        right_candidate_next_lanes,
                        current_map=current_map,
                    ),
                )
                has_right_adjacent = True
        except Exception:
            pass

    front_object_distance, front_object_speed_mps, left_lane_gap, right_lane_gap = _extract_front_and_lane_gaps(
        vehicle, current_lane
    )
    route_blocks = _get_route_blocks(current_map, ego_main_route_block_ids)
    left_branch_polyline, right_branch_polyline = _resolve_branch_polylines_from_route(
        current_pose,
        current_map,
        route_blocks,
        current_lane=current_lane,
    )

    return ModeContext(
        ego_speed_mps=float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6,
        ego_heading=float(getattr(vehicle, "heading_theta", 0.0)),
        current_lane_polyline=current_lane_polyline,
        current_lane_width=current_lane_width,
        left_lane_polyline=left_lane_polyline,
        right_lane_polyline=right_lane_polyline,
        left_branch_polyline=left_branch_polyline,
        right_branch_polyline=right_branch_polyline,
        front_object_distance=front_object_distance,
        front_object_speed_mps=front_object_speed_mps,
        left_lane_gap=left_lane_gap,
        right_lane_gap=right_lane_gap,
        current_ref_lane_count=len(current_ref_lanes),
        next_ref_lane_count=len(next_ref_lanes) if next_ref_lanes is not None else len(current_ref_lanes),
        has_left_adjacent=has_left_adjacent,
        has_right_adjacent=has_right_adjacent,
        has_left_branch=left_branch_polyline is not None,
        has_right_branch=right_branch_polyline is not None,
        dynamic_obstacles=_extract_dynamic_obstacles(vehicle),
    )
