from __future__ import annotations

import numpy as np

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pgblock.pg_block import PGBlockSocket
from metadrive.component.pgblock.straight import ConnectStraight
from metadrive.component.road_network import Road
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.constants import PGLineType
from metadrive.tests.vis_block.vis_block_base import TestBlock
from metadrive.utils.registry import get_metadrive_class


def _add_socket(
    network: NodeRoadNetwork,
    start_node: str,
    end_node: str,
    start_xy,
    end_xy,
    lane_num: int = 2,
    lane_width: float = 3.5,
):
    road = Road(start_node, end_node)
    neg_road = -road
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    lateral = np.array([direction[1], -direction[0]]) * lane_width

    for lane_idx in range(lane_num):
        offset = lane_idx * lateral
        line_types = (PGLineType.BROKEN, PGLineType.SIDE if lane_idx == lane_num - 1 else PGLineType.BROKEN)
        network.add_lane(road.start_node, road.end_node, StraightLane(start + offset, end + offset, lane_width, line_types))
        network.add_lane(
            neg_road.start_node,
            neg_road.end_node,
            StraightLane(end + offset, start + offset, lane_width, line_types),
        )
    return PGBlockSocket(road, neg_road)


def _add_one_way_socket(
    network: NodeRoadNetwork,
    start_node: str,
    end_node: str,
    start_xy,
    end_xy,
    lane_num: int = 1,
    lane_width: float = 3.5,
):
    socket = _add_socket(network, start_node, end_node, start_xy, end_xy, lane_num=lane_num, lane_width=lane_width)
    socket.is_one_way = True
    return socket


def test_connect_straight_registry_and_block_id():
    assert get_metadrive_class("ConnectStraight") is ConnectStraight
    assert PGBlockDistConfig.get_block("H") is ConnectStraight


def test_connect_straight_builds_bidirectional_link_between_two_dead_ends():
    test = TestBlock(window_type="none")
    try:
        global_network = NodeRoadNetwork()
        socket_a = _add_socket(global_network, "A0", "A1", (0.0, 0.0), (10.0, 0.0))
        socket_b = _add_socket(global_network, "B0", "B1", (30.0, 0.0), (20.0, 0.0))

        block = ConnectStraight(
            1,
            socket_a,
            global_network,
            1,
            secondary_pre_block_socket=socket_b,
        )
        success = block.construct_block(test.render, test.world, {})

        assert success
        assert len(block.get_socket_list()) == 0

        forward_road = Road(socket_a.positive_road.end_node, socket_b.negative_road.start_node)
        reverse_road = Road(socket_b.positive_road.end_node, socket_a.negative_road.start_node)
        forward_lanes = block.block_network.graph[forward_road.start_node][forward_road.end_node]
        reverse_lanes = block.block_network.graph[reverse_road.start_node][reverse_road.end_node]

        assert len(forward_lanes) == 2
        assert len(reverse_lanes) == 2
        assert [(road.start_node, road.end_node) for road in block.get_respawn_roads()] == [
            (forward_road.start_node, forward_road.end_node)
        ]
        assert np.allclose(forward_lanes[0].start, global_network.graph["A0"]["A1"][0].end)
        assert np.allclose(forward_lanes[0].end, global_network.graph["-B1"]["-B0"][0].start)
        assert np.allclose(reverse_lanes[0].start, global_network.graph["B0"]["B1"][0].end)
        assert np.allclose(reverse_lanes[0].end, global_network.graph["-A1"]["-A0"][0].start)
    finally:
        test.close()


def test_connect_straight_builds_one_way_link_between_two_one_way_dead_ends():
    test = TestBlock(window_type="none")
    try:
        global_network = NodeRoadNetwork()
        socket_a = _add_one_way_socket(global_network, "OA0", "OA1", (0.0, 0.0), (10.0, 0.0), lane_num=1)
        socket_b = _add_one_way_socket(global_network, "OB0", "OB1", (30.0, 0.0), (20.0, 0.0), lane_num=1)

        block = ConnectStraight(
            1,
            socket_a,
            global_network,
            1,
            secondary_pre_block_socket=socket_b,
        )
        success = block.construct_block(test.render, test.world, {})

        assert success
        assert len(block.get_socket_list()) == 0

        forward_road = Road(socket_a.positive_road.end_node, socket_b.positive_road.start_node)
        assert forward_road.start_node in block.block_network.graph
        assert forward_road.end_node in block.block_network.graph[forward_road.start_node]
        forward_lanes = block.block_network.graph[forward_road.start_node][forward_road.end_node]
        assert len(forward_lanes) == 1
        assert [(road.start_node, road.end_node) for road in block.get_respawn_roads()] == [
            (forward_road.start_node, forward_road.end_node)
        ]
        assert np.allclose(forward_lanes[0].start, global_network.graph["OA0"]["OA1"][0].end)
        assert np.allclose(forward_lanes[0].end, global_network.graph["OB0"]["OB1"][0].start)
    finally:
        test.close()
