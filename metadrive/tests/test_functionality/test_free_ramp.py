from __future__ import annotations

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.ramp import FreeInRampOnStraight, FreeOutRampOnStraight
from metadrive.component.pgblock.straight import OneWayStraight
from metadrive.component.pgblock.curve import OneWayCurve
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock
from metadrive.utils.registry import get_metadrive_class


def test_free_ramp_registry_and_block_ids():
    assert get_metadrive_class("FreeInRampOnStraight") is FreeInRampOnStraight
    assert get_metadrive_class("FreeOutRampOnStraight") is FreeOutRampOnStraight
    assert get_metadrive_class("OneWayStraight") is OneWayStraight
    assert get_metadrive_class("OneWayCurve") is OneWayCurve
    assert PGBlockDistConfig.get_block("g") is FreeInRampOnStraight
    assert PGBlockDistConfig.get_block("G") is FreeOutRampOnStraight
    assert PGBlockDistConfig.get_block("s") is OneWayStraight
    assert PGBlockDistConfig.get_block("c") is OneWayCurve


def test_free_in_ramp_exposes_branch_socket_with_single_lane():
    test = TestBlock(window_type="none")
    try:
        global_network = NodeRoadNetwork()
        first = FirstPGBlock(global_network, 3.0, 2, test.render, test.world, 30)
        block = FreeInRampOnStraight(1, first.get_socket(0), global_network, 1)
        success = block.construct_block(test.render, test.world, {"length": 30, "extension_length": 20})

        assert success
        assert len(block.get_socket_list()) == 2
        assert len(block.get_socket(0).get_positive_lanes(block.block_network)) == 2
        assert len(block.get_socket(1).get_positive_lanes(block.block_network)) == 1
        assert getattr(block.get_socket(0), "is_one_way", False) is False
        assert getattr(block.get_socket(1), "is_one_way", False) is True
    finally:
        test.close()


def test_free_out_ramp_exposes_branch_socket_with_single_lane():
    test = TestBlock(window_type="none")
    try:
        global_network = NodeRoadNetwork()
        first = FirstPGBlock(global_network, 3.0, 2, test.render, test.world, 30)
        block = FreeOutRampOnStraight(1, first.get_socket(0), global_network, 1)
        success = block.construct_block(test.render, test.world, {"length": 30, "extension_length": 20})

        assert success
        assert len(block.get_socket_list()) == 2
        assert len(block.get_socket(0).get_positive_lanes(block.block_network)) == 2
        assert len(block.get_socket(1).get_positive_lanes(block.block_network)) == 1
        assert getattr(block.get_socket(0), "is_one_way", False) is False
        assert getattr(block.get_socket(1), "is_one_way", False) is True
    finally:
        test.close()


def test_one_way_straight_and_curve_only_create_positive_lanes():
    test = TestBlock(window_type="none")
    try:
        global_network = NodeRoadNetwork()
        first = FirstPGBlock(global_network, 3.0, 2, test.render, test.world, 30)
        one_way_straight = OneWayStraight(1, first.get_socket(0), global_network, 1)
        assert one_way_straight.construct_block(test.render, test.world, {"length": 30})
        assert len(one_way_straight.get_socket(0).get_positive_lanes(one_way_straight.block_network)) == 2
        assert getattr(one_way_straight.get_socket(0), "is_one_way", False) is True
        assert one_way_straight.remove_negative_lanes is True

        one_way_curve = OneWayCurve(2, one_way_straight.get_socket(0), global_network, 1)
        assert one_way_curve.construct_block(
            test.render, test.world, {"length": 40, "radius": 35, "angle": 45, "dir": 1}
        )
        assert len(one_way_curve.get_socket(0).get_positive_lanes(one_way_curve.block_network)) == 2
        assert getattr(one_way_curve.get_socket(0), "is_one_way", False) is True
        assert one_way_curve.remove_negative_lanes is True
    finally:
        test.close()
