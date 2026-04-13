from __future__ import annotations

import copy
import types

import pytest

from metadrive.component.map.pg_map import PGMap


class _FakeSocket:
    def __init__(self, index: int, is_one_way: bool = False):
        self.index = f"socket{index}"
        self.is_one_way = is_one_way


class _FakeBlock:
    ID = "S"
    IS_ONE_WAY_BLOCK = False

    def __init__(self, block_index, pre_block_socket, global_network, random_seed, ignore_intersection_checking=True):
        self.block_index = block_index
        self.pre_block_socket = pre_block_socket
        self.global_network = global_network
        self.random_seed = random_seed
        self.ignore_intersection_checking = ignore_intersection_checking
        self.construct_config = None
        self.sockets = [_FakeSocket(0, is_one_way=self.IS_ONE_WAY_BLOCK), _FakeSocket(1, is_one_way=self.IS_ONE_WAY_BLOCK)]

    def construct_from_config(self, config, parent_node_path, physics_world):
        self.construct_config = copy.deepcopy(config)

    def get_socket(self, index):
        if index < 0 or index >= len(self.sockets):
            raise ValueError("socket out of range")
        return self.sockets[index]


class _FakeFirstBlock(_FakeBlock):
    def __init__(
        self,
        global_network,
        lane_width,
        lane_num,
        render_root_np,
        physics_world,
        length,
        start_point,
        ignore_intersection_checking=True,
    ):
        super().__init__(
            block_index=0,
            pre_block_socket=_FakeSocket(0),
            global_network=global_network,
            random_seed=None,
            ignore_intersection_checking=ignore_intersection_checking,
        )


class _FakeOneWayBlock(_FakeBlock):
    ID = "s"
    IS_ONE_WAY_BLOCK = True


class _FakeConnectBlock(_FakeBlock):
    ID = "H"

    def __init__(
        self,
        block_index,
        pre_block_socket,
        global_network,
        random_seed,
        secondary_pre_block_socket,
        ignore_intersection_checking=True,
    ):
        super().__init__(block_index, pre_block_socket, global_network, random_seed, ignore_intersection_checking)
        self.secondary_pre_block_socket = secondary_pre_block_socket


def _build_pg_map(monkeypatch):
    current_map = PGMap.__new__(PGMap)
    current_map.road_network = types.SimpleNamespace(graph={})
    current_map.blocks = []
    current_map._config = {"lane_num": 2, "lane_width": 3.5, "exit_length": 50, "start_position": [0, 0]}
    def _get_block(block_id):
        if block_id == "H":
            return _FakeConnectBlock
        return _FakeOneWayBlock if block_id in {"s", "c"} else _FakeBlock

    fake_engine = types.SimpleNamespace(
        global_random_seed=7,
        global_config={"block_dist_config": types.SimpleNamespace(get_block=_get_block)},
    )
    monkeypatch.setattr("metadrive.engine.engine_utils.get_engine", lambda: fake_engine)
    monkeypatch.setattr("metadrive.component.map.pg_map.FirstPGBlock", _FakeFirstBlock)
    return current_map


def test_config_generate_supports_branching_graph(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "g0", "id": "G", "parent_block_id": "s0", "parent_socket_index": 0, "length": 100},
        {"block_id": "s_main", "id": "S", "parent_block_id": "g0", "parent_socket_index": 0, "length": 120},
        {"block_id": "s_ramp", "id": "S", "parent_block_id": "g0", "parent_socket_index": 1, "length": 60},
    ]

    current_map._config_generate(blocks_config, object(), object())

    assert len(current_map.blocks) == 5
    assert current_map.blocks[0].graph_block_id == "root"
    assert current_map.blocks[2].graph_block_id == "g0"
    assert current_map.blocks[3].graph_parent_block_id == "g0"
    assert current_map.blocks[3].graph_parent_socket_index == 0
    assert current_map.blocks[4].graph_parent_block_id == "g0"
    assert current_map.blocks[4].graph_parent_socket_index == 1


def test_config_generate_rejects_duplicate_parent_socket(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "a", "id": "S", "parent_block_id": "s0", "parent_socket_index": 0, "length": 100},
        {"block_id": "b", "id": "S", "parent_block_id": "s0", "parent_socket_index": 0, "length": 120},
    ]

    with pytest.raises(ValueError, match="Parent socket already occupied"):
        current_map._config_generate(blocks_config, object(), object())


def test_config_generate_requires_parent_before_child(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "child", "id": "S", "parent_block_id": "missing", "parent_socket_index": 0, "length": 80},
    ]

    with pytest.raises(ValueError, match="Unknown parent_block_id"):
        current_map._config_generate(blocks_config, object(), object())


def test_config_generate_rejects_bidirectional_child_after_one_way_socket(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "s0", "id": "s", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "bad", "id": "S", "parent_block_id": "s0", "parent_socket_index": 0, "length": 80},
    ]

    with pytest.raises(ValueError, match="one-way socket"):
        current_map._config_generate(blocks_config, object(), object())


def test_config_generate_allows_one_way_child_after_one_way_socket(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "s0", "id": "s", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "c0", "id": "c", "parent_block_id": "s0", "parent_socket_index": 0, "length": 50, "radius": 45, "angle": 90, "dir": 1},
    ]

    current_map._config_generate(blocks_config, object(), object())
    assert current_map.blocks[1].graph_block_id == "s0"
    assert current_map.blocks[2].graph_block_id == "c0"


def test_config_generate_allows_connect_straight_between_two_one_way_sockets(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "s0", "id": "s", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "c0", "id": "c", "parent_block_id": "s0", "parent_socket_index": 0, "length": 50, "radius": 45, "angle": 90, "dir": 1},
        {
            "block_id": "h0",
            "id": "H",
            "parent_block_id": "c0",
            "parent_socket_index": 0,
            "secondary_parent_block_id": "root",
            "secondary_parent_socket_index": 0,
        },
    ]

    current_map._config_generate(blocks_config, object(), object())
    assert current_map.blocks[-1].graph_block_id == "h0"


def test_config_generate_requires_secondary_parent_for_connect_straight(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "cs0", "id": "H", "parent_block_id": "s0", "parent_socket_index": 0},
    ]

    with pytest.raises(ValueError, match="ConnectStraight requires secondary_parent_block_id"):
        current_map._config_generate(blocks_config, object(), object())


def test_config_generate_rejects_secondary_parent_for_regular_block(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {
            "block_id": "bad",
            "id": "S",
            "parent_block_id": "s0",
            "parent_socket_index": 0,
            "secondary_parent_block_id": "root",
            "secondary_parent_socket_index": 0,
            "length": 80,
        },
    ]

    with pytest.raises(ValueError, match="Only ConnectStraight may use secondary parent socket fields"):
        current_map._config_generate(blocks_config, object(), object())


def test_config_generate_constructs_connect_straight_with_two_parents(monkeypatch):
    current_map = _build_pg_map(monkeypatch)
    blocks_config = [
        {"block_id": "a0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "b0", "id": "S", "parent_block_id": "a0", "parent_socket_index": 1, "length": 60},
        {
            "block_id": "cs0",
            "id": "H",
            "parent_block_id": "a0",
            "parent_socket_index": 0,
            "secondary_parent_block_id": "b0",
            "secondary_parent_socket_index": 0,
        },
    ]

    current_map._config_generate(blocks_config, object(), object())
    connect_block = current_map.blocks[-1]
    assert connect_block.graph_secondary_parent_block_id == "b0"
    assert connect_block.graph_secondary_parent_socket_index == 0
    assert connect_block.secondary_pre_block_socket is current_map.blocks[2].get_socket(0)
