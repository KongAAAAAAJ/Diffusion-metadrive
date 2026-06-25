from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_custom_hybrid_map_module():
    module_name = "custom_hybrid_map_blocks_config_test"
    path = REPO_ROOT / "envs/diffusion_envs/custom_hybrid_map.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    big_module = types.ModuleType("metadrive.component.algorithm.BIG")
    big_module.BigGenerateMethod = type("BigGenerateMethod", (), {"BLOCK_SEQUENCE": "block_sequence"})
    big_module.BIG = type("BIG", (), {})
    register("metadrive.component.algorithm.BIG", big_module)

    dist_module = types.ModuleType("metadrive.component.algorithm.blocks_prob_dist")
    dist_module.PGBlockDistConfig = type("PGBlockDistConfig", (), {})
    register("metadrive.component.algorithm.blocks_prob_dist", dist_module)

    base_map_module = types.ModuleType("metadrive.component.map.base_map")
    base_map_module.BaseMap = type(
        "BaseMap",
        (),
        {
            "BLOCK_ID": "id",
            "GRAPH_BLOCK_ID": "block_id",
            "PARENT_BLOCK_ID": "parent_block_id",
            "PARENT_SOCKET_INDEX": "parent_socket_index",
            "PRE_BLOCK_SOCKET_INDEX": "pre_block_socket_index",
            "LANE_NUM": "lane_num",
            "LANE_WIDTH": "lane_width",
        },
    )
    register("metadrive.component.map.base_map", base_map_module)

    pg_map_module = types.ModuleType("metadrive.component.map.pg_map")
    pg_map_module.PGMap = type("PGMap", (), {})
    register("metadrive.component.map.pg_map", pg_map_module)

    first_block_module = types.ModuleType("metadrive.component.pgblock.first_block")
    first_block_module.FirstPGBlock = type("FirstPGBlock", (), {"ID": "I"})
    register("metadrive.component.pgblock.first_block", first_block_module)

    pg_map_manager_module = types.ModuleType("metadrive.manager.pg_map_manager")
    pg_map_manager_module.PGMapManager = type("PGMapManager", (), {})
    register("metadrive.manager.pg_map_manager", pg_map_manager_module)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in stubbed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def test_normalize_blocks_config_keeps_graph_config_shape():
    module = _load_custom_hybrid_map_module()
    blocks_config = [{"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 80}]

    normalized = module.MAHybridMap._normalize_blocks_config(blocks_config)

    assert normalized == blocks_config


def test_generate_prefers_blocks_config_over_sequence():
    module = _load_custom_hybrid_map_module()
    current_map = module.MAHybridMap.__new__(module.MAHybridMap)
    current_map.engine = types.SimpleNamespace(
        worldNP=object(),
        physics_world=object(),
        global_config={
            "hybrid_map_blocks_config": [{"block_id": "s0", "id": "S", "parent_block_id": "root", "parent_socket_index": 0, "length": 80}],
        },
        global_random_seed=7,
    )
    current_map.road_network = types.SimpleNamespace(after_init=lambda: None)
    current_map._config = {"lane_num": 2, "lane_width": 3.5, "exit_length": 50}
    current_map.blocks = []

    calls = {}

    def fake_config_generate(blocks_config, parent_node_path, physics_world):
        calls["blocks_config"] = blocks_config
        calls["parent_node_path"] = parent_node_path
        calls["physics_world"] = physics_world

    current_map._config_generate = fake_config_generate

    current_map._generate()

    assert calls["blocks_config"][0]["id"] == "S"
    assert calls["blocks_config"][0]["block_id"] == "s0"
    assert calls["parent_node_path"] is current_map.engine.worldNP
    assert calls["physics_world"] is current_map.engine.physics_world
