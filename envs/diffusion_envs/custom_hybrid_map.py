"""
MAHybridMap + MAHybridPGMapManager
===================================
支持在 env config 中通过 block 配置列表指定固定地图。

Config 关键字（在 BaseMultiEnv.default_config() 中声明）：
    use_hybrid_map         (bool)  : 是否启用固定地图入口，默认 False
    hybrid_map_blocks_config (List[dict] | None) :
        按显式父节点 + 父 socket 的 block 图配置生成地图。
        每项至少包含：
            - block_id
            - id
            - parent_block_id
            - parent_socket_index
        对于 ConnectStraight(H) 还需要：
            - secondary_parent_block_id
            - secondary_parent_socket_index
        其余字段按 block 类型附加，例如：
            - Straight / OneWayStraight: length
            - Curve / OneWayCurve: length, radius, angle, dir
        其中 s/c 表示 one-way branch block，适合接在 FreeRamp 的匝道支路后；
        H 表示连接两段 straight 断头路的双父特例 block。
"""

import copy

from metadrive.component.map.pg_map import PGMap
from metadrive.manager.pg_map_manager import PGMapManager


class MAHybridMap(PGMap):
    """
    PGMap 子类，固定走 PGMap._config_generate()。
    """

    @staticmethod
    def _normalize_blocks_config(blocks_config):
        if not blocks_config:
            return None

        return [copy.deepcopy(dict(block)) for block in blocks_config]

    def _generate(self):
        parent_node_path = self.engine.worldNP
        physics_world = self.engine.physics_world
        blocks_config = self._normalize_blocks_config(self.engine.global_config.get("hybrid_map_blocks_config"))

        if blocks_config is None:
            raise ValueError("hybrid_map_blocks_config is required when use_hybrid_map=True")

        self._config_generate(blocks_config, parent_node_path, physics_world)
        self.road_network.after_init()


class MAHybridPGMapManager(PGMapManager):
    """
    PGMapManager 子类：
      - use_hybrid_map=True  → 每个 seed 生成 MAHybridMap（固定 block 配置列表）
      - use_hybrid_map=False → 完全走父类逻辑（与原 PGMapManager 行为一致）
    """

    def reset(self):
        if not self.engine.global_config.get("use_hybrid_map", False):
            # 未启用，走默认 PGMap 逻辑
            super().reset()
            return

        config = self.engine.global_config
        current_seed = self.engine.global_seed
        # Long-running collection rebinds its single-scenario window to the
        # explicit episode seed before reset.  PGMapManager preallocates only
        # the construction-time key, so admit the newly bound deterministic
        # seed instead of depending on episode execution order.
        self.maps.setdefault(current_seed, None)

        if self.maps[current_seed] is None:
            map_config = config["map_config"].copy(unchangeable=False)
            map_config.update({"seed": current_seed})
            map_config = self.add_random_to_map(map_config)
            new_map = self.spawn_object(MAHybridMap, map_config=map_config, random_seed=None)
            self.current_map = new_map
            if config["store_map"]:
                self.maps[current_seed] = new_map
        else:
            self.load_map(self.maps[current_seed])
