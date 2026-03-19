"""
MAHybridMap + MAHybridPGMapManager
===================================
支持在 env config 中通过字符串/列表指定地图的 block 类型序列。

Config 关键字（在 BaseMultiEnv.default_config() 中声明）：
    use_hybrid_map      (bool)  : 是否启用指定序列地图，默认 False
    hybrid_map_sequence (str | List[str]) :
        block 类型序列，与 MetaDrive 的 map="SXC" 语法相同。
        字符含义：I=FirstBlock, S=Straight, C=Curve, X=Intersection,
                  T=T-Intersection, O=Roundabout, r=OnRamp, R=OffRamp,
                  y=Merge, Y=Split, B=Bidirection, P=ParkingLot, $=Tollgate
        示例："SC"  → Straight + Curve
              "SXS" → Straight + Intersection + Straight

使用方式：
    env = BaseMultiEnv(dict(
        use_hybrid_map=True,
        hybrid_map_sequence="SXC",
    ))
"""

from metadrive.component.algorithm.BIG import BigGenerateMethod, BIG
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.pg_map import PGMap
from metadrive.manager.pg_map_manager import PGMapManager


class MAHybridMap(PGMap):
    """
    PGMap 子类，按 global_config["hybrid_map_sequence"] 指定的 block 序列生成地图。
    生成逻辑与 PGMap._big_generate() 相同，仅将 GENERATE_TYPE 强制设为
    BigGenerateMethod.BLOCK_SEQUENCE。
    """

    def _generate(self):
        parent_node_path = self.engine.worldNP
        physics_world = self.engine.physics_world

        seq = self.engine.global_config.get("hybrid_map_sequence", "S")

        big_map = BIG(
            self._config.get(self.LANE_NUM, 2),
            self._config.get(self.LANE_WIDTH, 3.5),
            self.road_network,
            parent_node_path,
            physics_world,
            exit_length=self._config.get("exit_length", 50),
            random_seed=self.engine.global_random_seed,
            block_dist_config=self.engine.global_config.get("block_dist_config", PGBlockDistConfig),
        )
        big_map.generate(BigGenerateMethod.BLOCK_SEQUENCE, seq)
        self.blocks = big_map.blocks
        big_map.destroy()
        self.road_network.after_init()


class MAHybridPGMapManager(PGMapManager):
    """
    PGMapManager 子类：
      - use_hybrid_map=True  → 每个 seed 生成 MAHybridMap（固定 block 序列）
      - use_hybrid_map=False → 完全走父类逻辑（与原 PGMapManager 行为一致）
    """

    def reset(self):
        if not self.engine.global_config.get("use_hybrid_map", False):
            # 未启用，走默认 PGMap 逻辑
            super().reset()
            return

        config = self.engine.global_config
        current_seed = self.engine.global_seed

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
