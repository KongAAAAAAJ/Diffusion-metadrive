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

import numpy as np

from metadrive.component.lane.junction_lane import (
    build_lane_seam_drivable_surface,
)
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
        overlays = tuple(
            surface
            for block in self.blocks
            for surface in tuple(
                getattr(block, "junction_drivable_surfaces", ()) or ()
            )
        )
        overlays += self._build_equivalent_incoming_lane_surfaces()
        self.drivable_geometry_overlays = overlays
        self.road_network.drivable_geometry_overlays = overlays
        self.road_network.after_init()

    def _build_equivalent_incoming_lane_surfaces(self):
        """Build seams for navigation-equivalent incoming lanes.

        A FreeInRamp block builds its merge bend before a later block connects
        the external ramp route to the same road node.  Both incoming lanes
        end at the same pose, but only the internal bend initially owns the
        junction surface.  Navigation can legitimately select the external
        lane.  Its final tangent can differ slightly from the internal bend,
        so build a matching surface instead of attaching geometrically stale
        surface coordinates.
        """

        lanes = tuple(self.road_network.get_all_lanes())
        owners = tuple(
            lane
            for lane in lanes
            if tuple(
                getattr(lane, "junction_drivable_surfaces", ()) or ()
            )
        )
        created = []
        created_keys = set()
        for owner in owners:
            surfaces = tuple(owner.junction_drivable_surfaces)
            owner_index = tuple(getattr(owner, "index", ()) or ())
            if len(owner_index) < 3:
                continue
            successors = tuple(
                lane
                for lane in lanes
                if lane is not owner
                and tuple(getattr(lane, "index", ()) or ())[:1]
                == owner_index[1:2]
                and any(
                    surface
                    in tuple(
                        getattr(
                            lane, "junction_drivable_surfaces", ()
                        )
                        or ()
                    )
                    for surface in surfaces
                )
            )
            if not successors:
                continue
            owner_length = float(owner.length)
            owner_end = np.asarray(
                owner.position(owner_length, 0.0)[:2], dtype=np.float64
            )
            owner_heading = float(owner.heading_theta_at(owner_length))
            for lane in lanes:
                lane_length = float(getattr(lane, "length", 0.0) or 0.0)
                lane_index = tuple(getattr(lane, "index", ()) or ())
                if (
                    lane is owner
                    or lane_length <= 0.0
                    or len(lane_index) < 3
                    or lane_index[1] != owner_index[1]
                ):
                    continue
                lane_end = np.asarray(
                    lane.position(lane_length, 0.0)[:2], dtype=np.float64
                )
                lane_heading = float(lane.heading_theta_at(lane_length))
                heading_error = float(
                    np.arctan2(
                        np.sin(lane_heading - owner_heading),
                        np.cos(lane_heading - owner_heading),
                    )
                )
                if (
                    float(np.linalg.norm(lane_end - owner_end)) <= 0.05
                    and abs(heading_error) <= 0.2
                ):
                    for successor in successors:
                        successor_index = tuple(
                            getattr(successor, "index", ()) or ()
                        )
                        key = (lane_index, successor_index)
                        if key in created_keys:
                            continue
                        seam_transition_m = 16.0
                        # The offset ramp-to-mainline seam is an unstructured
                        # merge apron, not a 4 m routing lane.  At the bend an
                        # XL vehicle's corner sweeps about 2.25 m from the seam
                        # centre even while its reference path is centred.
                        # Represent the physical apron at its 5 m width so the
                        # footprint audit does not turn a valid merge into a
                        # permanent stop.  This changes only the map surface;
                        # collision gaps and kinematic limits stay unchanged.
                        surface = build_lane_seam_drivable_surface(
                            lane,
                            successor,
                            transition_m=seam_transition_m,
                            width_m=5.0,
                        )
                        surface.index = (
                            f"{lane_index[1]}-route-merge-junction",
                            f"{lane_index[1]}-route-merge-surface",
                            int(lane_index[2]),
                        )
                        existing = tuple(
                            getattr(
                                lane, "junction_drivable_surfaces", ()
                            )
                            or ()
                        )
                        lane.junction_drivable_surfaces = existing + (
                            surface,
                        )
                        lane.route_seam_transition_m = seam_transition_m
                        successor_existing = tuple(
                            getattr(
                                successor,
                                "junction_drivable_surfaces",
                                (),
                            )
                            or ()
                        )
                        successor.junction_drivable_surfaces = (
                            successor_existing + (surface,)
                        )
                        successor.route_seam_transition_m = (
                            seam_transition_m
                        )
                        created.append(surface)
                        created_keys.add(key)
        return tuple(created)


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
