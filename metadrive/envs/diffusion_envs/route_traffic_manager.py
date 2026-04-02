from __future__ import annotations

import math

from metadrive.envs.diffusion_envs.traffic_manager import CustomTrafficManager


MAX_VEHICLE_LENGTH = 10.0
MAX_VEHICLE_WIDTH = 2.5


class RouteAwareTrafficManager(CustomTrafficManager):
    """Traffic manager that avoids spawning background cars near the ego spawn zone."""

    def _iter_ego_spawn_zones(self):
        spawn_manager = getattr(self.engine, "spawn_manager", None)
        return list(getattr(spawn_manager, "ego_spawn_zones", []) or [])

    def _conflicts_with_ego_spawn(self, vehicle_config) -> bool:
        lane_index = tuple(vehicle_config["spawn_lane_index"])
        spawn_road = tuple(lane_index[:2])
        spawn_lane = lane_index
        spawn_longitude = float(vehicle_config["spawn_longitude"])
        spawn_lateral = float(vehicle_config.get("spawn_lateral", 0.0))
        lane_relaxation = bool(self.engine.global_config.get("traffic_spawn_lane_relaxation", True))
        min_gap_ahead = float(self.engine.global_config.get("traffic_spawn_min_gap_ahead", 12.0))
        min_gap_behind = float(self.engine.global_config.get("traffic_spawn_min_gap_behind", 8.0))

        for zone in self._iter_ego_spawn_zones():
            if tuple(zone.get("road", ())) != spawn_road:
                continue

            zone_lane = tuple(zone.get("spawn_lane_index", ()))
            same_lane = (zone_lane == spawn_lane) if zone_lane else True
            if (not same_lane) and lane_relaxation:
                continue

            ego_spawn_longitude = float(zone["spawn_longitude"])
            longitudinal_delta = spawn_longitude - ego_spawn_longitude
            if longitudinal_delta >= 0.0:
                longitudinal_conflict = longitudinal_delta < min_gap_ahead
            else:
                longitudinal_conflict = abs(longitudinal_delta) < min_gap_behind

            if not longitudinal_conflict:
                continue

            if same_lane:
                return True

            zone_lateral = float(zone.get("spawn_lateral", 0.0))
            if abs(spawn_lateral - zone_lateral) < self._lane_width_for(lane_index):
                return True
        return False

    def _lane_width_for(self, lane_index) -> float:
        current_map = getattr(self.engine, "current_map", None)
        if current_map is not None:
            lane = current_map.road_network.get_lane(lane_index)
            width = getattr(lane, "width", None)
            if width is not None:
                return float(width)
        return MAX_VEHICLE_WIDTH

    def _resolve_vehicle_dimensions(self, config, vehicle_config_key):
        model = str(self.engine.global_config.get(vehicle_config_key, {}).get("vehicle_model", "default"))
        dimensions = {
            "s": (4.3, 1.7),
            "m": (4.6, 1.85),
            "l": (4.87, 2.046),
            "xl": (5.74, 2.3),
            "default": (4.515, 1.852),
            "static_default": (4.515, 1.852),
        }
        length, width = dimensions.get(model, (MAX_VEHICLE_LENGTH, MAX_VEHICLE_WIDTH))
        width = float(config.get("width", width) or width)
        length = float(config.get("length", length) or length)
        return length, width

    def _passes_final_spawn_guard(self, vehicle_config) -> bool:
        current_map = getattr(self.engine, "current_map", None)
        if current_map is None:
            return True

        lane_index = tuple(vehicle_config["spawn_lane_index"])
        lane = current_map.road_network.get_lane(lane_index)
        candidate_longitude = float(vehicle_config["spawn_longitude"])
        candidate_lateral = float(vehicle_config.get("spawn_lateral", 0.0))
        candidate_position = lane.position(candidate_longitude, candidate_lateral)
        traffic_length, traffic_width = self._resolve_vehicle_dimensions(vehicle_config, "traffic_vehicle_config")

        for zone in self._iter_ego_spawn_zones():
            zone_lane_index = tuple(zone.get("spawn_lane_index", ()))
            zone_lane = current_map.road_network.get_lane(zone_lane_index)
            zone_longitude = float(zone["spawn_longitude"])
            zone_lateral = float(zone.get("spawn_lateral", 0.0))
            zone_position = zone_lane.position(zone_longitude, zone_lateral)
            ego_length, ego_width = self._resolve_vehicle_dimensions(zone, "vehicle_config")

            longitudinal_overlap = abs(float(candidate_position[0]) - float(zone_position[0])) < (
                traffic_length + ego_length
            ) / 2.0
            lateral_overlap = abs(float(candidate_position[1]) - float(zone_position[1])) < (
                traffic_width + ego_width
            ) / 2.0
            if longitudinal_overlap and lateral_overlap:
                return False
        return True

    def _spawn_traffic_vehicle_if_safe(self, vehicle_type, traffic_v_config):
        if self._conflicts_with_ego_spawn(traffic_v_config):
            return None
        if not self._passes_final_spawn_guard(traffic_v_config):
            return None

        traffic_v_config = dict(traffic_v_config)
        traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
        random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
        from metadrive.policy.idm_policy import IDMPolicy
        self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
        self._traffic_vehicles.append(random_v)
        return random_v

    def _create_basic_vehicles(self, map, traffic_density: float):
        for lane in self.respawn_lanes:
            total_num = int(lane.length / self.VEHICLE_GAP)
            vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
            self.np_random.shuffle(vehicle_longs)
            target_longs = vehicle_longs[:int(math.ceil(traffic_density * len(vehicle_longs)))]
            for long in target_longs:
                traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long}
                vehicle_type = self.random_vehicle_type()
                self._spawn_traffic_vehicle_if_safe(vehicle_type, traffic_v_config)

    def _create_trigger_vehicles(self, map, traffic_density: float) -> None:
        vehicle_num = 0
        for block in map.blocks[1:]:
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for lane in lanes:
                    if hasattr(self.engine, "object_manager") and lane in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(lane)

            potential_vehicle_configs = [
                config for config in potential_vehicle_configs
                if not self._conflicts_with_ego_spawn(config)
            ]

            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]

            from metadrive.policy.idm_policy import IDMPolicy
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                random_v = self._spawn_traffic_vehicle_if_safe(vehicle_type, v_config)
                if random_v is not None:
                    vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            from metadrive.manager.traffic_manager import BlockVehicles
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def after_step(self, *args, **kwargs):
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                v_to_remove.append(v)

        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)

            if self.mode in {"respawn", "hybrid"}:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                self._spawn_traffic_vehicle_if_safe(vehicle_type, traffic_v_config)

        return {}
