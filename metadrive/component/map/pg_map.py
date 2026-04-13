import copy
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.type import MetaDriveType
from metadrive.constants import PGLineType, PGLineColor
from typing import List

from panda3d.core import NodePath

from metadrive.component.algorithm.BIG import BigGenerateMethod, BIG
from metadrive.component.map.base_map import BaseMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.constants import Decoration
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.utils import Config


def parse_map_config(easy_map_config, new_map_config, default_config):
    assert isinstance(new_map_config, Config)
    assert isinstance(default_config, Config)

    # Return the user specified config if overwritten
    if easy_map_config is None or not default_config["map_config"].is_identical(new_map_config):
        new_map_config = default_config["map_config"].copy(unchangeable=False).update(new_map_config)
        assert default_config["map"] == easy_map_config
        return new_map_config

    if isinstance(easy_map_config, int):
        new_map_config[BaseMap.GENERATE_TYPE] = BigGenerateMethod.BLOCK_NUM
    elif isinstance(easy_map_config, str):
        new_map_config[BaseMap.GENERATE_TYPE] = BigGenerateMethod.BLOCK_SEQUENCE
    else:
        raise ValueError(
            "Unkown easy map config: {} and original map config: {}".format(easy_map_config, new_map_config)
        )
    new_map_config[BaseMap.GENERATE_CONFIG] = easy_map_config
    return new_map_config


class MapGenerateMethod:
    BIG_BLOCK_NUM = BigGenerateMethod.BLOCK_NUM
    BIG_BLOCK_SEQUENCE = BigGenerateMethod.BLOCK_SEQUENCE
    BIG_SINGLE_BLOCK = BigGenerateMethod.SINGLE_BLOCK
    PG_MAP_FILE = "pg_map_file"


class PGMap(BaseMap):
    def _generate(self):
        """
        We can override this function to introduce other methods!
        """
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        generate_type = self._config[self.GENERATE_TYPE]
        if generate_type == BigGenerateMethod.BLOCK_NUM or generate_type == BigGenerateMethod.BLOCK_SEQUENCE:
            self._big_generate(parent_node_path, physics_world)

        elif generate_type == MapGenerateMethod.PG_MAP_FILE:
            # other config such as lane width, num and seed will be valid, since they will be read from file
            blocks_config = self._config[self.GENERATE_CONFIG]
            self._config_generate(blocks_config, parent_node_path, physics_world)
        else:
            raise ValueError("Map can not be created by {}".format(generate_type))
        self.road_network.after_init()

    def _big_generate(self, parent_node_path: NodePath, physics_world: PhysicsWorld):
        big_map = BIG(
            self._config.get(self.LANE_NUM, 2),
            self._config.get(self.LANE_WIDTH, 3.5),
            self.road_network,
            parent_node_path,
            physics_world,
            # self._config["block_type_version"],
            exit_length=self._config.get("exit_length", 50),
            random_seed=self.engine.global_random_seed,
            block_dist_config=self.engine.global_config.get("block_dist_config", PGBlockDistConfig)
        )
        big_map.generate(self._config[self.GENERATE_TYPE], self._config[self.GENERATE_CONFIG])
        self.blocks = big_map.blocks
        big_map.destroy()

    def _config_generate(self, blocks_config: List, parent_node_path: NodePath, physics_world: PhysicsWorld):
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"
        root_block = FirstPGBlock(
            global_network=self.road_network,
            lane_width=self._config.get(self.LANE_WIDTH, 3.5),
            lane_num=self._config.get(self.LANE_NUM, 2),
            render_root_np=parent_node_path,
            physics_world=physics_world,
            length=self._config.get("exit_length", 50),
            start_point=self._config.get("start_position", [0, 0]),
            ignore_intersection_checking=True
        )
        root_block.graph_block_id = "root"
        root_block.graph_parent_block_id = None
        root_block.graph_parent_socket_index = None
        self.blocks.append(root_block)

        created_blocks = {"root": root_block}
        used_parent_sockets = set()
        for block_index, raw_block in enumerate(blocks_config, 1):
            b = copy.deepcopy(raw_block)
            block_graph_id = b.pop(self.GRAPH_BLOCK_ID)
            if block_graph_id == "root":
                raise ValueError("'root' is reserved for the implicit FirstPGBlock")
            if block_graph_id in created_blocks:
                raise ValueError(f"Duplicate block_id in map config: {block_graph_id}")

            parent_block_id = b.pop(self.PARENT_BLOCK_ID)
            if parent_block_id not in created_blocks:
                raise ValueError(
                    f"Unknown parent_block_id '{parent_block_id}' for block '{block_graph_id}'. "
                    "Configs must be ordered parent-before-child."
                )

            parent_socket_index = b.pop(self.PARENT_SOCKET_INDEX)
            if not isinstance(parent_socket_index, int):
                raise ValueError(
                    f"parent_socket_index must be int for block '{block_graph_id}', got {parent_socket_index!r}"
                )

            prospective_block_type_id = b[self.BLOCK_ID]
            if prospective_block_type_id != "H" and (parent_block_id, parent_socket_index) in used_parent_sockets:
                raise ValueError(
                    f"Parent socket already occupied: parent_block_id='{parent_block_id}', "
                    f"parent_socket_index={parent_socket_index}"
                )

            parent_block = created_blocks[parent_block_id]
            block_type_id = b.pop(self.BLOCK_ID)
            block_type = self.engine.global_config["block_dist_config"].get_block(block_type_id)
            parent_socket = parent_block.get_socket(parent_socket_index)
            if (
                getattr(parent_socket, "is_one_way", False)
                and not getattr(block_type, "IS_ONE_WAY_BLOCK", False)
                and block_type_id != "H"
            ):
                raise ValueError(
                    f"Cannot attach bidirectional block '{block_type.ID}' to one-way socket "
                    f"of parent '{parent_block_id}' (socket {parent_socket_index})"
                )
            secondary_parent_block_id = b.pop(self.SECONDARY_PARENT_BLOCK_ID, None)
            secondary_parent_socket_index = b.pop(self.SECONDARY_PARENT_SOCKET_INDEX, None)
            if block_type_id == "H":
                if secondary_parent_block_id is None or secondary_parent_socket_index is None:
                    raise ValueError("ConnectStraight requires secondary_parent_block_id and secondary_parent_socket_index")
                if secondary_parent_block_id not in created_blocks:
                    raise ValueError(
                        f"Unknown secondary_parent_block_id '{secondary_parent_block_id}' for block '{block_graph_id}'. "
                        "Configs must be ordered parent-before-child."
                    )
                if not isinstance(secondary_parent_socket_index, int):
                    raise ValueError(
                        f"secondary_parent_socket_index must be int for block '{block_graph_id}', "
                        f"got {secondary_parent_socket_index!r}"
                    )
                if parent_block_id == secondary_parent_block_id and parent_socket_index == secondary_parent_socket_index:
                    raise ValueError("ConnectStraight requires two distinct parent sockets")
                if (
                    block_type_id != "H"
                    and (secondary_parent_block_id, secondary_parent_socket_index) in used_parent_sockets
                ):
                    raise ValueError(
                        f"Secondary parent socket already occupied: parent_block_id='{secondary_parent_block_id}', "
                        f"parent_socket_index={secondary_parent_socket_index}"
                    )
                secondary_parent_block = created_blocks[secondary_parent_block_id]
                secondary_parent_socket = secondary_parent_block.get_socket(secondary_parent_socket_index)
                block = block_type(
                    block_index,
                    parent_socket,
                    self.road_network,
                    random_seed=self.engine.global_random_seed,
                    secondary_pre_block_socket=secondary_parent_socket,
                    ignore_intersection_checking=True
                )
            else:
                if secondary_parent_block_id is not None or secondary_parent_socket_index is not None:
                    raise ValueError(
                        f"Only ConnectStraight may use secondary parent socket fields, got block '{block_type_id}'"
                    )
                block = block_type(
                    block_index,
                    parent_socket,
                    self.road_network,
                    random_seed=self.engine.global_random_seed,
                    ignore_intersection_checking=True
                )
            block.graph_block_id = block_graph_id
            block.graph_parent_block_id = parent_block_id
            block.graph_parent_socket_index = parent_socket_index
            if block_type_id == "H":
                block.graph_secondary_parent_block_id = secondary_parent_block_id
                block.graph_secondary_parent_socket_index = secondary_parent_socket_index
            block.construct_from_config(b, parent_node_path, physics_world)
            self.blocks.append(block)
            created_blocks[block_graph_id] = block
            used_parent_sockets.add((parent_block_id, parent_socket_index))
            if block_type_id == "H":
                used_parent_sockets.add((secondary_parent_block_id, secondary_parent_socket_index))

    @property
    def road_network_type(self):
        return NodeRoadNetwork

    def get_meta_data(self):
        assert self.blocks is not None and len(self.blocks) > 0, "Please generate Map before saving it"
        map_config = []
        for b in self.blocks:
            if isinstance(b, FirstPGBlock):
                continue
            b_config = b.get_config()
            json_config = b_config.get_serializable_dict()
            json_config[self.GRAPH_BLOCK_ID] = getattr(b, "graph_block_id", b.name)
            json_config[self.BLOCK_ID] = b.ID
            json_config[self.PARENT_BLOCK_ID] = getattr(b, "graph_parent_block_id", "root")
            json_config[self.PARENT_SOCKET_INDEX] = getattr(b, "graph_parent_socket_index", 0)
            if hasattr(b, "graph_secondary_parent_block_id"):
                json_config[self.SECONDARY_PARENT_BLOCK_ID] = b.graph_secondary_parent_block_id
                json_config[self.SECONDARY_PARENT_SOCKET_INDEX] = b.graph_secondary_parent_socket_index
            map_config.append(json_config)

        saved_data = copy.deepcopy({self.BLOCK_SEQUENCE: map_config, "map_config": self.config.copy()})
        saved_data.update(super(PGMap, self).get_meta_data())
        return saved_data

    def show_coordinates(self):
        lanes = []
        for to_ in self.road_network.graph.values():
            for lanes_to_add in to_.values():
                lanes += lanes_to_add
        self._show_coordinates(lanes)

    def get_boundary_line_vector(self, interval):
        map = self
        ret = {}
        for _from in map.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in map.road_network.graph[_from].keys():
                for l in map.road_network.graph[_from][_to]:
                    sides = 2 if l is map.road_network.graph[_from][_to][-1] or decoration else 1
                    for side in range(sides):
                        type = l.line_types[side]
                        if type == PGLineType.NONE:
                            continue
                        color = l.line_colors[side]
                        line_type = self.get_line_type(type, color)
                        lateral = l.width / 2
                        if side == 0:
                            lateral *= -1
                        ret["{}_{}".format(l.index, side)] = {
                            "type": line_type,
                            "polyline": l.get_polyline(interval, lateral),
                            "speed_limit_kmh": l.speed_limit
                        }
        return ret

    def get_line_type(self, type, color):
        if type == PGLineType.CONTINUOUS and color == PGLineColor.YELLOW:
            return MetaDriveType.LINE_SOLID_SINGLE_YELLOW
        elif type == PGLineType.BROKEN and color == PGLineColor.YELLOW:
            return MetaDriveType.LINE_BROKEN_SINGLE_YELLOW
        elif type == PGLineType.CONTINUOUS and color == PGLineColor.GREY:
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
        elif type == PGLineType.BROKEN and color == PGLineColor.GREY:
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        elif type == PGLineType.SIDE:
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
        else:
            # Unknown line type
            return MetaDriveType.LINE_UNKNOWN
