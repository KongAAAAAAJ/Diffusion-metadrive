from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace
from metadrive.component.pgblock.create_pg_block_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad
from metadrive.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metadrive.component.road_network import Road
from metadrive.constants import PGLineType
import numpy as np


class Straight(PGBlock):
    """
    Straight Road
    ----------------------------------------
    ----------------------------------------
    ----------------------------------------
    """
    ID = "S"
    SOCKET_NUM = 1
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.STRAIGHT)

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)  # only one part in simple block like straight, and curve
        para = self.get_config()
        length = para[Parameter.length]
        basic_lane = self.positive_basic_lane
        assert isinstance(basic_lane, StraightLane), "Straight road can only connect straight type"
        new_lane = ExtendStraightLane(basic_lane, length, [PGLineType.BROKEN, PGLineType.SIDE])
        start = self.pre_block_socket.positive_road.end_node
        end = self.add_road_node()
        socket = Road(start, end)
        _socket = -socket

        # create positive road
        no_cross = CreateRoadFrom(
            new_lane,
            self.positive_lane_num,
            socket,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking,
            side_lane_line_type=self.side_lane_line_type,
            center_line_type=self.center_line_type
        )
        if not self.remove_negative_lanes:
            # create negative road
            no_cross = CreateAdverseRoad(
                socket,
                self.block_network,
                self._global_network,
                ignore_intersection_checking=self.ignore_intersection_checking,
                side_lane_line_type=self.side_lane_line_type,
                center_line_type=self.center_line_type
            ) and no_cross

        self.add_sockets(PGBlockSocket(socket, _socket))
        return no_cross


class OneWayStraight(Straight):
    ID = "s"
    IS_ONE_WAY_BLOCK = True

    def __init__(
        self,
        block_index: int,
        pre_block_socket,
        global_network,
        random_seed,
        ignore_intersection_checking=False,
        side_lane_line_type=None,
        center_line_type=None,
    ):
        super().__init__(
            block_index,
            pre_block_socket,
            global_network,
            random_seed,
            ignore_intersection_checking=ignore_intersection_checking,
            remove_negative_lanes=True,
            side_lane_line_type=side_lane_line_type,
            center_line_type=center_line_type,
        )

    def _try_plug_into_previous_block(self) -> bool:
        no_cross = super()._try_plug_into_previous_block()
        self.get_socket(0).is_one_way = True
        return no_cross


class ConnectStraight(Straight):
    ID = "H"
    SOCKET_NUM = 0

    def __init__(
        self,
        block_index: int,
        pre_block_socket,
        global_network,
        random_seed,
        secondary_pre_block_socket,
        ignore_intersection_checking=False,
        side_lane_line_type=None,
        center_line_type=None,
    ):
        self.secondary_pre_block_socket = secondary_pre_block_socket
        self.secondary_pre_block_socket_index = secondary_pre_block_socket.index
        is_one_way_connection = getattr(pre_block_socket, "is_one_way", False) and getattr(
            secondary_pre_block_socket, "is_one_way", False
        )
        super().__init__(
            block_index,
            pre_block_socket,
            global_network,
            random_seed,
            ignore_intersection_checking=ignore_intersection_checking,
            remove_negative_lanes=is_one_way_connection,
            side_lane_line_type=side_lane_line_type,
            center_line_type=center_line_type,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _validate_connectable_sockets(self):
        is_first_one_way = getattr(self.pre_block_socket, "is_one_way", False)
        is_second_one_way = getattr(self.secondary_pre_block_socket, "is_one_way", False)
        if is_first_one_way != is_second_one_way:
            raise ValueError("ConnectStraight requires both sockets to share the same one-way setting")

        is_one_way = is_first_one_way

        first_positive = self.pre_block_socket.get_positive_lanes(self._global_network)
        second_positive = self.secondary_pre_block_socket.get_positive_lanes(self._global_network)
        first_negative = []
        second_negative = []
        if not is_one_way:
            first_negative = self.pre_block_socket.get_negative_lanes(self._global_network)
            second_negative = self.secondary_pre_block_socket.get_negative_lanes(self._global_network)

        if len(first_positive) != len(second_positive):
            raise ValueError("ConnectStraight requires matching lane counts on both sockets")
        if not is_one_way and len(first_negative) != len(second_negative):
            raise ValueError("ConnectStraight requires matching adverse lane counts on both sockets")
        if not first_positive:
            raise ValueError("ConnectStraight requires non-empty positive lanes")

        lanes_to_check = first_positive + second_positive
        if not is_one_way:
            lanes_to_check += first_negative + second_negative
        if not all(isinstance(lane, StraightLane) for lane in lanes_to_check):
            raise ValueError("ConnectStraight only supports straight-to-straight socket connections")

        first_width = first_positive[0].width_at(0)
        second_width = second_positive[0].width_at(0)
        if abs(first_width - second_width) > 1e-3:
            raise ValueError("ConnectStraight requires matching lane widths")

        first_heading = first_positive[0].heading_theta_at(first_positive[0].length)
        second_heading = second_positive[0].heading_theta_at(second_positive[0].length)
        heading_diff = abs(self._wrap_angle(first_heading - second_heading))
        if not is_one_way and abs(heading_diff - np.pi) > np.deg2rad(15):
            raise ValueError("ConnectStraight requires two straight sockets to face each other")

        return first_positive, second_positive, first_negative, second_negative, is_one_way

    def _lane_line_types(self, lane_idx: int, lane_num: int):
        left_line_type = PGLineType.BROKEN
        right_line_type = PGLineType.SIDE if lane_idx == lane_num - 1 else PGLineType.BROKEN
        return left_line_type, right_line_type

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)
        first_positive, second_positive, first_negative, second_negative, is_one_way = self._validate_connectable_sockets()
        lane_num = len(first_positive)

        if is_one_way:
            forward_road = Road(
                self.pre_block_socket.positive_road.end_node,
                self.secondary_pre_block_socket.positive_road.start_node,
            )
        else:
            forward_road = Road(
                self.pre_block_socket.positive_road.end_node,
                self.secondary_pre_block_socket.negative_road.start_node,
            )
            reverse_road = Road(
                self.secondary_pre_block_socket.positive_road.end_node,
                self.pre_block_socket.negative_road.start_node,
            )

        forward_targets = second_positive if is_one_way else second_negative
        for lane_idx, (start_lane, end_lane) in enumerate(zip(first_positive, forward_targets)):
            line_types = self._lane_line_types(lane_idx, lane_num)
            lane = StraightLane(
                start_lane.end,
                end_lane.start,
                start_lane.width_at(0),
                line_types,
                speed_limit=min(start_lane.speed_limit, end_lane.speed_limit),
                priority=start_lane.priority,
            )
            self.block_network.add_lane(forward_road.start_node, forward_road.end_node, lane)

        self.add_respawn_roads(forward_road)

        if not is_one_way:
            for lane_idx, (start_lane, end_lane) in enumerate(zip(second_positive, first_negative)):
                line_types = self._lane_line_types(lane_idx, lane_num)
                lane = StraightLane(
                    start_lane.end,
                    end_lane.start,
                    start_lane.width_at(0),
                    line_types,
                    speed_limit=min(start_lane.speed_limit, end_lane.speed_limit),
                    priority=start_lane.priority,
                )
                self.block_network.add_lane(reverse_road.start_node, reverse_road.end_node, lane)

        return True
