from types import SimpleNamespace

from envs.diffusion_envs.base_multi_env import BaseMultiEnv


class StubLane:
    def __init__(self, valid_points):
        self._valid_points = {tuple(point) for point in valid_points}

    def point_on_lane(self, point):
        return tuple(point) in self._valid_points


def make_vehicle(
    bounding_box,
    current_ref_lanes=None,
    lane=None,
    out_of_route=False,
    on_lane=False,
):
    navigation = SimpleNamespace(current_ref_lanes=current_ref_lanes)
    return SimpleNamespace(
        bounding_box=list(bounding_box),
        navigation=navigation,
        lane=lane,
        out_of_route=out_of_route,
        on_lane=on_lane,
    )


def make_env(out_of_route_done=False):
    env = BaseMultiEnv.__new__(BaseMultiEnv)
    env.config = {"out_of_route_done": out_of_route_done}
    return env


def test_out_of_road_is_false_when_vehicle_only_touches_line_but_still_overlaps_lane():
    lane = StubLane(valid_points=[(0.0, 0.0)])
    vehicle = make_vehicle(
        bounding_box=[(0.0, 0.0), (5.0, 0.0), (5.0, 1.0), (0.0, 1.0)],
        current_ref_lanes=[lane],
    )

    assert make_env()._is_out_of_road(vehicle) is False


def test_out_of_road_is_false_when_one_corner_remains_inside_any_candidate_lane():
    lane = StubLane(valid_points=[(1.0, 1.0)])
    vehicle = make_vehicle(
        bounding_box=[(-2.0, -1.0), (4.0, -1.0), (4.0, 4.0), (1.0, 1.0)],
        current_ref_lanes=[lane],
    )

    assert make_env()._is_out_of_road(vehicle) is False


def test_out_of_road_is_true_when_all_vehicle_corners_are_outside_candidate_lanes():
    lane = StubLane(valid_points=[(9.0, 9.0)])
    vehicle = make_vehicle(
        bounding_box=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        current_ref_lanes=[lane],
        on_lane=False,
    )

    assert make_env()._is_out_of_road(vehicle) is True


def test_out_of_road_falls_back_to_vehicle_lane_when_current_ref_lanes_missing():
    lane = StubLane(valid_points=[(2.0, 2.0)])
    vehicle = make_vehicle(
        bounding_box=[(2.0, 2.0), (5.0, 2.0), (5.0, 5.0), (2.0, 5.0)],
        current_ref_lanes=None,
        lane=lane,
    )

    assert make_env()._is_out_of_road(vehicle) is False


def test_out_of_road_preserves_out_of_route_override_when_enabled():
    lane = StubLane(valid_points=[(0.0, 0.0)])
    vehicle = make_vehicle(
        bounding_box=[(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
        current_ref_lanes=[lane],
        out_of_route=True,
    )

    assert make_env(out_of_route_done=True)._is_out_of_road(vehicle) is True
