from __future__ import annotations

import math

import numpy as np

from metadrive.policy.diffusion_policy.preference_guidance import inject_external_preference_point


class _StraightLane:
    length = 100.0

    def local_coordinates(self, position):
        return float(position[0]), float(position[1])

    def position(self, longitudinal, lateral):
        return np.asarray([float(longitudinal), float(lateral)], dtype=np.float32)


class _Vehicle:
    def __init__(self, position, heading=0.0, lane=None):
        self.position = np.asarray(position, dtype=np.float32)
        self.heading_theta = float(heading)
        self.lane = lane


class _Env:
    def __init__(self):
        lane = _StraightLane()
        self.agents = {
            "agent0": _Vehicle([10.0, 0.0], lane=lane),
            "agent1": _Vehicle([0.0, -2.0], heading=0.0, lane=lane),
            "agent2": _Vehicle([0.0, 2.0], heading=math.pi / 2.0, lane=lane),
        }


def test_external_preference_uses_leader_lane_before_planner_export():
    planner_batch = {"agent0": {}, "agent1": {}, "agent2": {}}

    metadata = inject_external_preference_point(
        planner_batch,
        coarse_by_agent={},
        agent_ids=["agent0", "agent1", "agent2"],
        env=_Env(),
        target_speed_km_h=36.0,
        horizon_s=2.0,
    )

    np.testing.assert_allclose(planner_batch["agent0"]["preference_point"], [20.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(planner_batch["agent1"]["preference_point"], [20.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(planner_batch["agent2"]["preference_point"], [-2.0, -20.0], atol=1e-5)
    assert metadata["agent0"]["preference_source"] == "leader_lane"


def test_external_preference_falls_back_to_keep_lane_coarse_endpoint():
    coarse = np.zeros((4, 8, 2), dtype=np.float32)
    coarse[0, -1] = [7.0, -1.0]
    coarse[2, -1] = [99.0, 99.0]
    planner_batch = {"agent0": {}}

    metadata = inject_external_preference_point(
        planner_batch,
        coarse_by_agent={"agent0": coarse},
        agent_ids=["agent0"],
        env=None,
    )

    np.testing.assert_allclose(planner_batch["agent0"]["preference_point"], [7.0, -1.0], atol=1e-5)
    assert metadata["agent0"]["preference_source"] == "fallback_keep_lane_coarse"
