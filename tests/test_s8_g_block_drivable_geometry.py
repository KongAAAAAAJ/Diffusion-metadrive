from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from envs.diffusion_envs.base_multi_env import BaseMultiEnv
from envs.observations.semantic_bev import (
    BEVChannel,
    MetaDriveSceneAdapter,
    SemanticBEVRasterizer,
)
from expert_dataset.collect_joint_bev import SensorlessJointBEVPlatoonEnv
from models.platoon_planner.platoon_normal_planner import (
    audit_dense_footprint_on_lanes,
)
from models.platoon_planner.route_chain_geometry import (
    build_continuous_lane_chain_path,
)


def test_s8_g_block_has_non_routing_drivable_junction_surface() -> None:
    env = SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "use_render": False,
            "use_hybrid_map": True,
            "num_scenarios": 1,
            "start_seed": 17,
            "traffic_density": 0.0,
            "scenario_id": "S8_ego_exit_to_ramp",
            "local_route": "R6_exit_to_ramp",
            "horizon": 80,
        }
    )
    try:
        env.set_runtime_scenario_route(
            "S8_ego_exit_to_ramp", "R6_exit_to_ramp"
        )
        env.reset(seed=17)
        network = env.current_map.road_network
        lane_keys = (
            ("3C0_1_", "4G0_0_", 2),
            ("4G0_0_", "4G1_1_", 0),
            ("4G1_1_", "4G1_2_", 0),
            ("4G1_2_", "4G1_3_", 0),
            ("4G1_3_", "4G1_4_", 0),
            ("4G1_4_", "15s0_0_", 0),
        )
        lanes = [network.graph[start][end][index] for start, end, index in lane_keys]
        overlays = tuple(network.drivable_geometry_overlays)

        assert len(overlays) == 1
        assert overlays[0].width == 3.5
        assert overlays[0].need_lane_localization is False
        assert not any(overlays[0] is lane for lane in network.get_all_lanes())
        assert tuple(lanes[0].junction_drivable_surfaces) == overlays
        assert tuple(lanes[1].junction_drivable_surfaces) == overlays
        drivable_candidates = BaseMultiEnv._get_candidate_drivable_lanes(
            SimpleNamespace(
                lane=lanes[0],
                navigation=SimpleNamespace(current_ref_lanes=[lanes[0]]),
            )
        )
        assert overlays[0] in drivable_candidates

        path = build_continuous_lane_chain_path(
            lanes,
            start_s=float(lanes[0].length) - 12.0,
            step_m=0.1,
            seam_transition_m=8.0,
        )[:450]
        valid, detail = audit_dense_footprint_on_lanes(
            path,
            lanes,
            (5.74, 2.3),
            dense_dt_s=0.1,
        )
        assert valid, detail

        adapter = MetaDriveSceneAdapter()
        snapshots = tuple(
            adapter.capture_snapshot(env, timestamp)
            for timestamp in (0.0, 0.5, 1.0)
        )
        scene = adapter.build_scene(env, "agent0", snapshots)
        assert any(
            np.array_equal(polygon, overlays[0].polygon.astype(np.float32))
            for polygon in scene.drivable_polygons
        )
        surface_midpoint = overlays[0].position(0.5 * overlays[0].length, 0.0)
        scene_at_junction = replace(
            scene,
            ego_pose=np.asarray(
                [
                    surface_midpoint[0],
                    surface_midpoint[1],
                    overlays[0].heading_theta_at(0.5 * overlays[0].length),
                ],
                dtype=np.float32,
            ),
        )
        bev = SemanticBEVRasterizer(adapter.config).rasterize(scene_at_junction)
        local = np.asarray(surface_midpoint, dtype=np.float32)[None, :2]
        col, row = SemanticBEVRasterizer(adapter.config).world_to_pixel(
            local, scene_at_junction.ego_pose
        )[0]
        row_index = int(np.clip(round(float(row)), 0, 255))
        col_index = int(np.clip(round(float(col)), 0, 255))
        assert bev[BEVChannel.DRIVABLE, row_index, col_index] == 255
    finally:
        env.close()
