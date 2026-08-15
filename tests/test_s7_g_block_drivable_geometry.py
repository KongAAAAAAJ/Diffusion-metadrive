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


def test_s7_g_block_has_non_routing_drivable_merge_surface() -> None:
    env = SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "use_render": False,
            "use_hybrid_map": True,
            "num_scenarios": 1,
            "start_seed": 17,
            "traffic_density": 0.0,
            "scenario_id": "S7_ego_merge_from_ramp",
            "local_route": "R7_merge_core",
            "horizon": 80,
        }
    )
    try:
        env.set_runtime_scenario_route(
            "S7_ego_merge_from_ramp", "R7_merge_core"
        )
        env.reset(seed=17)
        network = env.current_map.road_network
        ramp_lane = network.graph["18c0_1_"]["9g0_0_"][0]
        mainline_right_lane = network.graph["9g0_0_"]["9g0_1_"][2]
        all_overlays = tuple(network.drivable_geometry_overlays)
        overlays = tuple(
            surface
            for surface in all_overlays
            if "route-merge-junction" in str(surface.index[0])
        )

        assert len(overlays) == 1
        assert overlays[0].width == 5.0
        assert ramp_lane.route_seam_transition_m == 20.0
        assert overlays[0].need_lane_localization is False
        assert not any(overlays[0] is lane for lane in network.get_all_lanes())
        assert tuple(ramp_lane.junction_drivable_surfaces) == overlays
        assert all(
            surface in tuple(mainline_right_lane.junction_drivable_surfaces)
            for surface in overlays
        )
        drivable_candidates = BaseMultiEnv._get_candidate_drivable_lanes(
            SimpleNamespace(
                lane=ramp_lane,
                navigation=SimpleNamespace(current_ref_lanes=[ramp_lane]),
            )
        )
        assert overlays[0] in drivable_candidates

        path = build_continuous_lane_chain_path(
            [ramp_lane, mainline_right_lane],
            start_s=float(ramp_lane.length) - 12.0,
            step_m=0.1,
            seam_transition_m=20.0,
        )[:450]
        valid, detail = audit_dense_footprint_on_lanes(
            path,
            [ramp_lane, mainline_right_lane],
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
        surface_midpoint = overlays[0].position(
            0.5 * overlays[0].length, 0.0
        )
        scene_at_junction = replace(
            scene,
            ego_pose=np.asarray(
                [
                    surface_midpoint[0],
                    surface_midpoint[1],
                    overlays[0].heading_theta_at(
                        0.5 * overlays[0].length
                    ),
                ],
                dtype=np.float32,
            ),
        )
        bev = SemanticBEVRasterizer(adapter.config).rasterize(
            scene_at_junction
        )
        local = np.asarray(surface_midpoint, dtype=np.float32)[None, :2]
        col, row = SemanticBEVRasterizer(adapter.config).world_to_pixel(
            local, scene_at_junction.ego_pose
        )[0]
        row_index = int(np.clip(round(float(row)), 0, 255))
        col_index = int(np.clip(round(float(col)), 0, 255))
        assert bev[BEVChannel.DRIVABLE, row_index, col_index] == 255
    finally:
        env.close()
