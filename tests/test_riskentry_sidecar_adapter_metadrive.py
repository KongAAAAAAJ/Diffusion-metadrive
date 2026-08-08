from __future__ import annotations

import numpy as np

from expert_dataset.collect_joint_bev import SensorlessJointBEVPlatoonEnv
from expert_dataset.riskentry_sidecar_adapter import MetaDriveRiskEntrySidecarAdapter


def test_real_sensorless_metadrive_registry_and_two_frame_derivatives():
    env = SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "use_render": False,
            "use_hybrid_map": True,
            "num_scenarios": 1,
            "start_seed": 17,
            "traffic_density": 0.0,
            "scenario_id": "S5_hard_brake_lead",
            "local_route": "R1_entry_straight",
            "horizon": 10,
        }
    )
    try:
        env.set_runtime_scenario_route("S5_hard_brake_lead", "R1_entry_straight")
        env.reset(seed=17)
        adapter = MetaDriveRiskEntrySidecarAdapter(decision_dt_s=0.1)
        initial = adapter.capture_frame(env, step_index=0, timestamp_s=0.0)

        assert [item.actor_id for item in adapter.actor_records[:3]] == ["P0", "P1", "P2"]
        assert [item.source_object_id for item in adapter.actor_records[:3]] == [
            "agent0",
            "agent1",
            "agent2",
        ]
        assert all(item.lane_valid for item in initial.frame.actors[:3])
        assert all(not item.acceleration_valid for item in initial.frame.actors)

        actions = {
            agent_id: np.asarray([0.0, 0.0], dtype=np.float32)
            for agent_id in ("agent0", "agent1", "agent2")
        }
        _, _, terminated, truncated, info = env.step(actions)
        following = adapter.capture_frame(
            env,
            step_index=1,
            timestamp_s=0.1,
            transition_info=info,
            terminated=terminated,
            truncated=truncated,
        )

        persistent = set(item.actor_id for item in initial.frame.actors) & set(
            item.actor_id for item in following.frame.actors
        )
        assert {"P0", "P1", "P2"}.issubset(persistent)
        assert all(
            item.acceleration_valid and item.yaw_rate_valid
            for item in following.frame.actors
            if item.actor_id in persistent
        )
        assert all(np.isfinite(item.velocity_x_mps) for item in following.frame.actors)
        assert all(np.isfinite(item.velocity_y_mps) for item in following.frame.actors)
    finally:
        env.close()

