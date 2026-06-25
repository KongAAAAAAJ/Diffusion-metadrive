from types import SimpleNamespace

from models.diffusion.test_refine_grpo import _build_env_config


def test_explicit_scenario_id_removes_yaml_scenario_ids():
    config = {
        "env_config": {
            "scenario_ids": ["S5_hard_brake_lead"],
        },
    }
    args = SimpleNamespace(
        render=0,
        save_3d_video=0,
        device="cpu",
        scenario_id="S4_curve_following",
        local_route="",
        start_seed=10,
        num_scenarios=1,
        traffic_density=0.04,
        random_traffic=0,
    )

    env_config = _build_env_config(config, args, planner=object())

    assert env_config["scenario_id"] == "S4_curve_following"
    assert "scenario_ids" not in env_config
    assert env_config["local_route"] == "R2_entry_curve"


def test_explicit_local_route_overrides_default_scenario_route():
    config = {
        "env_config": {
            "scenario_ids": ["S5_hard_brake_lead"],
            "local_route": "R1_entry_straight",
        },
    }
    args = SimpleNamespace(
        render=0,
        save_3d_video=0,
        device="cpu",
        scenario_id="S4_curve_following",
        local_route="R5_ramp_curve",
        start_seed=10,
        num_scenarios=1,
        traffic_density=0.04,
        random_traffic=0,
    )

    env_config = _build_env_config(config, args, planner=object())

    assert env_config["scenario_id"] == "S4_curve_following"
    assert "scenario_ids" not in env_config
    assert env_config["local_route"] == "R5_ramp_curve"
