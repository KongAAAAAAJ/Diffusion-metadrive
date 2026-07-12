import pytest

from models.diffusion.test_refine_grpo import _build_env_config, _resolve_test_config, parse_args


@pytest.mark.parametrize(
    "argv",
    [
        ["--checkpoint", "/tmp/ignored.ckpt"],
        ["--scenario-id", "S5_hard_brake_lead"],
        ["--local-route", "R1_entry_straight"],
        ["--episodes", "2"],
        ["--start-seed", "10"],
        ["--num-scenarios", "2"],
        ["--traffic-density", "0.04"],
        ["--random-traffic", "0"],
        ["--max-steps", "5"],
        ["--device", "cpu"],
        ["--output-dir", "/tmp/out"],
        ["--save-combined-traj-frames", "0"],
        ["--save-traj-plots", "0"],
        ["--save-trajectory-data", "0"],
        ["--save-2d-video", "0"],
        ["--save-3d-video", "1"],
        ["--video-fps", "10"],
        ["--render", "0"],
    ],
)
def test_yaml_owned_options_are_not_cli_overrides(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)

    args = parse_args([])

    assert not hasattr(args, "checkpoint")
    assert not hasattr(args, "scenario_id")
    assert not hasattr(args, "episodes")
    assert hasattr(args, "refine_train_config_path")


def test_yaml_scenario_ids_are_preserved_without_default_cli_override():
    config = {
        "env_config": {
            "scenario_ids": ["S5_hard_brake_lead"],
            "planner_device": "cpu",
        },
    }
    test_config = _resolve_test_config(config)

    env_config = _build_env_config(config, test_config, planner=object())

    assert env_config["scenario_ids"] == ["S5_hard_brake_lead"]
    assert "scenario_id" not in env_config
    assert "local_route" not in env_config


def test_top_level_start_seed_is_used_when_env_seed_is_missing():
    config = {
        "start_seed": 42,
        "env_config": {
            "planner_device": "cpu",
            "num_scenarios": 3,
        },
    }
    test_config = _resolve_test_config(config)

    env_config = _build_env_config(config, test_config, planner=object())

    assert env_config["start_seed"] == 42
    assert env_config["num_scenarios"] == 3


def test_save_3d_video_enables_render_in_resolved_env_config():
    config = {
        "test_config": {"save_3d_video": True},
        "env_config": {
            "planner_device": "cpu",
            "use_render": False,
        },
    }
    test_config = _resolve_test_config(config)

    env_config = _build_env_config(config, test_config, planner=object())

    assert env_config["use_render"] is True
