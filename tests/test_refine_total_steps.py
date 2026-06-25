from train.train_refine_grpo import parse_args, resolve_total_timesteps


def test_resolve_total_timesteps_uses_yaml_when_cli_not_provided():
    assert resolve_total_timesteps(None, {"total_timesteps": 200000}) == 200000


def test_resolve_total_timesteps_prefers_explicit_cli_value():
    assert resolve_total_timesteps(50000, {"total_timesteps": 200000}) == 50000


def test_parse_args_matches_refine_shell_env_defaults(monkeypatch):
    monkeypatch.setenv("CONFIG", "/tmp/refine.yaml")
    monkeypatch.setenv("SCENARIO_IDS", "S1_free_cruise_straight,S2_free_cruise_curve")

    args = parse_args([])

    assert args.config == "/tmp/refine.yaml"
    assert args.scenario_ids == "S1_free_cruise_straight,S2_free_cruise_curve"
    assert args.num_agents is None
    assert args.planner_device == ""


def test_parse_args_no_longer_exposes_cls_grpo_bootstrap(monkeypatch):
    monkeypatch.setenv("SCENARIO_IDS", "S5_hard_brake_lead")

    args = parse_args([])

    assert not hasattr(args, "cls_grpo_ckpt_dir")
    assert not hasattr(args, "cls_grpo_full_ckpt")
