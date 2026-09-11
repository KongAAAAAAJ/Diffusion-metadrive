from pathlib import Path


def test_joint_grpo_contains_multisource_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / 'models/bev_planner/joint_grpo.py').read_text()
    assert 'risk_pact_use_platoon_actor' in source
    assert 'risk_pact_use_road_boundary' in source
    assert 'build_platoon_actor_state' in source
    assert 'build_drivable_signed_distance' in source
    assert 'risk_pact/road_risk_mean' in source
    assert 'risk_pact/platoon_risk_mean' in source


def test_yaml_exposes_three_component_switches() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / 'configs/train/bev_joint_grpo.yaml').read_text()
    assert 'background_actor: true' in source
    assert 'platoon_actor: true' in source
    assert 'road_boundary: true' in source
