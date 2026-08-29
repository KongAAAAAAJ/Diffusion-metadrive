from __future__ import annotations

import csv
import copy
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluation.bev_reward_comparison as reward_comparison
from evaluation.bev_reward_comparison import (
    CSV_COLUMNS,
    REWARD_COLUMNS,
    RewardComparisonConfig,
    RewardComparisonError,
    _model_specs,
    _reward_row,
    _write_reward_csv,
    reward_values,
)
from models.bev_planner.joint_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JOINT_REWARD_CONTRACT,
    JOINT_REWARD_CONTRACT_SHA256,
    JointRewardConfig,
    compose_joint_reward,
    joint_reward_config_sha256,
)
from models.bev_planner.joint_grpo import JointGRPOConfig, JointGRPOError
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizerConfig,
)
from scenarios.bev_round13_contract import primary_scenario_contract
from train.bev_joint_grpo import load_grpo_config_from_checkpoint


def _result():
    config = JointRewardConfig()
    result = compose_joint_reward(
        progress_score=np.asarray([0.8]),
        formation_penalty=np.asarray([0.2]),
        gap_penalty=np.asarray([0.3]),
        ttc_penalty=np.asarray([0.4]),
        road_penalty=np.asarray([0.5]),
        comfort_penalty=np.asarray([0.6]),
        collision=np.asarray([True]),
        out_of_drivable=np.asarray([False]),
        clearance_violation=np.asarray([False]),
        config=config,
    )
    return config, result


def _valid_grpo_contract_payload() -> dict[str, object]:
    reward_config = JointRewardConfig()
    optimizer_config = KinematicTrajectoryOptimizerConfig()
    application = dict(GRPO_OPEN_REWARD_APPLICATION_CONTRACT)
    return {
        "run_mode": "smoke",
        "reward_contract_version": JOINT_REWARD_CONTRACT["version"],
        "reward_contract_sha256": JOINT_REWARD_CONTRACT_SHA256,
        "reward_config": asdict(reward_config),
        "reward_config_sha256": joint_reward_config_sha256(reward_config),
        "reward_application_contract": application,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": application["reward_input_domain"],
        "candidate_selection_domain": application["candidate_selection_domain"],
        "execution_input_domain": application["execution_input_domain"],
        "best_checkpoint_metric": application["best_checkpoint_metric"],
        "tracking_expansion_enabled": application["tracking_expansion_enabled"],
        "calibration_required": application["calibration_required"],
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "scenario_seeds": [31, 47],
        "environment_steps": 77_500,
        "scenario_contract_sha256": primary_scenario_contract()["sha256"],
        "trajectory_optimizer_config": asdict(optimizer_config),
        "trajectory_optimizer_sha256": optimizer_config.sha256(),
    }


def test_reward_comparator_has_no_legacy_evaluator_dependency() -> None:
    source = Path(reward_comparison.__file__).read_text(encoding="utf-8")

    assert "bev_four_model_evaluator" not in source


def test_grpo_checkpoint_contract_accepts_only_frozen_diagnostic_tau_d() -> None:
    payload = _valid_grpo_contract_payload()

    binding = reward_comparison._validate_grpo_checkpoint_contract(payload)

    assert binding == (
        JOINT_REWARD_CONTRACT["version"],
        JOINT_REWARD_CONTRACT_SHA256,
        joint_reward_config_sha256(JointRewardConfig()),
    )

    wrong_application = copy.deepcopy(payload)
    wrong_application["reward_input_domain"] = "tau_a"
    with pytest.raises(RewardComparisonError, match="application metadata"):
        reward_comparison._validate_grpo_checkpoint_contract(wrong_application)

    wrong_reward = copy.deepcopy(payload)
    wrong_reward_config = dict(wrong_reward["reward_config"])
    wrong_reward_config["progress_weight"] = 0.5
    wrong_reward["reward_config"] = wrong_reward_config
    wrong_reward["reward_config_sha256"] = joint_reward_config_sha256(
        JointRewardConfig(**wrong_reward_config)
    )
    with pytest.raises(RewardComparisonError, match="active frozen joint reward"):
        reward_comparison._validate_grpo_checkpoint_contract(wrong_reward)

    formal = copy.deepcopy(payload)
    formal["run_mode"] = "formal"
    with pytest.raises(RewardComparisonError, match="run_mode mismatch"):
        reward_comparison._validate_grpo_checkpoint_contract(formal)


def test_initial_scene_hash_is_order_stable_and_scene_sensitive() -> None:
    class Vehicle:
        def __init__(self, name: str, x: float) -> None:
            self.name = name
            self.position = np.asarray([x, 1.0], dtype=np.float64)
            self.heading_theta = 0.1
            self.speed_km_h = 20.0
            self.lane_index = ("road", 0, 0)

    agents = {
        agent_id: Vehicle(agent_id, float(index))
        for index, agent_id in enumerate(reward_comparison.AGENT_IDS)
    }
    background_a = Vehicle("background_a", 10.0)
    background_b = Vehicle("background_b", 20.0)
    traffic_manager = SimpleNamespace(
        traffic_vehicles=[background_a, background_b]
    )
    engine = SimpleNamespace(
        traffic_manager=traffic_manager,
        get_policy=lambda name: SimpleNamespace(),
    )
    env = SimpleNamespace(agents=agents, engine=engine)

    first = reward_comparison._initial_scene_sha256(env)
    traffic_manager.traffic_vehicles.reverse()
    reordered = reward_comparison._initial_scene_sha256(env)
    background_a.position[0] += 1.0
    changed = reward_comparison._initial_scene_sha256(env)

    assert reordered == first
    assert changed != first


def test_signed_reward_terms_sum_to_total_reward() -> None:
    config, result = _result()

    values = reward_values(result, config)

    assert tuple(values) == REWARD_COLUMNS
    assert values["progress_reward"] == pytest.approx(0.47 * 0.8)
    assert values["formation_reward"] == pytest.approx(-0.2)
    assert values["gap_reward"] == pytest.approx(-1.185 * 0.3)
    assert values["ttc_reward"] == pytest.approx(-0.5 * 0.4)
    assert values["road_reward"] == pytest.approx(-0.5 * 0.5)
    assert values["comfort_reward"] == pytest.approx(-0.0225 * 0.6)
    assert values["collision_reward"] == pytest.approx(-5.0)
    assert values["out_of_drivable_reward"] == pytest.approx(0.0)
    assert values["total_reward"] == pytest.approx(
        sum(values[name] for name in REWARD_COLUMNS[1:])
    )


def test_reward_row_is_one_three_vehicle_joint_reward() -> None:
    config, result = _result()

    row = _reward_row(
        model_id="stage1_a",
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        seed=31,
        step=7,
        result=result,
        config=config,
    )

    assert tuple(row) == CSV_COLUMNS
    assert row["step"] == 7
    assert math.isclose(
        float(row["total_reward"]),
        sum(float(row[name]) for name in REWARD_COLUMNS[1:]),
        rel_tol=1.0e-6,
        abs_tol=1.0e-6,
    )


def test_reward_csv_contains_all_steps_in_one_file(tmp_path) -> None:
    config, result = _result()
    rows = [
        _reward_row(
            model_id=model,
            scenario=("S5_hard_brake_lead", "R1_entry_straight"),
            seed=31,
            step=step,
            result=result,
            config=config,
        )
        for model in ("stage1_a", "grpo_open")
        for step in (4, 5)
    ]
    output = tmp_path / "step_rewards.csv"

    _write_reward_csv(rows, output)

    with output.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        stored = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
    assert len(stored) == 4
    assert {row["model"] for row in stored} == {"stage1_a", "grpo_open"}


def test_model_specs_bind_grpo_to_exact_stage1_checkpoint(tmp_path) -> None:
    stage1 = tmp_path / "stage1.pt"
    grpo = tmp_path / "grpo.pt"
    stage1.write_bytes(b"stage1")
    grpo.write_bytes(b"grpo")

    baseline, candidate = _model_specs(stage1, grpo)

    assert baseline.model_id == "stage1_a"
    assert candidate.model_id == "grpo_open"
    assert candidate.reward_domain == "tau_d"
    assert candidate.source_checkpoint == stage1.resolve()
    assert candidate.source_checkpoint_sha256 == baseline.checkpoint_sha256


def test_grpo_checkpoint_config_round_trips_group_size_three(
    tmp_path: Path,
) -> None:
    expected = JointGRPOConfig(group_size=3)
    checkpoint = tmp_path / "grpo.pt"
    torch.save({"grpo_config": asdict(expected)}, checkpoint)

    restored = load_grpo_config_from_checkpoint(checkpoint)

    assert restored == expected
    assert restored.group_size == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("group_size"),
        lambda value: value.update({"unexpected": 1}),
        lambda value: value.update({"group_size": 1}),
        lambda value: value.update({"group_size": "3"}),
    ],
)
def test_grpo_checkpoint_config_rejects_invalid_mapping(
    tmp_path: Path,
    mutate,
) -> None:
    raw = asdict(JointGRPOConfig(group_size=3))
    mutate(raw)
    checkpoint = tmp_path / "invalid_grpo.pt"
    torch.save({"grpo_config": raw}, checkpoint)

    with pytest.raises(JointGRPOError, match="config"):
        load_grpo_config_from_checkpoint(checkpoint)


def test_load_policy_passes_exact_checkpoint_config_to_stage1_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_config = JointGRPOConfig(group_size=3)
    planner = torch.nn.Identity()
    trainer = SimpleNamespace(planner=planner)
    observed: dict[str, object] = {}
    source_sha = "a" * 64
    spec = SimpleNamespace(
        model_id="grpo_open",
        kind="grpo",
        variant="A",
        reward_domain="tau_d",
        checkpoint=tmp_path / "grpo.pt",
        checkpoint_sha256="b" * 64,
        source_checkpoint=tmp_path / "stage1.pt",
        source_checkpoint_sha256=source_sha,
    )

    def fake_config_loader(path: Path) -> JointGRPOConfig:
        observed["config_path"] = path
        return expected_config

    def fake_stage1_loader(
        path: Path,
        *,
        device: torch.device,
        config: JointGRPOConfig,
        allow_diagnostic_source: bool,
    ):
        observed.update(
            {
                "source_path": path,
                "device": device,
                "config": config,
                "allow_diagnostic_source": allow_diagnostic_source,
            }
        )
        return trainer, {}, source_sha

    def fake_checkpoint_loader(
        path: Path,
        loaded_trainer,
        *,
        expected_source_stage1_sha256: str,
    ):
        observed.update(
            {
                "checkpoint_path": path,
                "trainer": loaded_trainer,
                "expected_source_sha": expected_source_stage1_sha256,
            }
        )
        return _valid_grpo_contract_payload()

    monkeypatch.setattr(
        reward_comparison,
        "load_grpo_config_from_checkpoint",
        fake_config_loader,
    )
    monkeypatch.setattr(
        reward_comparison,
        "load_stage1_a_for_grpo",
        fake_stage1_loader,
    )
    monkeypatch.setattr(
        reward_comparison,
        "load_grpo_checkpoint",
        fake_checkpoint_loader,
    )

    result = reward_comparison._load_policy(
        spec,
        device=torch.device("cpu"),
        reward_bindings={},
    )

    assert result is planner
    assert observed["config_path"] == spec.checkpoint
    assert observed["config"] is expected_config
    assert observed["config"] == JointGRPOConfig(group_size=3)
    assert observed["source_path"] == spec.source_checkpoint
    assert observed["checkpoint_path"] == spec.checkpoint
    assert observed["trainer"] is trainer
    assert observed["expected_source_sha"] == source_sha


def test_reward_comparison_config_rejects_invalid_step_cap() -> None:
    with pytest.raises(RewardComparisonError, match="max_steps"):
        RewardComparisonConfig(max_steps=0)


def test_evaluate_model_records_raw_reward_once_per_ready_step_with_seeded_noise(
    monkeypatch,
) -> None:
    reward_config, reward_result = _result()
    active_model = [""]
    events: list[tuple[str, str]] = []
    scored_shapes: list[tuple[int, ...]] = []
    noise_by_model: dict[str, list[torch.Tensor]] = {
        "stage1_a": [],
        "grpo_open": [],
    }

    class FakeEnv:
        def __init__(self, scenario, seed):
            self.scenario = scenario
            self.seed = seed
            self.config = {}
            self._last_planner_batch = {}

        def step(self, action):
            return {}, {}, {"__all__": False}, {"__all__": False}, {}

        def close(self):
            return None

    class FakeBuilder:
        def __init__(self, agent_ids):
            self.captures = 0

        def reset(self):
            self.captures = 0

        def capture_state(self, env, timestamp):
            self.captures += 1

        def history_ready(self):
            return self.captures > 1

        def build_model_inputs(self, env):
            return SimpleNamespace(
                mode_valid_mask=np.ones((3, 10), dtype=np.bool_)
            )

        def augment_v2_model_inputs(self, env, values, **kwargs):
            return values

    class FakeRuleMaker:
        has_active_lane_change_commitments = False
        is_formation_locked = True

        def reset(self, env, agent_ids):
            return None

        def propose_joint_actions(self, *args, **kwargs):
            proposal = SimpleNamespace(proposal_id=1)
            return SimpleNamespace(batch_id=1, proposals=(proposal,))

        def accept_joint_action(self, batch_id, proposal_id):
            return None

    class FakeRewardBackend:
        def score(self, env, values, trajectories):
            events.append(("score", active_model[0]))
            scored_shapes.append(np.asarray(trajectories).shape)
            return reward_result

    def fake_forward(planner, batch, *, diffusion_noise):
        noise_by_model[planner.model_id].append(diffusion_noise.detach().cpu().clone())
        fill = 1.0 if planner.model_id == "stage1_a" else 2.0
        return {
            "selected_trajectory": torch.full((1, 3, 8, 3), fill),
            "selected_mode": torch.zeros((1, 3), dtype=torch.int64),
        }

    def fake_optimize(values, raw, modes, *, optimizer):
        events.append(("optimizer", active_model[0]))
        return SimpleNamespace(
            optimized_trajectories=np.asarray(raw, dtype=np.float32)
        )

    monkeypatch.setattr(
        reward_comparison, "_new_env", lambda scenario, seed: FakeEnv(scenario, seed)
    )
    monkeypatch.setattr(reward_comparison, "JointBEVSampleBuilder", FakeBuilder)
    monkeypatch.setattr(reward_comparison, "make_rule_maker", lambda config: FakeRuleMaker())
    monkeypatch.setattr(reward_comparison, "KinematicTrajectoryOptimizer", object)
    monkeypatch.setattr(reward_comparison, "simulator_decision_dt_s", lambda env: 0.1)
    monkeypatch.setattr(
        reward_comparison,
        "_initial_state_signature",
        lambda env: np.asarray([env.seed], dtype=np.float64),
    )
    monkeypatch.setattr(
        reward_comparison,
        "_initial_scene_sha256",
        lambda env: f"{env.scenario!r}:{env.seed}",
    )
    monkeypatch.setattr(
        reward_comparison,
        "constant_velocity_actions",
        lambda env: {
            agent_id: np.zeros((8, 3), dtype=np.float32)
            for agent_id in reward_comparison.AGENT_IDS
        },
    )
    monkeypatch.setattr(
        reward_comparison,
        "hard_valid_modes_by_rule_action",
        lambda agent_ids, mask: mask,
    )
    monkeypatch.setattr(
        reward_comparison,
        "joint_proposal_actions",
        lambda proposal, agent_ids: {agent_id: 0 for agent_id in agent_ids},
    )
    monkeypatch.setattr(
        reward_comparison,
        "execution_mode_valid_mask",
        lambda values, *, optimizer: values.mode_valid_mask,
    )
    monkeypatch.setattr(
        reward_comparison,
        "model_inputs_to_batch",
        lambda values, device, *, mode_valid_mask: {},
    )
    monkeypatch.setattr(reward_comparison, "_sync", lambda device: None)
    monkeypatch.setattr(reward_comparison, "planner_forward_from_batch", fake_forward)
    monkeypatch.setattr(
        reward_comparison, "optimize_selected_model_trajectories", fake_optimize
    )
    monkeypatch.setattr(
        reward_comparison,
        "diffusion_mode_feedback_actions",
        lambda modes, agent_ids, **kwargs: (
            {agent_id: 0 for agent_id in agent_ids},
            {agent_id: 0 for agent_id in agent_ids},
            {agent_id: False for agent_id in agent_ids},
        ),
    )
    monkeypatch.setattr(
        reward_comparison,
        "match_joint_action_proposal",
        lambda batch, feedback, agent_ids: batch.proposals[0],
    )
    monkeypatch.setattr(
        reward_comparison, "episode_has_ended", lambda *args, **kwargs: False
    )

    config = RewardComparisonConfig(device="cpu", seeds=(31,), max_steps=3)
    reference_states = {}
    reference_scenes = {}
    rows_by_model = {}
    for model_id in ("stage1_a", "grpo_open"):
        active_model[0] = model_id
        rows_by_model[model_id] = reward_comparison._evaluate_model(
            model_id,
            SimpleNamespace(model_id=model_id),
            device=torch.device("cpu"),
            config=config,
            reward_backend=FakeRewardBackend(),
            reward_config=reward_config,
            reference_initial_states=reference_states,
            reference_initial_scenes=reference_scenes,
        )

    expected_steps = {
        (scenario_id, step)
        for scenario_id, _ in config.scenarios
        for step in (1, 2)
    }
    for rows in rows_by_model.values():
        counts = Counter((row["scenario"], row["step"]) for row in rows)
        assert set(counts) == expected_steps
        assert set(counts.values()) == {1}
    assert scored_shapes == [(1, 3, 8, 3)] * (2 * len(expected_steps))
    assert all(
        events[index][0] == "score"
        and events[index + 1] == ("optimizer", events[index][1])
        for index in range(0, len(events), 2)
    )
    assert len(noise_by_model["stage1_a"]) == len(expected_steps)
    assert all(
        torch.equal(stage1_noise, grpo_noise)
        for stage1_noise, grpo_noise in zip(
            noise_by_model["stage1_a"], noise_by_model["grpo_open"]
        )
    )
