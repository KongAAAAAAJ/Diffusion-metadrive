from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from evaluation import gate_bev_frozen_baseline as gate
from models.bev_planner import TrajectoryOptimizationError


class _Vehicle:
    def __init__(self, speed_km_h: float) -> None:
        self.speed_km_h = speed_km_h
        self.position = np.zeros(2, dtype=np.float64)
        self.heading_theta = 0.0


class _Env:
    def __init__(self) -> None:
        self.agents = {
            agent_id: _Vehicle(18.0 + role)
            for role, agent_id in enumerate(gate.AGENT_IDS)
        }
        self.step_calls = 0
        self.closed = False

    def step(self, _action):
        self.step_calls += 1
        return (
            {},
            {},
            {"__all__": False},
            {"__all__": False},
            {agent_id: {} for agent_id in gate.AGENT_IDS},
        )

    def close(self) -> None:
        self.closed = True


def _values() -> SimpleNamespace:
    coarse = np.zeros((3, 10, 8, 3), dtype=np.float32)
    coarse[1, :, :, 0] = 1.0
    return SimpleNamespace(
        coarse_trajectories=coarse,
        ego_state=np.asarray([[5.0], [6.0], [7.0]], dtype=np.float32),
        mode_valid_mask=np.ones((3, 10), dtype=np.bool_),
    )


class _Builder:
    def __init__(self, _agent_ids) -> None:
        self.values = _values()

    def reset(self) -> None:
        return None

    def capture_state(self, _env, _timestamp) -> None:
        return None

    def history_ready(self) -> bool:
        return True

    def build_model_inputs(self, _env):
        return self.values


class _Optimizer:
    def _project_one(self, raw, _coarse, _speed, _mode):
        if float(np.asarray(raw)[0, 0]) == 1.0:
            raise TrajectoryOptimizationError("role-one exact projection audit")
        return np.asarray(raw), 0.0, object()


class _Trainer:
    def __init__(self) -> None:
        self.planner = object()
        self.inference_calls = 0

    def infer_frozen_pretrain_from_inputs(self, _batch):
        self.inference_calls += 1
        return {
            "selected_trajectory": torch.zeros((1, 3, 8, 3)),
            "selected_mode": torch.zeros((1, 3), dtype=torch.int64),
        }

    def sample_groups(self, _batch):
        raise AssertionError("the frozen baseline gate must not sample GRPO groups")


def _patch_runtime(monkeypatch, *, execute_failure: str | None = None):
    environments: list[_Env] = []
    trainer = _Trainer()

    def new_env(_scenario, _seed):
        env = _Env()
        environments.append(env)
        return env

    def condition(_rule_maker, _env, _builder, values, **_kwargs):
        return SimpleNamespace(
            model_inputs=values,
            rule_actions={agent_id: 0 for agent_id in gate.AGENT_IDS},
            is_commitment=False,
        )

    def execute(**kwargs):
        if execute_failure is not None:
            raise TrajectoryOptimizationError(execute_failure)
        step_result = kwargs["env"].step({})
        optimization = SimpleNamespace()
        diagnostics = {"forced_safe_stops": 0, "condition_failures": 0}
        return step_result, optimization, diagnostics, None, None

    monkeypatch.setattr(gate, "GATE_SCENARIOS", (("S5_hard_brake_lead", "R1"),))
    monkeypatch.setattr(gate, "GATE_SEEDS", (17, 23))
    monkeypatch.setattr(gate, "BASELINE_STEPS_PER_BUCKET", 2)
    monkeypatch.setattr(gate, "MAX_WARMUP_ENVIRONMENT_STEPS", 2)
    monkeypatch.setattr(gate, "JointBEVSampleBuilder", _Builder)
    monkeypatch.setattr(gate, "KinematicTrajectoryOptimizer", _Optimizer)
    monkeypatch.setattr(gate, "_new_env", new_env)
    monkeypatch.setattr(gate, "_new_online_rule_maker", lambda _planner, _env: object())
    monkeypatch.setattr(gate, "simulator_decision_dt_s", lambda _env: 0.1)
    monkeypatch.setattr(gate, "_scenario_ready_for_primary_sampling", lambda _env: True)
    monkeypatch.setattr(
        gate,
        "_scenario_summary",
        lambda _env: {"scenario_realized": True, "scenario_id": "S5"},
    )
    monkeypatch.setattr(gate, "_condition_online_model_inputs", condition)
    monkeypatch.setattr(
        gate,
        "execution_mode_valid_mask",
        lambda _values, optimizer: np.ones((3, 10), dtype=np.bool_),
    )
    monkeypatch.setattr(gate, "model_inputs_to_batch", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(gate, "execute_cached_frozen_baseline", execute)
    monkeypatch.setattr(
        gate,
        "load_stage1_a_for_grpo",
        lambda *_args, **_kwargs: (
            trainer,
            {"run_mode": "smoke", "dataset_fingerprint": "dataset"},
            "a" * 64,
        ),
    )
    return trainer, environments


def test_gate_runs_only_cached_frozen_baseline_once_per_state(
    monkeypatch, tmp_path: Path
) -> None:
    trainer, environments = _patch_runtime(monkeypatch)
    output = tmp_path / "report.json"

    report = gate.run_frozen_baseline_gate(
        stage1_checkpoint=tmp_path / "stage1.ckpt",
        output=output,
        device="cpu",
    )

    assert report["passed"] is True
    assert report["totals"]["baseline_execution_steps"] == 4
    assert trainer.inference_calls == 4
    assert [env.step_calls for env in environments] == [2, 2]
    assert all(env.closed for env in environments)
    assert report["execution_contract"]["grpo_sampler_called"] is False
    assert report["gates"] == {
        "all_ten_buckets_completed": True,
        "exactly_200_baseline_steps_per_bucket": True,
        "exact_total_baseline_steps": True,
        "one_inference_mask_optimizer_finalize_and_step_per_baseline_state": True,
        "no_collision_or_out_of_drivable": True,
        "sampled_grpo_candidates_absent": True,
    }
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_stop_mask_failure_records_role_speed_mode_and_exact_audit(
    monkeypatch, tmp_path: Path
) -> None:
    _trainer, environments = _patch_runtime(monkeypatch)

    def reject_stop(_values, optimizer):
        del optimizer
        raise TrajectoryOptimizationError("calibrated execution contract rejected STOP")

    monkeypatch.setattr(gate, "execution_mode_valid_mask", reject_stop)
    output = tmp_path / "stop_failure.json"
    report = gate.run_frozen_baseline_gate(
        stage1_checkpoint=tmp_path / "stage1.ckpt",
        output=output,
        device="cpu",
    )

    assert report["passed"] is False
    assert len(report["buckets"]) == 1
    failure = report["buckets"][0]["failure"]
    assert failure["stage"] == "execution_mode_valid_mask"
    assert failure["exception"] == "calibrated execution contract rejected STOP"
    assert failure["selected_modes"] is None
    assert failure["model_input_speeds_mps"] == [5.0, 6.0, 7.0]
    assert failure["failed_roles"] == [1]
    assert failure["projection_audits"]["stop"][1] == {
        "role": 1,
        "agent_id": "agent1",
        "mode": int(gate.ModeIndex.STOP),
        "speed_mps": 6.0,
        "accepted": False,
        "exception": "role-one exact projection audit",
    }
    assert environments[0].step_calls == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_selected_optimizer_failure_records_frozen_modes_and_projection_audits(
    monkeypatch, tmp_path: Path
) -> None:
    trainer, environments = _patch_runtime(
        monkeypatch,
        execute_failure="trajectory group=0 role=1 mode=0: selected failure",
    )
    output = tmp_path / "optimizer_failure.json"
    report = gate.run_frozen_baseline_gate(
        stage1_checkpoint=tmp_path / "stage1.ckpt",
        output=output,
        device="cpu",
    )

    assert report["passed"] is False
    assert trainer.inference_calls == 1
    failure = report["buckets"][0]["failure"]
    assert failure["stage"] == "baseline_optimizer_or_rule_finalize"
    assert failure["selected_modes"] == [0, 0, 0]
    assert failure["exception"].endswith("selected failure")
    assert failure["failed_roles"] == [1]
    assert "selected" in failure["projection_audits"]
    assert environments[0].step_calls == 0
