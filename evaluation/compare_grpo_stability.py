"""Validate the bounded single-step mode-isolated GRPO diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


MANIFEST_FORMAT = "stage2_grpo_stability_manifest_v7"
REPORT_FORMAT = "stage2_grpo_stability_report_v7"
ARMS = ("single_step_fixed_scale_mode_residual",)
PAIRED_SEEDS = (17, 23, 42)
SCENARIO_SEEDS = (17, 23)
VALIDATION_SEEDS = (31, 47)
TARGET_ACCEPTED_UPDATE_STATES = 500
VALIDATION_INTERVAL_ROLLOUTS = 20
TAIL_STATES = (420, 440, 460, 480, 500)
TRAJECTORIES_PER_MODE = 48
LEARNING_RATE = 1e-5
BC_WEIGHT = 0.1
REFERENCE_KL_WEIGHT = 0.02
MAX_GRAD_NORM = 1.0
ADVANTAGE_SCALE = 1.0
BASELINE_TOLERANCE = 1e-6
POST_UPDATE_REFERENCE_KL_MAX = 0.25
MAX_ADAPTER_RELATIVE_DRIFT = 0.02
ALIGNED_DDIM_PATH = {
    "timesteps": [8, 5, 3, 0],
    "previous_timesteps": [5, 3, 0, -1],
    "eta": 1.0,
    "decoder_calls": 4,
    "stochastic_log_probability_transitions": 3,
}


class GRPOStabilityComparisonError(ValueError):
    """Raised when a diagnostic manifest or run artifact violates v7."""


@dataclass(frozen=True)
class _Run:
    arm: str
    seed: int
    run_dir: Path
    states: np.ndarray
    simulator_gain: np.ndarray
    selected_gain: np.ndarray
    s7_out_delta: np.ndarray
    safety_eligible_checkpoint: bool
    training_status: str

    def value_at(self, state: int, values: np.ndarray) -> float:
        indices = np.flatnonzero(self.states == state)
        if indices.size != 1:
            raise GRPOStabilityComparisonError(
                f"{self.arm}/seed{self.seed} is missing validation state {state}"
            )
        return float(values[int(indices[0])])

    @property
    def tail_simulator_gain(self) -> float:
        return float(
            np.mean(
                [self.value_at(state, self.simulator_gain) for state in TAIL_STATES]
            )
        )

    @property
    def first_difference_std(self) -> float | None:
        if self.simulator_gain.size < 2:
            return None
        return float(np.std(np.diff(self.simulator_gain)))


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GRPOStabilityComparisonError(
            f"unable to read {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise GRPOStabilityComparisonError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GRPOStabilityComparisonError(
            f"unable to read validation state bank: {path}"
        ) from exc
    return digest.hexdigest()


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GRPOStabilityComparisonError(f"{label} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GRPOStabilityComparisonError(
            f"{label} must be a SHA256 digest"
        ) from exc
    return value.lower()


def _commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise GRPOStabilityComparisonError(f"{label} must be a full Git commit")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GRPOStabilityComparisonError(
            f"{label} must be a full Git commit"
        ) from exc
    return value.lower()


def _resolve(base: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GRPOStabilityComparisonError(f"{label} must be a path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _finite(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GRPOStabilityComparisonError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise GRPOStabilityComparisonError(f"{label} must be finite")
    return result


def _arm_specs(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(ARMS):
        raise GRPOStabilityComparisonError(
            "manifest arms must contain the fixed-scale residual arm"
        )
    result: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        raw = value[arm]
        if not isinstance(raw, Mapping):
            raise GRPOStabilityComparisonError(
                f"manifest arm {arm} must be an object"
            )
        required = {
            "commit",
            "online_config_format",
            "online_report_format",
            "ddim_path",
            "advantage",
            "policy_update",
            "anchor_scope",
            "trainable_parameters",
            "stability_guard",
        }
        if set(raw) != required:
            raise GRPOStabilityComparisonError(
                f"manifest arm {arm} fields mismatch"
            )
        spec = {str(name): item for name, item in raw.items()}
        spec["commit"] = _commit(spec["commit"], label=f"{arm} commit")
        if spec["ddim_path"] != ALIGNED_DDIM_PATH:
            raise GRPOStabilityComparisonError(
                f"{arm} must use the aligned DDIM path"
            )
        if (
            spec["advantage"] != "fixed_scale_baseline_relative_safe_truncation"
            or spec["policy_update"] != "single_step_on_policy"
            or spec["anchor_scope"] != "all_valid_executable"
            or spec["trainable_parameters"] != "mode_specific_output_residuals"
            or spec["stability_guard"]
            != {
                "post_update_reference_kl_max": POST_UPDATE_REFERENCE_KL_MAX,
                "max_adapter_relative_drift": MAX_ADAPTER_RELATIVE_DRIFT,
            }
        ):
            raise GRPOStabilityComparisonError(
                f"{arm} optimization semantics mismatch"
            )
        result[arm] = spec
    return result


def _validate_common(value: object) -> None:
    expected = {
        "trajectories_per_mode": TRAJECTORIES_PER_MODE,
        "learning_rate": LEARNING_RATE,
        "bc_weight": BC_WEIGHT,
        "reference_kl_weight": REFERENCE_KL_WEIGHT,
        "max_grad_norm": MAX_GRAD_NORM,
        "advantage_scale": ADVANTAGE_SCALE,
        "baseline_tolerance": BASELINE_TOLERANCE,
        "weight_decay": 0.0,
    }
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise GRPOStabilityComparisonError("manifest hyperparameters mismatch")


def _history(
    report: Mapping[str, object], *, label: str
) -> tuple[np.ndarray, ...]:
    raw = report.get("validation_selection_history")
    if not isinstance(raw, list) or not raw:
        raise GRPOStabilityComparisonError(f"{label} has no validation history")
    records: list[tuple[int, float, float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise GRPOStabilityComparisonError(
                f"{label} history entry {index} is invalid"
            )
        state_value = _finite(
            item.get("accepted_update_state"), label=f"{label} state"
        )
        state = int(state_value)
        if (
            state_value != state
            or state <= 0
            or state % VALIDATION_INTERVAL_ROLLOUTS
        ):
            raise GRPOStabilityComparisonError(
                f"{label} validation state is invalid"
            )
        records.append(
            (
                state,
                _finite(
                    item.get("simulator_reward_gain"),
                    label=f"{label} simulator gain",
                ),
                _finite(
                    item.get("selected_reward_gain"),
                    label=f"{label} selected gain",
                ),
                _finite(
                    item.get("s7_out_delta"), label=f"{label} S7 out delta"
                ),
            )
        )
    records.sort(key=lambda row: row[0])
    states = np.asarray([row[0] for row in records], dtype=np.int64)
    if np.unique(states).size != states.size:
        raise GRPOStabilityComparisonError(
            f"{label} has duplicate validation states"
        )
    return (
        states,
        np.asarray([row[1] for row in records], dtype=np.float64),
        np.asarray([row[2] for row in records], dtype=np.float64),
        np.asarray([row[3] for row in records], dtype=np.float64),
    )


def _validate_semantics(
    config: Mapping[str, object], *, arm: str, spec: Mapping[str, object]
) -> None:
    policy = config.get("policy_update_contract")
    collection = config.get("rollout_collection_contract")
    if not isinstance(policy, Mapping) or not isinstance(collection, Mapping):
        raise GRPOStabilityComparisonError(
            f"{arm} is missing optimizer contracts"
        )
    if policy.get("version") != "stage2_joint_grpo_optimizer_v8":
        raise GRPOStabilityComparisonError(
            f"{arm} optimizer contract is not v8"
        )
    if collection.get("version") != "stage2_joint_grpo_persistent_episode_v7":
        raise GRPOStabilityComparisonError(
            f"{arm} collection contract is not v7"
        )
    if policy.get("ppo_ratio_or_clipping") is not False:
        raise GRPOStabilityComparisonError(f"{arm} still enables PPO replay")
    if policy.get("ddim_path") != ALIGNED_DDIM_PATH:
        raise GRPOStabilityComparisonError(f"{arm} optimizer DDIM path mismatch")
    if "collision_or_out=-1" not in str(policy.get("advantage", "")):
        raise GRPOStabilityComparisonError(
            f"{arm} does not record fixed-scale safety truncation"
        )
    regularization = str(policy.get("reference_regularization", ""))
    if "every hard-valid optimizer-executable mode" not in regularization:
        raise GRPOStabilityComparisonError(
            f"{arm} does not anchor all valid modes"
        )
    if policy.get("trainable_parameters") != (
        "ten zero-initialized mode-specific output residual weights and biases"
    ):
        raise GRPOStabilityComparisonError(
            f"{arm} does not record isolated residual parameters"
        )


def _load_run(
    *,
    root: Path,
    raw: Mapping[str, object],
    specs: Mapping[str, Mapping[str, object]],
    source_sha: str,
) -> _Run:
    arm = raw.get("arm")
    seed = raw.get("seed")
    if arm not in ARMS or isinstance(seed, bool) or seed not in PAIRED_SEEDS:
        raise GRPOStabilityComparisonError("run arm/seed is invalid")
    arm = str(arm)
    seed = int(seed)
    run_dir = _resolve(
        root, raw.get("run_dir"), label=f"{arm}/seed{seed} run_dir"
    )
    config = _load_json(
        run_dir / "config.json", label=f"{arm}/seed{seed} config"
    )
    report = _load_json(
        run_dir / "report.json", label=f"{arm}/seed{seed} report"
    )
    spec = specs[arm]
    if config.get("format") != spec["online_config_format"]:
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} config format mismatch"
        )
    if report.get("format") != spec["online_report_format"]:
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} report format mismatch"
        )
    run_commit = _commit(
        config.get("implementation_commit"), label="run implementation_commit"
    )
    if run_commit != spec["commit"]:
        raise GRPOStabilityComparisonError(f"{arm}/seed{seed} commit mismatch")
    if (
        _digest(config.get("source_stage1_sha256"), label="run source SHA")
        != source_sha
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} source checkpoint mismatch"
        )
    online = config.get("online_config")
    if not isinstance(online, Mapping):
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} online config is missing"
        )
    expected_online = {
        "seed": seed,
        "trajectories_per_mode": TRAJECTORIES_PER_MODE,
        "total_rollout_groups": TARGET_ACCEPTED_UPDATE_STATES,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
    }
    for name, expected in expected_online.items():
        if online.get(name) != expected:
            raise GRPOStabilityComparisonError(
                f"{arm}/seed{seed} {name} mismatch"
            )
    if (
        config.get("run_mode") != "smoke"
        or report.get("diagnostic_only") is not True
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} must be diagnostic-only"
        )
    status = str(report.get("training_status"))
    if status not in (
        "complete",
        "diagnostic_early_stop",
        "stability_guard_rejected",
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} training status is invalid"
        )
    _validate_semantics(config, arm=arm, spec=spec)
    if status == "stability_guard_rejected" and not report.get(
        "validation_selection_history"
    ):
        states = np.asarray([], dtype=np.int64)
        simulator = selected = s7 = np.asarray([], dtype=np.float64)
    else:
        states, simulator, selected, s7 = _history(
            report, label=f"{arm}/seed{seed}"
        )
    accepted = int(report.get("accepted_update_states", -1))
    if status == "complete" and accepted != TARGET_ACCEPTED_UPDATE_STATES:
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} did not reach state 500"
        )
    eligible = report.get("checkpoint_selection_status") == "eligible"
    if eligible != bool(report.get("best_checkpoint")):
        raise GRPOStabilityComparisonError(
            f"{arm}/seed{seed} best checkpoint status conflicts"
        )
    return _Run(
        arm=arm,
        seed=seed,
        run_dir=run_dir,
        states=states,
        simulator_gain=simulator,
        selected_gain=selected,
        s7_out_delta=s7,
        safety_eligible_checkpoint=eligible,
        training_status=status,
    )


def _plot(runs: Sequence[_Run], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    for arm in ARMS:
        arm_runs = [run for run in runs if run.arm == arm]
        common = sorted(
            set.intersection(*(set(run.states.tolist()) for run in arm_runs))
        )
        if not common:
            continue
        means = [
            np.mean(
                [run.value_at(state, run.simulator_gain) for run in arm_runs]
            )
            for state in common
        ]
        axis.plot(common, means, label=arm)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Accepted update state")
    axis.set_ylabel("Closed-loop simulator reward gain")
    axis.set_title("Bounded single-step fixed-scale GRPO diagnostic")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _tail_gain_or_none(run: _Run) -> float | None:
    if not all(state in run.states for state in TAIL_STATES):
        return None
    return run.tail_simulator_gain


def compare_grpo_stability(
    manifest_path: Path, output_dir: Path
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_json(manifest_path, label="GRPO stability manifest")
    required = {
        "format",
        "diagnostic_only",
        "source_stage1_sha256",
        "paired_seeds",
        "scenario_seeds",
        "validation_seeds",
        "target_accepted_update_states",
        "validation_interval_rollouts",
        "validation_state_bank",
        "validation_state_bank_sha256",
        "hyperparameters",
        "arms",
        "historical_control_run27",
        "runs",
    }
    if set(manifest) != required or manifest.get("format") != MANIFEST_FORMAT:
        raise GRPOStabilityComparisonError(
            "manifest fields or format mismatch"
        )
    if manifest.get("diagnostic_only") is not True:
        raise GRPOStabilityComparisonError(
            "stability comparison must be diagnostic-only"
        )
    expected_scalars = {
        "paired_seeds": list(PAIRED_SEEDS),
        "scenario_seeds": list(SCENARIO_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "target_accepted_update_states": TARGET_ACCEPTED_UPDATE_STATES,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
    }
    for name, expected in expected_scalars.items():
        if manifest.get(name) != expected:
            raise GRPOStabilityComparisonError(f"manifest {name} mismatch")
    _validate_common(manifest.get("hyperparameters"))
    specs = _arm_specs(manifest.get("arms"))
    historical = manifest.get("historical_control_run27")
    if (
        not isinstance(historical, Mapping)
        or set(historical)
        != {"run_dir", "training_status", "accepted_update_states"}
        or historical.get("training_status") != "diagnostic_early_stop"
        or historical.get("accepted_update_states") != 240
    ):
        raise GRPOStabilityComparisonError(
            "historical_control_run27 must record the stopped 240-state run"
        )
    historical_run_dir = _resolve(
        manifest_path.parent,
        historical.get("run_dir"),
        label="historical_control_run27 run_dir",
    )
    historical_report = _load_json(
        historical_run_dir / "report.json",
        label="historical_control_run27 report",
    )
    if (
        historical_report.get("training_status") != "diagnostic_early_stop"
        or historical_report.get("accepted_update_states") != 240
    ):
        raise GRPOStabilityComparisonError(
            "historical_control_run27 report conflicts with the manifest"
        )
    source_sha = _digest(
        manifest.get("source_stage1_sha256"), label="source SHA"
    )
    bank_path = _resolve(
        manifest_path.parent,
        manifest.get("validation_state_bank"),
        label="validation_state_bank",
    )
    bank_sha = _digest(
        manifest.get("validation_state_bank_sha256"),
        label="validation_state_bank_sha256",
    )
    if _sha256(bank_path) != bank_sha:
        raise GRPOStabilityComparisonError(
            "validation state bank digest mismatch"
        )
    raw_runs = manifest.get("runs")
    if (
        not isinstance(raw_runs, list)
        or len(raw_runs) != len(PAIRED_SEEDS)
    ):
        raise GRPOStabilityComparisonError(
            "manifest must contain exactly three fixed-scale residual runs"
        )
    runs: list[_Run] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_runs:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"arm", "seed", "run_dir"}
        ):
            raise GRPOStabilityComparisonError(
                "manifest run entry fields mismatch"
            )
        run = _load_run(
            root=manifest_path.parent,
            raw=raw,
            specs=specs,
            source_sha=source_sha,
        )
        key = (run.arm, run.seed)
        if key in seen:
            raise GRPOStabilityComparisonError(
                "manifest has a duplicate arm/seed run"
            )
        seen.add(key)
        runs.append(run)
    if seen != {(arm, seed) for arm in ARMS for seed in PAIRED_SEEDS}:
        raise GRPOStabilityComparisonError(
            "manifest arm/seed matrix is incomplete"
        )

    by_key = {(run.arm, run.seed): run for run in runs}
    stable = [
        by_key[("single_step_fixed_scale_mode_residual", seed)]
        for seed in PAIRED_SEEDS
    ]
    eligible_all = all(run.safety_eligible_checkpoint for run in stable)
    guard_rejected = any(
        run.training_status == "stability_guard_rejected" for run in stable
    )
    stable_tail = [_tail_gain_or_none(run) for run in stable]
    complete_tail = [value for value in stable_tail if value is not None]
    tail_nonnegative_count = sum(value >= 0.0 for value in complete_tail)
    passed = (
        not guard_rejected
        and eligible_all
        and tail_nonnegative_count >= 2
        and len(complete_tail) == len(PAIRED_SEEDS)
        and float(np.mean(complete_tail)) >= 0.0
        and sum(value > 0.0 for value in complete_tail) >= 2
        and float(np.mean(complete_tail)) > 0.0
    )
    output_dir = Path(output_dir).resolve()
    plot_path = _plot(
        runs, output_dir / "simulator_reward_gain_fixed_scale_residual.png"
    )
    report: dict[str, object] = {
        "format": REPORT_FORMAT,
        "diagnostic_only": True,
        "eligible_for_formal_conclusions": False,
        "manifest": str(manifest_path),
        "validation_state_bank": str(bank_path),
        "validation_state_bank_sha256": bank_sha,
        "historical_control_run27": {
            **dict(historical),
            "run_dir": str(historical_run_dir),
        },
        "arms": {name: dict(specs[name]) for name in ARMS},
        "per_run": [
            {
                "arm": run.arm,
                "seed": run.seed,
                "run_dir": str(run.run_dir),
                "training_status": run.training_status,
                "safety_eligible_checkpoint": run.safety_eligible_checkpoint,
                "tail_simulator_gain_mean": (
                    _tail_gain_or_none(run)
                ),
                "simulator_gain_first_difference_std": (
                    run.first_difference_std
                ),
            }
            for run in runs
        ],
        "stable_arm_500_state_gate": {
            "all_three_seeds_have_safety_eligible_checkpoint": eligible_all,
            "any_stability_guard_rejection": guard_rejected,
            "tail_nonnegative_seed_count": tail_nonnegative_count,
            "tail_simulator_gain_cross_seed_mean": (
                float(np.mean(complete_tail))
                if len(complete_tail) == len(PAIRED_SEEDS)
                else None
            ),
            "paired_frozen_tail_gain_positive_seed_count": sum(
                value > 0.0 for value in complete_tail
            ),
            "passed": passed,
        },
        "plot": str(plot_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_grpo_stability(args.manifest, args.output_dir)
    print(json.dumps(report["stable_arm_500_state_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALIGNED_DDIM_PATH",
    "ARMS",
    "GRPOStabilityComparisonError",
    "MANIFEST_FORMAT",
    "PAIRED_SEEDS",
    "REPORT_FORMAT",
    "TAIL_STATES",
    "compare_grpo_stability",
    "main",
]
