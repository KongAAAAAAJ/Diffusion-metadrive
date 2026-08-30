"""Compare paired single-epoch and clipped multi-epoch GRPO diagnostics.

The input manifest is a JSON object with this exact shape (``run_dir`` may be
absolute or relative to the manifest):

.. code-block:: json

   {
     "format": "stage2_grpo_stability_ab_manifest_v1",
     "source_stage1_sha256": "<64 lowercase hex characters>",
     "paired_seeds": [17, 23, 42],
     "baseline_update_epochs": 1,
     "clipped_update_epochs": 4,
     "group_size": 24,
     "total_rollout_groups": 100,
     "validation_interval_rollouts": 20,
     "clip_epsilon": 0.2,
     "scenario_seeds": [17, 23],
     "validation_seeds": [31, 47],
     "runs": [
       {"arm": "baseline", "seed": 17, "run_dir": "run_8"},
       {"arm": "clipped", "seed": 17, "run_dir": "run_9"},
       {"arm": "baseline", "seed": 23, "run_dir": "run_10"},
       {"arm": "clipped", "seed": 23, "run_dir": "run_11"},
       {"arm": "baseline", "seed": 42, "run_dir": "run_12"},
       {"arm": "clipped", "seed": 42, "run_dir": "run_13"}
     ]
   }

``runs`` must contain exactly one baseline and one clipped entry for each of
the three paired seeds (six entries total). The command validates each run's
v4 config/report, the complete frozen ``JointGRPOConfig``, and the
``validation/reward_gain`` TensorBoard series, then writes ``report.json`` and
``validation_reward_gain_ab.png``. Results remain diagnostic-only and do not
authorize formal conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


MANIFEST_FORMAT = "stage2_grpo_stability_ab_manifest_v1"
REPORT_FORMAT = "stage2_grpo_stability_ab_report_v1"
ONLINE_CONFIG_FORMAT = "bev_joint_grpo_online_config_v4"
ONLINE_REPORT_FORMAT = "bev_joint_grpo_online_report_v4"
VALIDATION_REWARD_GAIN_TAG = "validation/reward_gain"
PAIRED_SEEDS = (17, 23, 42)
ARM_UPDATE_EPOCHS = {"baseline": 1, "clipped": 4}
GROUP_SIZE = 24
TOTAL_ROLLOUT_GROUPS = 100
VALIDATION_INTERVAL_ROLLOUTS = 20
CLIP_EPSILON = 0.2
SCENARIO_SEEDS = (17, 23)
VALIDATION_SEEDS = (31, 47)
SCENARIOS = (
    ("S5_hard_brake_lead", "R1_entry_straight"),
    ("S6_background_merge_in", "R6_mainline_merge_approach"),
    ("S7_ego_merge_from_ramp", "R7_merge_core"),
    ("S8_ego_exit_to_ramp", "R6_exit_to_ramp"),
    ("S9_narrow_channel_negotiation", "R8_narrow_channel"),
)
EXPECTED_GRPO_CONFIG = {
    "group_size": GROUP_SIZE,
    "initial_noise_timestep": 8,
    "denoise_steps": 4,
    "eta": 1.0,
    "mode_pg_weight": 1.0,
    "trajectory_pg_weight": 1.0,
    "bc_weight": 0.1,
    "reference_kl_weight": 0.02,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "advantage_eps": 1e-6,
    "xy_beta_m": 1.0,
    "heading_beta_rad": 0.1,
    "heading_bc_weight": 0.2,
}


class GRPOStabilityComparisonError(ValueError):
    """Raised when a persisted clipped-GRPO A/B input is invalid."""


@dataclass(frozen=True)
class _RunSeries:
    arm: str
    seed: int
    run_dir: Path
    rollout_steps: np.ndarray
    reward_gain: np.ndarray
    optimizer_steps: int
    wall_time_seconds: float

    @property
    def adjacent_delta_std(self) -> float:
        return float(np.std(np.diff(self.reward_gain), ddof=0))

    @property
    def final_reward_gain(self) -> float:
        return float(self.reward_gain[-1])


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GRPOStabilityComparisonError(
            f"unable to read {label}: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise GRPOStabilityComparisonError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise GRPOStabilityComparisonError(
            f"{label} fields mismatch: expected {sorted(expected)}"
        )


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GRPOStabilityComparisonError(f"{label} must be a positive integer")
    return int(value)


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GRPOStabilityComparisonError(
            f"{label} must be a non-negative integer"
        )
    return int(value)


def _finite_float(value: object, *, label: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive " if positive else ""
        raise GRPOStabilityComparisonError(
            f"{label} must be a {qualifier}finite scalar"
        )
    return float(value)


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise GRPOStabilityComparisonError(f"{label} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GRPOStabilityComparisonError(
            f"{label} must be a SHA256 digest"
        ) from exc
    return value


def _validate_grpo_config(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GRPOStabilityComparisonError(f"{label} must be an object")
    _require_exact_keys(value, set(EXPECTED_GRPO_CONFIG), label=label)
    for name, expected in EXPECTED_GRPO_CONFIG.items():
        actual = value.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise GRPOStabilityComparisonError(
                f"{label} {name} mismatch: expected {expected!r}"
            )
    return {str(name): item for name, item in value.items()}


def _load_reward_gain(tb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    if VALIDATION_REWARD_GAIN_TAG not in accumulator.Tags().get("scalars", ()):
        raise GRPOStabilityComparisonError(
            f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} was not found"
        )
    events = accumulator.Scalars(VALIDATION_REWARD_GAIN_TAG)
    if not events:
        raise GRPOStabilityComparisonError(
            f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} contains no events"
        )
    records: list[tuple[int, float]] = []
    observed_steps: set[int] = set()
    for event in events:
        step = int(event.step)
        if step in observed_steps:
            raise GRPOStabilityComparisonError(
                f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} "
                f"has duplicate rollout step {step}"
            )
        observed_steps.add(step)
        value = float(event.value)
        if not math.isfinite(value):
            raise GRPOStabilityComparisonError(
                f"TensorBoard tag {VALIDATION_REWARD_GAIN_TAG!r} "
                f"step {step} must be finite"
            )
        records.append((step, value))
    records.sort(key=lambda item: item[0])
    steps = np.asarray([item[0] for item in records], dtype=np.int64)
    values = np.asarray([item[1] for item in records], dtype=np.float64)
    expected = np.arange(
        VALIDATION_INTERVAL_ROLLOUTS,
        TOTAL_ROLLOUT_GROUPS + 1,
        VALIDATION_INTERVAL_ROLLOUTS,
        dtype=np.int64,
    )
    if not np.array_equal(steps, expected):
        raise GRPOStabilityComparisonError(
            "validation/reward_gain must use fresh-rollout steps "
            f"{expected.tolist()}, got {steps.tolist()}"
        )
    return steps, values


def _validate_manifest(path: Path) -> tuple[str, tuple[Mapping[str, object], ...]]:
    manifest = _load_json(path, label="GRPO stability A/B manifest")
    _require_exact_keys(
        manifest,
        {
            "format",
            "source_stage1_sha256",
            "paired_seeds",
            "baseline_update_epochs",
            "clipped_update_epochs",
            "group_size",
            "total_rollout_groups",
            "validation_interval_rollouts",
            "clip_epsilon",
            "scenario_seeds",
            "validation_seeds",
            "runs",
        },
        label="GRPO stability A/B manifest",
    )
    expected_values = {
        "format": MANIFEST_FORMAT,
        "paired_seeds": list(PAIRED_SEEDS),
        "baseline_update_epochs": ARM_UPDATE_EPOCHS["baseline"],
        "clipped_update_epochs": ARM_UPDATE_EPOCHS["clipped"],
        "group_size": GROUP_SIZE,
        "total_rollout_groups": TOTAL_ROLLOUT_GROUPS,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
        "clip_epsilon": CLIP_EPSILON,
        "scenario_seeds": list(SCENARIO_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
    }
    for name, expected in expected_values.items():
        actual = manifest.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise GRPOStabilityComparisonError(f"manifest {name} mismatch")
    source_sha = _sha256(
        manifest.get("source_stage1_sha256"),
        label="manifest source_stage1_sha256",
    )
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 6:
        raise GRPOStabilityComparisonError("manifest runs must contain six entries")
    runs: list[Mapping[str, object]] = []
    observed: set[tuple[str, int]] = set()
    for index, value in enumerate(raw_runs):
        if not isinstance(value, Mapping):
            raise GRPOStabilityComparisonError(
                f"manifest runs[{index}] must be an object"
            )
        _require_exact_keys(value, {"arm", "seed", "run_dir"}, label=f"runs[{index}]")
        arm = value.get("arm")
        seed = value.get("seed")
        run_dir = value.get("run_dir")
        if not isinstance(arm, str) or arm not in ARM_UPDATE_EPOCHS:
            raise GRPOStabilityComparisonError(f"runs[{index}].arm is invalid")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in PAIRED_SEEDS
        ):
            raise GRPOStabilityComparisonError(f"runs[{index}].seed is invalid")
        if not isinstance(run_dir, str) or not run_dir:
            raise GRPOStabilityComparisonError(
                f"runs[{index}].run_dir must be a non-empty string"
            )
        identity = (str(arm), int(seed))
        if identity in observed:
            raise GRPOStabilityComparisonError(
                f"manifest contains duplicate run {identity}"
            )
        observed.add(identity)
        runs.append(value)
    expected_identities = {
        (arm, seed) for arm in ARM_UPDATE_EPOCHS for seed in PAIRED_SEEDS
    }
    if observed != expected_identities:
        raise GRPOStabilityComparisonError(
            "manifest must contain one baseline and clipped run for each paired seed"
        )
    return source_sha, tuple(runs)


def _run_binding(config: Mapping[str, object]) -> dict[str, object]:
    online = config.get("online_config")
    if not isinstance(online, Mapping):
        raise GRPOStabilityComparisonError("run config online_config is invalid")
    policy_update = config.get("policy_update_contract")
    if not isinstance(policy_update, Mapping):
        raise GRPOStabilityComparisonError(
            "run config policy_update_contract is invalid"
        )
    return {
        "grpo_config": config.get("grpo_config"),
        "source_stage1_sha256": config.get("source_stage1_sha256"),
        "reward_contract_sha256": config.get("reward_contract_sha256"),
        "reward_config_sha256": config.get("reward_config_sha256"),
        "reward_application_contract_sha256": config.get(
            "reward_application_contract_sha256"
        ),
        "trajectory_optimizer_sha256": config.get(
            "trajectory_optimizer_sha256"
        ),
        "scenario_contract_sha256": config.get("scenario_contract_sha256"),
        "policy_update_contract_common": {
            str(name): value
            for name, value in policy_update.items()
            if name != "update_epochs"
        },
        "device": online.get("device"),
        "scenarios": online.get("scenarios"),
        "scenario_seeds": online.get("scenario_seeds"),
        "environment_steps_per_episode": online.get(
            "environment_steps_per_episode"
        ),
        "group_size": online.get("group_size"),
        "total_rollout_groups": online.get("total_rollout_groups"),
        "clip_epsilon": online.get("clip_epsilon"),
        "validation_interval_rollouts": online.get(
            "validation_interval_rollouts"
        ),
    }


def _load_run(
    manifest_path: Path,
    entry: Mapping[str, object],
    *,
    source_sha: str,
    expected_binding: Mapping[str, object] | None,
) -> tuple[_RunSeries, dict[str, object]]:
    arm = str(entry["arm"])
    seed = int(entry["seed"])
    run_dir = Path(str(entry["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = manifest_path.parent / run_dir
    run_dir = run_dir.resolve()
    config = _load_json(run_dir / "config.json", label=f"{arm}/{seed} config")
    report = _load_json(run_dir / "report.json", label=f"{arm}/{seed} report")
    if config.get("format") != ONLINE_CONFIG_FORMAT:
        raise GRPOStabilityComparisonError(f"{arm}/{seed} config format mismatch")
    if report.get("format") != ONLINE_REPORT_FORMAT:
        raise GRPOStabilityComparisonError(f"{arm}/{seed} report format mismatch")
    config_grpo = _validate_grpo_config(
        config.get("grpo_config"), label=f"{arm}/{seed} config grpo_config"
    )
    report_grpo = _validate_grpo_config(
        report.get("grpo_config"), label=f"{arm}/{seed} report grpo_config"
    )
    if report_grpo != config_grpo:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} config/report grpo_config mismatch"
        )
    online = config.get("online_config")
    if not isinstance(online, Mapping):
        raise GRPOStabilityComparisonError(f"{arm}/{seed} online_config is invalid")
    expected_online = {
        "device": "cuda",
        "seed": seed,
        "group_size": GROUP_SIZE,
        "total_rollout_groups": TOTAL_ROLLOUT_GROUPS,
        "update_epochs": ARM_UPDATE_EPOCHS[arm],
        "clip_epsilon": CLIP_EPSILON,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
        "environment_steps_per_episode": 100,
        "scenarios": [list(value) for value in SCENARIOS],
        "scenario_seeds": list(SCENARIO_SEEDS),
        "resume_checkpoint": None,
    }
    for name, expected in expected_online.items():
        actual = online.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise GRPOStabilityComparisonError(
                f"{arm}/{seed} online_config {name} mismatch"
            )
    if config.get("source_stage1_sha256") != source_sha:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} source Stage1 checkpoint mismatch"
        )
    contract = config.get("policy_update_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("version") != "stage2_joint_grpo_optimizer_v2"
        or contract.get("update_epochs") != ARM_UPDATE_EPOCHS[arm]
        or contract.get("clip_epsilon") != CLIP_EPSILON
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} policy update contract mismatch"
        )
    binding = _run_binding(config)
    if expected_binding is not None and binding != dict(expected_binding):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} fairness binding differs from the other runs"
        )

    if report.get("diagnostic_only") is not True or report.get(
        "eligible_for_formal_training"
    ) is not False:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} must remain diagnostic-only"
        )
    if report.get("checkpoint_round_trip") is not True:
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} checkpoint round-trip did not pass"
        )
    sampled = _positive_int(
        report.get("sampled_rollouts"), label=f"{arm}/{seed} sampled_rollouts"
    )
    uninformative = _non_negative_int(
        report.get("uninformative_rollouts"),
        label=f"{arm}/{seed} uninformative_rollouts",
    )
    optimizer_steps = _positive_int(
        report.get("optimizer_steps"), label=f"{arm}/{seed} optimizer_steps"
    )
    target_rollouts = _positive_int(
        report.get("target_rollout_groups"),
        label=f"{arm}/{seed} target_rollout_groups",
    )
    sampled_this_run = _positive_int(
        report.get("sampled_rollouts_this_run"),
        label=f"{arm}/{seed} sampled_rollouts_this_run",
    )
    optimizer_steps_this_run = _positive_int(
        report.get("optimizer_steps_this_run"),
        label=f"{arm}/{seed} optimizer_steps_this_run",
    )
    if (
        sampled != TOTAL_ROLLOUT_GROUPS
        or target_rollouts != TOTAL_ROLLOUT_GROUPS
        or sampled_this_run != TOTAL_ROLLOUT_GROUPS
        or uninformative > sampled
    ):
        raise GRPOStabilityComparisonError(f"{arm}/{seed} rollout counters mismatch")
    expected_optimizer_steps = (
        sampled - uninformative
    ) * ARM_UPDATE_EPOCHS[arm]
    if (
        optimizer_steps != expected_optimizer_steps
        or optimizer_steps_this_run != expected_optimizer_steps
    ):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} optimizer-step count mismatch"
        )
    if report.get("validation_seeds") != list(VALIDATION_SEEDS):
        raise GRPOStabilityComparisonError(
            f"{arm}/{seed} validation seeds mismatch"
        )
    wall_time_seconds = _finite_float(
        report.get("wall_time_seconds"),
        label=f"{arm}/{seed} wall_time_seconds",
        positive=True,
    )
    steps, values = _load_reward_gain(run_dir / "tb")
    return (
        _RunSeries(
            arm=arm,
            seed=seed,
            run_dir=run_dir,
            rollout_steps=steps,
            reward_gain=values,
            optimizer_steps=optimizer_steps,
            wall_time_seconds=wall_time_seconds,
        ),
        binding,
    )


def _render_comparison_plot(
    runs: Mapping[tuple[str, int], _RunSeries], output_path: Path
) -> Path:
    fig, (gain_ax, stability_ax) = plt.subplots(2, 1, figsize=(10.0, 8.4))
    colors = {"baseline": "#4C78A8", "clipped": "#E45756"}
    for arm in ("baseline", "clipped"):
        arm_runs = [runs[(arm, seed)] for seed in PAIRED_SEEDS]
        stacked = np.stack([run.reward_gain for run in arm_runs], axis=0)
        for run in arm_runs:
            gain_ax.plot(
                run.rollout_steps,
                run.reward_gain,
                color=colors[arm],
                linewidth=1.0,
                alpha=0.32,
                label=f"{arm} seed {run.seed}",
            )
        gain_ax.plot(
            arm_runs[0].rollout_steps,
            stacked.mean(axis=0),
            color=colors[arm],
            linewidth=2.5,
            marker="o",
            label=f"{arm} mean",
        )
    gain_ax.axhline(0.0, color="#777777", linewidth=0.8)
    gain_ax.set_xlabel("Fresh rollout group")
    gain_ax.set_ylabel("Validation reward gain")
    gain_ax.set_title("Paired GRPO validation reward gain")
    gain_ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.75)
    gain_ax.legend(frameon=False, fontsize=7.5, ncol=2)

    x = np.arange(len(PAIRED_SEEDS), dtype=np.float64)
    width = 0.34
    for offset, arm in ((-width / 2, "baseline"), (width / 2, "clipped")):
        stability_ax.bar(
            x + offset,
            [runs[(arm, seed)].adjacent_delta_std for seed in PAIRED_SEEDS],
            width=width,
            color=colors[arm],
            label=arm,
        )
    stability_ax.set_xticks(x, [str(seed) for seed in PAIRED_SEEDS])
    stability_ax.set_xlabel("Paired training seed")
    stability_ax.set_ylabel("Std. of adjacent validation-gain deltas")
    stability_ax.set_title("Validation-gain volatility by paired seed")
    stability_ax.grid(
        color="#D9D9D9", linewidth=0.8, alpha=0.75, axis="y"
    )
    stability_ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compare_grpo_stability(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    """Validate six paired runs and emit the frozen diagnostic A/B result."""

    manifest_path = Path(manifest_path).resolve()
    source_sha, entries = _validate_manifest(manifest_path)
    runs: dict[tuple[str, int], _RunSeries] = {}
    fairness_binding: dict[str, object] | None = None
    for entry in entries:
        run, binding = _load_run(
            manifest_path,
            entry,
            source_sha=source_sha,
            expected_binding=fairness_binding,
        )
        if fairness_binding is None:
            fairness_binding = binding
        runs[(run.arm, run.seed)] = run

    pairs: list[dict[str, object]] = []
    volatility_reduced_count = 0
    baseline_finals: list[float] = []
    clipped_finals: list[float] = []
    for seed in PAIRED_SEEDS:
        baseline = runs[("baseline", seed)]
        clipped = runs[("clipped", seed)]
        if not np.array_equal(baseline.rollout_steps, clipped.rollout_steps):
            raise GRPOStabilityComparisonError(
                f"paired seed {seed} validation rollout steps are not aligned"
            )
        reduced = clipped.adjacent_delta_std < baseline.adjacent_delta_std
        volatility_reduced_count += int(reduced)
        baseline_finals.append(baseline.final_reward_gain)
        clipped_finals.append(clipped.final_reward_gain)
        pairs.append(
            {
                "seed": seed,
                "validation_rollout_groups": baseline.rollout_steps.tolist(),
                "baseline_reward_gain": baseline.reward_gain.tolist(),
                "clipped_reward_gain": clipped.reward_gain.tolist(),
                "baseline_adjacent_delta_std": baseline.adjacent_delta_std,
                "clipped_adjacent_delta_std": clipped.adjacent_delta_std,
                "volatility_reduced": reduced,
                "baseline_final_reward_gain": baseline.final_reward_gain,
                "clipped_final_reward_gain": clipped.final_reward_gain,
                "final_reward_gain_delta": (
                    clipped.final_reward_gain - baseline.final_reward_gain
                ),
                "baseline_optimizer_steps": baseline.optimizer_steps,
                "clipped_optimizer_steps": clipped.optimizer_steps,
                "baseline_wall_time_seconds": baseline.wall_time_seconds,
                "clipped_wall_time_seconds": clipped.wall_time_seconds,
            }
        )

    baseline_final_mean = float(np.mean(baseline_finals))
    clipped_final_mean = float(np.mean(clipped_finals))
    volatility_gate_passed = volatility_reduced_count >= 2
    final_mean_non_degraded = clipped_final_mean >= baseline_final_mean
    output_dir = Path(output_dir).resolve()
    plot_path = _render_comparison_plot(
        runs, output_dir / "validation_reward_gain_ab.png"
    )
    report = {
        "format": REPORT_FORMAT,
        "diagnostic_only": True,
        "eligible_for_formal_conclusions": False,
        "manifest": str(manifest_path),
        "source_stage1_sha256": source_sha,
        "paired_seeds": list(PAIRED_SEEDS),
        "group_size": GROUP_SIZE,
        "total_rollout_groups_per_run": TOTAL_ROLLOUT_GROUPS,
        "validation_interval_rollouts": VALIDATION_INTERVAL_ROLLOUTS,
        "baseline_update_epochs": ARM_UPDATE_EPOCHS["baseline"],
        "clipped_update_epochs": ARM_UPDATE_EPOCHS["clipped"],
        "clip_epsilon": CLIP_EPSILON,
        "pairs": pairs,
        "summary": {
            "volatility_reduced_pair_count": volatility_reduced_count,
            "required_volatility_reduced_pair_count": 2,
            "volatility_gate_passed": volatility_gate_passed,
            "baseline_final_reward_gain_mean": baseline_final_mean,
            "clipped_final_reward_gain_mean": clipped_final_mean,
            "final_reward_gain_mean_delta": (
                clipped_final_mean - baseline_final_mean
            ),
            "final_mean_non_degraded": final_mean_non_degraded,
            "passed": volatility_gate_passed and final_mean_non_degraded,
        },
        "plot": str(plot_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare the six-run Stage-2 clipped-GRPO A/B "
            "diagnostic. See the module docstring for the exact manifest schema."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "stage2_grpo_stability_ab_manifest_v1 JSON; relative run_dir "
            "values resolve from this file"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for report.json and validation_reward_gain_ab.png",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = compare_grpo_stability(args.manifest, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
