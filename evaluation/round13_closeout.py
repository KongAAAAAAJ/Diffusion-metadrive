"""Machine-verifiable Round 13 infrastructure closeout contract."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Mapping

from expert_dataset.joint_bev_storage import (
    STORAGE_FORMAT,
    STORAGE_SCHEMA_VERSION,
)
from expert_dataset.semantic_bev_codec import packed_bev_contract
from models.bev_planner.joint_reward import JointRewardConfig
from models.bev_planner.mode_contract import (
    HardModeMaskConfig,
    MODE_NAMES,
    NUM_MODES,
    TRAJECTORY_DIM,
    TRAJECTORY_STEPS,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizerConfig,
)
from scenarios.bev_round13_contract import primary_scenario_contract
from train.bev_joint_grpo import GRPO_CHECKPOINT_SCHEMA_VERSION
from train.train_bev_diffusion_stage1 import CHECKPOINT_SCHEMA_VERSION


ROUND13_REPORTS = (
    "evaluation/ROUND13_89_REPORT.md",
    "evaluation/ROUND13_91_REPORT.md",
    "evaluation/ROUND13_92_REPORT.md",
    "evaluation/ROUND13_94_REPORT.md",
    "evaluation/ROUND13_95_REPORT.md",
)


class Round13CloseoutError(RuntimeError):
    """Raised when the frozen Round 13 contract no longer matches the repo."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise Round13CloseoutError(f"required Round 13 evidence is missing: {path}") from exc
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_contract(value: object) -> dict[str, object]:
    payload = json.loads(
        json.dumps(
            dataclasses.asdict(value),
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return {"config": payload, "sha256": _canonical_sha256(payload)}


def build_round13_closeout(repo_root: Path | str) -> dict[str, object]:
    root = Path(repo_root).resolve()
    reports = {
        relative: _file_sha256(root / relative) for relative in ROUND13_REPORTS
    }
    bev_codec = packed_bev_contract()
    scenario = primary_scenario_contract()
    return {
        "format": "bev_round13_closeout_v1",
        "round": "13.96",
        "status": "infrastructure_accepted",
        "scope": {
            "accepted": [
                "S5-S9 expert/data contract",
                "BEV-only Stage 1 A/B training chain",
                "proxy-simulator reward calibration chain",
                "A/B joint GRPO training chain",
                "four-model evaluator and S9 execution feasibility",
                "semantic BEV P95 below 100ms",
            ],
            "waived": [
                "bit-exact repeated closed-loop physics for the same seed",
            ],
            "not_claimed": [
                "diagnostic checkpoint policy superiority",
                "formal paper conclusions",
                "full planning-tick P95 below 100ms",
            ],
        },
        "reproducibility_policy": {
            "bit_exact_closed_loop_required": False,
            "initial_scene_exact_required": True,
            "identical_input_noise_model_output_exact_required": True,
            "formal_results_require_multi_seed_statistics": True,
        },
        "formal_eligibility": {
            "current_checkpoints_eligible": False,
            "metadata_override_forbidden": True,
            "required_order": [
                "formal joint-BEV dataset pilot and full verifier",
                "formal Stage 1 A/B training",
                "fresh A/B reward calibration",
                "formal A/B GRPO training",
                "semantic-preserving Round 14 cleanup",
                "formal multi-seed S5-S9 evaluation",
            ],
        },
        "scenario_contract": scenario,
        "data_contract": {
            "storage_schema_version": STORAGE_SCHEMA_VERSION,
            "storage_format": STORAGE_FORMAT,
            "joint_first": True,
            "bev_codec": bev_codec,
            "bev_codec_sha256": _canonical_sha256(bev_codec),
        },
        "mode_contract": {
            "mode_names": list(MODE_NAMES),
            "num_modes": NUM_MODES,
            "trajectory_steps": TRAJECTORY_STEPS,
            "trajectory_dim": TRAJECTORY_DIM,
            "hard_mask": _config_contract(HardModeMaskConfig()),
        },
        "execution_contract": _config_contract(
            KinematicTrajectoryOptimizerConfig()
        ),
        "reward_contract": _config_contract(JointRewardConfig()),
        "checkpoint_contract": {
            "stage1_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "grpo_schema_version": GRPO_CHECKPOINT_SCHEMA_VERSION,
        },
        "latency_evidence_ms": {
            "three_role_model_inference_p95_max": 20.84,
            "semantic_bev_build_p95_max": 89.97,
            "full_planning_tick_p95_max": 125.14,
            "model_inference_limit": 100.0,
            "full_tick_is_nonblocking_for_formal_training": True,
        },
        "evidence_sha256": reports,
        "next_hard_gate": {
            "name": "formal_joint_bev_15000_step_pilot",
            "requires_full_verifier": True,
            "allows_old_diagnostic_data": False,
        },
    }


def verify_round13_closeout(
    payload: object, repo_root: Path | str
) -> dict[str, object]:
    expected = build_round13_closeout(repo_root)
    if not isinstance(payload, Mapping) or dict(payload) != expected:
        raise Round13CloseoutError(
            "Round 13 closeout no longer matches frozen code/evidence contracts"
        )
    return expected


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Round13CloseoutError(f"invalid closeout manifest: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("ROUND13_96_FREEZE.json"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        payload = build_round13_closeout(args.repo_root)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        payload = verify_round13_closeout(
            _read_json(args.manifest), args.repo_root
        )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "scenario_sha256": payload["scenario_contract"]["sha256"],
                "optimizer_sha256": payload["execution_contract"]["sha256"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
