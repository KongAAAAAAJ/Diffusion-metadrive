#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import dataclasses
import json
from pathlib import Path

from train.train_bev_joint_grpo_online import (
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    PRIMARY_S5_S9_SCENARIOS,
    OnlineGRPOError,
    _validate_calibration_reward_contract,
    _validate_calibration_trajectory_optimizer_contract,
    run_joint_reward_calibration,
)
from models.bev_planner import JointRewardConfig, sha256_file
from scenarios.bev_round13_contract import primary_scenario_contract


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate joint BEV proxy rewards against simulator branches"
    )
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--states-per-episode", type=int, default=3)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write the complete JSON report without echoing it to stdout",
    )
    parser.add_argument(
        "--phase", choices=("development", "holdout"), required=True
    )
    parser.add_argument("--tracking-envelope-report", type=Path)
    arguments = parser.parse_args()
    if arguments.phase == "holdout" and arguments.tracking_envelope_report is None:
        parser.error("holdout calibration requires --tracking-envelope-report")
    if arguments.phase == "development" and arguments.tracking_envelope_report is not None:
        parser.error("development calibration cannot consume a tracking envelope")
    reward_config = JointRewardConfig()
    development_report_sha256 = None
    if arguments.tracking_envelope_report is not None:
        try:
            development = json.loads(
                arguments.tracking_envelope_report.read_text(encoding="utf-8")
            )
            envelope = development["tracking"]["recommended_envelope"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            parser.error(f"invalid tracking envelope report: {exc}")
        if (
            development.get("format") != "bev_joint_reward_calibration_v3"
            or development.get("calibration_phase") != "development"
        ):
            parser.error("tracking envelope must come from development calibration")
        source_sha256 = sha256_file(arguments.stage1_checkpoint)
        expected_scenario_contract = primary_scenario_contract()
        if (
            development.get("variant") != arguments.variant
            or development.get("source_stage1_sha256") != source_sha256
            or development.get("source_stage1_checkpoint")
            != str(arguments.stage1_checkpoint.resolve())
            or development.get("diagnostic_subset_smoke") is not False
            or tuple(
                tuple(value) for value in development.get("scenarios", ())
            )
            != PRIMARY_S5_S9_SCENARIOS
            or tuple(development.get("seeds", ())) != DEVELOPMENT_SEEDS
            or development.get("states_per_episode")
            != arguments.states_per_episode
            or development.get("scenario_contract")
            != expected_scenario_contract
            or development.get("scenario_contract_sha256")
            != expected_scenario_contract["sha256"]
        ):
            parser.error(
                "development tracking envelope provenance does not match holdout"
            )
        try:
            development_reward_config = _validate_calibration_reward_contract(
                development
            )
        except OnlineGRPOError as exc:
            parser.error(str(exc))
        try:
            _validate_calibration_trajectory_optimizer_contract(development)
        except OnlineGRPOError as exc:
            parser.error(str(exc))
        if development_reward_config != JointRewardConfig():
            parser.error(
                "development tracking envelope reward config is not the V2 base config"
            )
        if not bool(envelope.get("within_controller_limits", False)):
            parser.error(
                "development tracking exceeds controller limits; fix control before holdout"
            )
        reward_config = dataclasses.replace(
            reward_config,
            tracking_longitudinal_margin_m=float(
                envelope["longitudinal_margin_m"]
            ),
            tracking_lateral_margin_m=float(envelope["lateral_margin_m"]),
            tracking_heading_margin_rad=float(envelope["heading_margin_rad"]),
        )
        development_report_sha256 = sha256_file(
            arguments.tracking_envelope_report
        )
    report = run_joint_reward_calibration(
        variant=arguments.variant,
        source_checkpoint=arguments.stage1_checkpoint,
        output_path=arguments.output,
        device=arguments.device,
        reward_config=reward_config,
        scenarios=PRIMARY_S5_S9_SCENARIOS,
        seeds=(
            DEVELOPMENT_SEEDS
            if arguments.phase == "development"
            else HOLDOUT_SEEDS
        ),
        states_per_episode=arguments.states_per_episode,
        calibration_phase=arguments.phase,
        development_calibration_sha256=development_report_sha256,
    )
    if not arguments.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
