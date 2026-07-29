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
    run_joint_reward_calibration,
)
from models.bev_planner import JointRewardConfig


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
        "--phase", choices=("development", "holdout"), required=True
    )
    parser.add_argument("--tracking-envelope-report", type=Path)
    arguments = parser.parse_args()
    reward_config = JointRewardConfig()
    if arguments.tracking_envelope_report is not None:
        try:
            development = json.loads(
                arguments.tracking_envelope_report.read_text(encoding="utf-8")
            )
            envelope = development["tracking"]["recommended_envelope"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            parser.error(f"invalid tracking envelope report: {exc}")
        if (
            development.get("format") != "bev_joint_reward_calibration_v2"
            or development.get("calibration_phase") != "development"
        ):
            parser.error("tracking envelope must come from development calibration")
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
