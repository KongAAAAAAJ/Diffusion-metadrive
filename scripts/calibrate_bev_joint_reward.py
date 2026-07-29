#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

from train.train_bev_joint_grpo_online import (
    DIAGNOSTIC_SCENARIOS,
    DIAGNOSTIC_SEEDS,
    run_joint_reward_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate joint BEV proxy rewards against simulator branches"
    )
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--states-per-episode", type=int, default=3)
    parser.add_argument("--scenario-limit", type=int, default=4)
    parser.add_argument("--seed-limit", type=int, default=2)
    arguments = parser.parse_args()
    if not 1 <= arguments.scenario_limit <= len(DIAGNOSTIC_SCENARIOS):
        parser.error("--scenario-limit is outside the diagnostic scenario set")
    if not 1 <= arguments.seed_limit <= len(DIAGNOSTIC_SEEDS):
        parser.error("--seed-limit is outside the diagnostic seed set")
    report = run_joint_reward_calibration(
        variant=arguments.variant,
        source_checkpoint=arguments.stage1_checkpoint,
        output_path=arguments.output,
        device=arguments.device,
        scenarios=DIAGNOSTIC_SCENARIOS[: arguments.scenario_limit],
        seeds=DIAGNOSTIC_SEEDS[: arguments.seed_limit],
        states_per_episode=arguments.states_per_episode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
