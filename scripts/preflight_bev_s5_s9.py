#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

from train.train_bev_joint_grpo_online import run_s5_s9_preflight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight the frozen S5--S9 online BEV scenario contract"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_s5_s9_preflight(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
