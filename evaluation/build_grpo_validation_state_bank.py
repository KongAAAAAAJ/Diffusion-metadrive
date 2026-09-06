"""Build the fixed S5--S9 validation state bank for bounded GRPO diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.train_bev_joint_grpo_online import (
    _checkpoint_file_sha256,
    build_grpo_validation_state_bank,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_grpo_validation_state_bank(args.output)
    print(
        json.dumps(
            {
                "format": payload["format"],
                "records": len(payload["records"]),
                "output": str(args.output.resolve()),
                "sha256": _checkpoint_file_sha256(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
