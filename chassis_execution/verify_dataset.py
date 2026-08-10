"""CLI for the CF-3 full mmap dataset verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import ChassisExecutionStorageError, verify_chassis_execution_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = verify_chassis_execution_dataset(args.root)
    except ChassisExecutionStorageError as exc:
        parser.error(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
