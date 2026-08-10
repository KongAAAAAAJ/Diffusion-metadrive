"""Command-line verifier for CF-2 Windows TruckSim exchange roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trucksim_exchange import TruckSimExportError, verify_trucksim_export_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = verify_trucksim_export_root(args.root)
    except TruckSimExportError as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
