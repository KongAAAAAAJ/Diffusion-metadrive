from __future__ import annotations

import re
from pathlib import Path


RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def create_numbered_run_dir(output_root: Path) -> Path:
    existing_run_dirs = [
        path for path in output_root.iterdir() if path.is_dir() and RUN_DIR_PATTERN.match(path.name)
    ] if output_root.exists() else []
    next_index = len(existing_run_dirs) + 1
    run_dir = output_root / f"run_{next_index}"
    while run_dir.exists():
        next_index += 1
        run_dir = output_root / f"run_{next_index}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
