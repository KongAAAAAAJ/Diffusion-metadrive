"""Diagnostic helpers for controller debugging."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path


DEFAULT_CONTROLLER_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

_PID_DEBUG_HISTORY: dict[str, dict[str, list[float]]] = {}


def _safe_agent_id(agent_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(agent_id)).strip("_") or "agent"


def reset_pid_debug_history(agent_id: str | None = None) -> None:
    """Clear stored PID debug samples."""
    if agent_id is None:
        _PID_DEBUG_HISTORY.clear()
    else:
        _PID_DEBUG_HISTORY.pop(str(agent_id), None)


def get_pid_debug_history(agent_id: str | None = None) -> dict:
    """Return a copy of stored PID debug samples."""
    if agent_id is None:
        return copy.deepcopy(_PID_DEBUG_HISTORY)
    return copy.deepcopy(_PID_DEBUG_HISTORY.get(str(agent_id), {}))


def save_pid_debug_plot(
    *,
    agent_id: str,
    steering: float,
    actual_heading: float,
    heading_error: float,
    output_dir: str | Path | None = None,
    max_history: int = 200,
) -> Path:
    """Append one PID debug sample and save a 3x1 non-interactive plot."""
    agent_key = str(agent_id)
    reference_heading = float(actual_heading) + float(heading_error)
    history = _PID_DEBUG_HISTORY.setdefault(
        agent_key,
        {
            "steering": [],
            "actual_heading": [],
            "reference_heading": [],
            "heading_error": [],
        },
    )
    history["steering"].append(float(steering))
    history["actual_heading"].append(float(actual_heading))
    history["reference_heading"].append(reference_heading)
    history["heading_error"].append(float(heading_error))

    keep = max(int(max_history), 1)
    for values in history.values():
        if len(values) > keep:
            del values[:-keep]

    save_dir = Path(output_dir) if output_dir is not None else DEFAULT_CONTROLLER_OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"pid_debug_{_safe_agent_id(agent_key)}_latest.png"

    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = list(range(len(history["steering"])))
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(x, history["steering"], color="tab:blue", linewidth=1.8)
    axes[0].set_ylabel("steering")
    axes[0].set_title(f"PID Debug - {agent_key}")

    axes[1].plot(x, history["actual_heading"], label="actual", color="tab:green", linewidth=1.8)
    axes[1].plot(x, history["reference_heading"], label="reference", color="tab:orange", linewidth=1.8)
    axes[1].set_ylabel("heading")
    axes[1].legend(loc="best")

    axes[2].plot(x, history["heading_error"], color="tab:red", linewidth=1.8)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("heading_err")
    axes[2].set_xlabel("control step")

    for axis in axes:
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path
