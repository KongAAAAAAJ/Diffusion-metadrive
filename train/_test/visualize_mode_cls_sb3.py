from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def visualize(run_dir: Path, output_dir: Path | None = None) -> Path:
    log_path = run_dir / "mode_selection_debug.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing debug log: {log_path}")
    records = _load_jsonl(log_path)
    if not records:
        raise ValueError(f"Debug log is empty: {log_path}")
    output_dir = output_dir or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    executed = np.asarray([record["executed_mode"] for record in records], dtype=np.int64)
    pretrained = np.asarray([record["pretrained_argmax_mode"] for record in records], dtype=np.int64)
    rewards = np.asarray([record.get("reward", 0.0) for record in records], dtype=np.float32)
    invalid = np.asarray([record.get("invalid_mode_rate", 0.0) for record in records], dtype=np.float32)

    max_mode = int(max(executed.max(initial=0), pretrained.max(initial=0)))
    bins = np.arange(max_mode + 2) - 0.5

    plt.figure(figsize=(8, 4))
    plt.hist(executed.reshape(-1), bins=bins, alpha=0.7, label="executed")
    plt.hist(pretrained.reshape(-1), bins=bins, alpha=0.5, label="pretrained_argmax")
    plt.xlabel("mode index")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mode_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="reward")
    plt.xlabel("env step")
    plt.ylabel("scalar reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(invalid, label="invalid_mode_rate")
    plt.xlabel("env step")
    plt.ylabel("rate")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "invalid_mode_rate.png", dpi=160)
    plt.close()

    endpoints = np.asarray(records[0]["candidate_endpoints"], dtype=np.float32)
    plt.figure(figsize=(6, 6))
    for agent_idx in range(endpoints.shape[0]):
        plt.scatter(endpoints[agent_idx, :, 0], endpoints[agent_idx, :, 1], label=f"agent{agent_idx}", s=24)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "candidate_endpoints_first_step.png", dpi=160)
    plt.close()

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SB3 mode-selection PPO debug logs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    output = visualize(Path(args.run_dir), Path(args.output_dir) if args.output_dir else None)
    print(f"[mode_cls_ppo] figures: {output}")


if __name__ == "__main__":
    main()
