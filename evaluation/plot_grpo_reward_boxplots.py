"""Plot Stage 1 versus GRPO per-step reward distributions."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


MODEL_COLUMN = "model"
MODEL_ORDER = ("stage1_a", "grpo_open")
MODEL_LABELS = {
    "stage1_a": "Stage 1 (before GRPO)",
    "grpo_open": "GRPO (after fine-tuning)",
}
MODEL_COLORS = {
    "stage1_a": "#4C78A8",
    "grpo_open": "#F58518",
}
REWARD_COLUMNS = (
    "total_reward",
    "progress_reward",
    "formation_reward",
    "gap_reward",
    "ttc_reward",
    "road_reward",
    "comfort_reward",
    "collision_reward",
    "out_of_drivable_reward",
)
GROUPED_REWARD_COLUMNS = (
    "total_reward",
    "progress_reward",
    "ttc_reward",
    "comfort_reward",
)
SINGLE_REWARD_COLUMNS = (
    "formation_reward",
    "gap_reward",
    "road_reward",
    "collision_reward",
    "out_of_drivable_reward",
)
GROUPED_REWARD_BOXPLOT_FILENAME = (
    "total_progress_ttc_comfort_reward_boxplots.png"
)
REWARD_LABELS = {
    "total_reward": "Total reward",
    "progress_reward": "Progress reward",
    "formation_reward": "Formation reward",
    "gap_reward": "Gap reward",
    "ttc_reward": "TTC reward",
    "road_reward": "Road reward",
    "comfort_reward": "Comfort reward",
    "collision_reward": "Collision reward",
    "out_of_drivable_reward": "Out-of-drivable reward",
}


class RewardBoxplotError(ValueError):
    """Raised when the persisted per-step reward CSV violates its contract."""


def load_step_rewards(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load and strictly validate the two-model per-step reward CSV."""

    csv_path = Path(csv_path)
    values: dict[str, dict[str, list[float]]] = {
        model: {reward: [] for reward in REWARD_COLUMNS} for model in MODEL_ORDER
    }

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [
            column
            for column in (MODEL_COLUMN, *REWARD_COLUMNS)
            if column not in fieldnames
        ]
        if missing:
            raise RewardBoxplotError(
                f"reward CSV is missing required columns: {', '.join(missing)}"
            )

        observed_models: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            model = row[MODEL_COLUMN]
            if model not in MODEL_ORDER:
                raise RewardBoxplotError(
                    f"row {row_number} has unsupported model {model!r}; "
                    f"expected only {MODEL_ORDER}"
                )
            observed_models.add(model)

            for reward in REWARD_COLUMNS:
                raw_value = row[reward]
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise RewardBoxplotError(
                        f"row {row_number} column {reward!r} is not numeric: "
                        f"{raw_value!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise RewardBoxplotError(
                        f"row {row_number} column {reward!r} must be finite, "
                        f"got {raw_value!r}"
                    )
                values[model][reward].append(value)

    expected_models = set(MODEL_ORDER)
    if observed_models != expected_models:
        missing_models = sorted(expected_models - observed_models)
        raise RewardBoxplotError(
            "reward CSV must contain both stage1_a and grpo_open; "
            f"missing models: {', '.join(missing_models)}"
        )

    return {
        model: {
            reward: np.asarray(model_values[reward], dtype=np.float64)
            for reward in REWARD_COLUMNS
        }
        for model, model_values in values.items()
    }


def _mean_legend_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=6,
        label="Mean",
    )


def _draw_reward(
    ax: Axes,
    reward: str,
    values: dict[str, dict[str, np.ndarray]],
) -> None:
    distributions = [values[model][reward] for model in MODEL_ORDER]
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    boxplot = ax.boxplot(
        distributions,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
            "zorder": 5,
        },
        medianprops={"color": "black", "linewidth": 1.5, "zorder": 4},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, model in zip(boxplot["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.72)

    jitter_rng = np.random.default_rng(0)
    for position, model, distribution in zip(
        positions, MODEL_ORDER, distributions
    ):
        jittered_positions = position + jitter_rng.uniform(
            -0.14, 0.14, size=len(distribution)
        )
        ax.scatter(
            jittered_positions,
            distribution,
            marker="o",
            s=16,
            alpha=0.30,
            color=MODEL_COLORS[model],
            edgecolors="none",
            zorder=3,
        )

    means = [float(np.mean(distribution)) for distribution in distributions]
    for position, mean in zip(positions, means):
        ax.annotate(
            f"mean = {mean:.4f}",
            xy=(position, mean),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
        )

    ax.set_xticks(positions, [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_ylabel("Weighted signed reward")
    ax.set_title(REWARD_LABELS[reward])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_reward(
    reward: str,
    values: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _draw_reward(ax, reward, values)
    ax.set_title(f"Per-step distribution: {REWARD_LABELS[reward]}")
    ax.legend(
        handles=[_mean_legend_handle()],
        loc="best",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_rewards(
    values: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.8))
    for ax, reward in zip(axes.flat, GROUPED_REWARD_COLUMNS):
        _draw_reward(ax, reward, values)

    fig.suptitle("Per-step reward distributions", fontsize=16)
    fig.legend(
        handles=[_mean_legend_handle()],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_reward_boxplots(csv_path: Path, output_dir: Path) -> tuple[Path, ...]:
    """Generate one grouped and five individual two-checkpoint reward plots."""

    values = load_step_rewards(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_output = output_dir / GROUPED_REWARD_BOXPLOT_FILENAME
    _plot_grouped_rewards(values, grouped_output)
    outputs = [grouped_output]
    for reward in SINGLE_REWARD_COLUMNS:
        output_path = output_dir / f"{reward}_boxplot.png"
        _plot_reward(reward, values, output_path)
        outputs.append(output_path)

    for reward in GROUPED_REWARD_COLUMNS:
        (output_dir / f"{reward}_boxplot.png").unlink(missing_ok=True)
    return tuple(outputs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Stage 1 versus GRPO boxplots from the unified per-step "
            "reward CSV."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    for output_path in generate_reward_boxplots(args.input_csv, args.output_dir):
        print(output_path)


if __name__ == "__main__":
    main()
