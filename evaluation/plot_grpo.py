"""Plot GRPO training diagnostics from TensorBoard summaries."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


ADVANTAGE_VECTOR_TAG = "advantage/vector"
REWARD_CURVE_TAGS = (
    "raw_proxy_reward_mean",
    "raw_proxy_reward_max",
    "validation/raw_proxy_reward_mean",
    "validation/pretrain_reward",
)
GRPO_LOSS_CURVE_TAGS = (
    "loss/total",
    "loss/mode_pg",
    "loss/trajectory_pg",
)
KL_LOSS_CURVE_TAGS = (
    "loss/reference_kl",
    "loss/mode_reference_kl",
    "loss/trajectory_reference_kl",
)

_SCALAR_STEP_ALIGNMENT_GROUPS = (
    REWARD_CURVE_TAGS[:2],
    REWARD_CURVE_TAGS[2:],
    GRPO_LOSS_CURVE_TAGS,
    KL_LOSS_CURVE_TAGS,
)

_CURVE_SPECS = {
    "reward_curve": (
        "reward_curve.png",
        REWARD_CURVE_TAGS,
        "GRPO raw reward curves",
        "Raw tau_d reward",
    ),
    "grpo_loss_curve": (
        "grpo_loss_curve.png",
        GRPO_LOSS_CURVE_TAGS,
        "GRPO loss curves",
        "Loss",
    ),
    "kl_loss_curve": (
        "kl_loss_curve.png",
        KL_LOSS_CURVE_TAGS,
        "GRPO reference KL curves",
        "Unweighted KL divergence",
    ),
}


class AdvantageHeatmapError(ValueError):
    """Raised when persisted GRPO plot inputs violate their contract."""


def _load_event_accumulator(
    tb_dir: Path,
) -> event_accumulator.EventAccumulator:
    accumulator = event_accumulator.EventAccumulator(
        str(Path(tb_dir)),
        size_guidance={
            event_accumulator.TENSORS: 0,
            event_accumulator.SCALARS: 0,
        },
    )
    accumulator.Reload()
    return accumulator


def _load_advantage_vectors_from_accumulator(
    accumulator: event_accumulator.EventAccumulator,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    if tag not in accumulator.Tags().get("tensors", ()):
        raise AdvantageHeatmapError(
            f"TensorBoard tensor tag {tag!r} was not found"
        )

    events = accumulator.Tensors(tag)
    if not events:
        raise AdvantageHeatmapError(
            f"TensorBoard tensor tag {tag!r} contains no events"
        )

    records: list[tuple[int, np.ndarray]] = []
    observed_steps: set[int] = set()
    group_size: int | None = None
    for event in events:
        step = int(event.step)
        if step in observed_steps:
            raise AdvantageHeatmapError(
                f"TensorBoard tensor tag {tag!r} has duplicate step {step}"
            )
        observed_steps.add(step)

        tensor = np.asarray(tensor_util.make_ndarray(event.tensor_proto))
        if tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] <= 0:
            raise AdvantageHeatmapError(
                f"step {step} advantage tensor must have shape [1,G], "
                f"got {tensor.shape}"
            )
        current_group_size = int(tensor.shape[1])
        if group_size is None:
            group_size = current_group_size
        elif current_group_size != group_size:
            raise AdvantageHeatmapError(
                "advantage tensors must use one consistent group size; "
                f"expected {group_size}, got {current_group_size} at step {step}"
            )
        try:
            finite = bool(np.isfinite(tensor).all())
        except TypeError as exc:
            raise AdvantageHeatmapError(
                f"step {step} advantage tensor must be numeric"
            ) from exc
        if not finite:
            raise AdvantageHeatmapError(
                f"step {step} advantage tensor must contain only finite values"
            )
        records.append((step, tensor[0]))

    records.sort(key=lambda record: record[0])
    steps = np.asarray([step for step, _ in records], dtype=np.int64)
    values = np.stack([row for _, row in records], axis=0)
    return steps, values


def load_advantage_vectors(
    tb_dir: Path,
    tag: str = ADVANTAGE_VECTOR_TAG,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all ``[1, G]`` advantage tensors ordered by optimizer step."""

    return _load_advantage_vectors_from_accumulator(
        _load_event_accumulator(tb_dir), tag
    )


def _render_advantage_heatmap(
    steps: np.ndarray,
    values: np.ndarray,
    output_path: Path,
) -> Path:
    heatmap = values.T
    absolute_limit = float(np.max(np.abs(values)))
    if absolute_limit == 0.0:
        absolute_limit = 1.0

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    image = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-absolute_limit,
        vmax=absolute_limit,
    )

    tick_count = min(len(steps), 12)
    tick_indices = np.unique(
        np.linspace(0, len(steps) - 1, num=tick_count, dtype=np.int64)
    )
    ax.set_xticks(
        tick_indices,
        [str(int(steps[index])) for index in tick_indices],
    )
    ax.set_yticks(
        np.arange(values.shape[1]),
        [f"g{index}" for index in range(values.shape[1])],
    )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("GRPO sample slot")
    ax.set_title("GRPO advantage by optimizer step and sample slot")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Normalized advantage")
    fig.text(
        0.5,
        0.015,
        "Group slots are independent samples and have no identity across steps.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_advantage_heatmap(tb_dir: Path, output_path: Path) -> Path:
    """Generate a raw group-slot-by-optimizer-step advantage heatmap."""

    steps, values = load_advantage_vectors(tb_dir)
    return _render_advantage_heatmap(steps, values, output_path)


def _load_scalar_series(
    accumulator: event_accumulator.EventAccumulator,
    tags: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    available_tags = set(accumulator.Tags().get("scalars", ()))
    missing_tags = [tag for tag in tags if tag not in available_tags]
    if missing_tags:
        raise AdvantageHeatmapError(
            "TensorBoard is missing required scalar tags: "
            + ", ".join(missing_tags)
        )

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        if not events:
            raise AdvantageHeatmapError(
                f"TensorBoard scalar tag {tag!r} contains no events"
            )
        observed_steps: set[int] = set()
        records: list[tuple[int, float]] = []
        for event in events:
            step = int(event.step)
            if step in observed_steps:
                raise AdvantageHeatmapError(
                    f"TensorBoard scalar tag {tag!r} has duplicate step {step}"
                )
            observed_steps.add(step)
            value = float(event.value)
            if not np.isfinite(value):
                raise AdvantageHeatmapError(
                    f"scalar tag {tag!r} step {step} must be finite"
                )
            records.append((step, value))
        records.sort(key=lambda record: record[0])
        series[tag] = (
            np.asarray([step for step, _ in records], dtype=np.int64),
            np.asarray([value for _, value in records], dtype=np.float64),
        )
    return series


def _render_scalar_curve(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    tags: tuple[str, ...],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for tag in tags:
        steps, values = series[tag]
        marker = "o" if len(steps) <= 100 else None
        ax.plot(
            steps,
            values,
            marker=marker,
            markersize=3.5,
            linewidth=1.5,
            label=tag,
        )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _validate_aligned_scalar_steps(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    for tags in _SCALAR_STEP_ALIGNMENT_GROUPS:
        expected_steps = series[tags[0]][0]
        for tag in tags[1:]:
            if not np.array_equal(series[tag][0], expected_steps):
                raise AdvantageHeatmapError(
                    "TensorBoard scalar tags must use aligned optimizer steps: "
                    + ", ".join(tags)
                )


def generate_grpo_plots(
    tb_dir: Path,
    output_dir: Path,
) -> Mapping[str, Path]:
    """Generate the fixed GRPO advantage, reward, loss, and KL plots."""

    accumulator = _load_event_accumulator(tb_dir)
    steps, advantages = _load_advantage_vectors_from_accumulator(
        accumulator, ADVANTAGE_VECTOR_TAG
    )
    all_scalar_tags = tuple(
        tag
        for _, tags, _, _ in _CURVE_SPECS.values()
        for tag in tags
    )
    scalar_series = _load_scalar_series(accumulator, all_scalar_tags)
    _validate_aligned_scalar_steps(scalar_series)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {
        "advantage_heatmap": _render_advantage_heatmap(
            steps,
            advantages,
            output_dir / "advantage_vector_heatmap.png",
        )
    }
    for key, (filename, tags, title, ylabel) in _CURVE_SPECS.items():
        outputs[key] = _render_scalar_curve(
            scalar_series,
            tags,
            title=title,
            ylabel=ylabel,
            output_path=output_dir / filename,
        )
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot GRPO training diagnostics from TensorBoard events."
    )
    parser.add_argument("--tensorboard-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    for name, output_path in generate_grpo_plots(
        args.tensorboard_dir, args.output_dir
    ).items():
        print(f"{name}={output_path}")


if __name__ == "__main__":
    main()
