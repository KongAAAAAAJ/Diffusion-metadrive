"""Plot group-relative GRPO advantage vectors from TensorBoard summaries."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


ADVANTAGE_VECTOR_TAG = "advantage/vector"


class AdvantageHeatmapError(ValueError):
    """Raised when persisted GRPO advantage vectors violate their contract."""


def load_advantage_vectors(
    tb_dir: Path,
    tag: str = ADVANTAGE_VECTOR_TAG,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all ``[1, G]`` advantage tensors ordered by optimizer step."""

    accumulator = event_accumulator.EventAccumulator(
        str(Path(tb_dir)),
        size_guidance={event_accumulator.TENSORS: 0},
    )
    accumulator.Reload()
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


def generate_advantage_heatmap(tb_dir: Path, output_path: Path) -> Path:
    """Generate a raw group-slot-by-optimizer-step advantage heatmap."""

    steps, values = load_advantage_vectors(tb_dir)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a GRPO advantage heatmap from TensorBoard tensors."
    )
    parser.add_argument("--tensorboard-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    print(generate_advantage_heatmap(args.tensorboard_dir, args.output))


if __name__ == "__main__":
    main()
