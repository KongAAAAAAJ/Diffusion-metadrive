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
FRESH_ROLLOUT_AXIS_LABEL = "Fresh sampling attempt"
ACCEPTED_ROLLOUT_AXIS_LABEL = "Accepted update state"
OPTIMIZER_STEP_AXIS_LABEL = "Optimizer step"
ROLLOUT_AXIS_LABELS = (
    FRESH_ROLLOUT_AXIS_LABEL,
    ACCEPTED_ROLLOUT_AXIS_LABEL,
)
LEGACY_REWARD_CURVE_TAGS = (
    "raw_proxy_reward_mean",
    "raw_proxy_reward_max",
)
SAME_MODE_PRETRAIN_REWARD_TAG = "same_mode_pretrain_reward_mean"
# Kept as an import alias for callers written before the same-mode rename.
FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG = SAME_MODE_PRETRAIN_REWARD_TAG
REWARD_CURVE_TAGS = (
    "vehicle_reward_mean",
    SAME_MODE_PRETRAIN_REWARD_TAG,
    "same_mode_reward_gain_mean",
)
VALIDATION_REWARD_CURVE_TAGS = (
    "validation/vehicle_reward_mean",
    "validation/same_mode_pretrain_reward_mean",
    "validation/vehicle_reward_gain",
)
GRPO_LOSS_CURVE_TAGS = (
    "loss/total",
    "loss/trajectory_pg",
    "loss/behavior_cloning",
)
KL_LOSS_CURVE_TAGS = (
    "loss/reference_kl",
    "loss/trajectory_reference_kl",
)
POST_UPDATE_KL_TAGS = (
    "policy/post_update_reference_kl",
)
ADAPTER_DRIFT_TAGS = (
    "policy/adapter_drift_max",
)
ACTIVE_SIGNAL_TAGS = (
    "signal_mode/count",
    "no_signal_mode/count",
)
GUARD_SIGNAL_TAGS = (
    "stability_guard/rejected",
    "zero_signal_epoch",
)
POLICY_STABILITY_CURVE_TAGS = (
    *POST_UPDATE_KL_TAGS,
    *ADAPTER_DRIFT_TAGS,
    *ACTIVE_SIGNAL_TAGS,
    *GUARD_SIGNAL_TAGS,
)

_SCALAR_STEP_ALIGNMENT_GROUPS = (
    VALIDATION_REWARD_CURVE_TAGS,
    GRPO_LOSS_CURVE_TAGS,
    KL_LOSS_CURVE_TAGS,
    POLICY_STABILITY_CURVE_TAGS,
)

_CURVE_SPECS = {
    "reward_curve": (
        "reward_curve.png",
        REWARD_CURVE_TAGS,
        "Per-vehicle same-mode reward curves",
        "Counterfactual tau_d reward / gain",
        FRESH_ROLLOUT_AXIS_LABEL,
    ),
    "validation_reward_curve": (
        "validation_reward_curve.png",
        VALIDATION_REWARD_CURVE_TAGS,
        "Per-vehicle validation reward and gain curves",
        "Counterfactual tau_d reward / gain",
        FRESH_ROLLOUT_AXIS_LABEL,
    ),
    "grpo_loss_curve": (
        "grpo_loss_curve.png",
        GRPO_LOSS_CURVE_TAGS,
        "GRPO loss curves",
        "Loss",
        OPTIMIZER_STEP_AXIS_LABEL,
    ),
    "kl_loss_curve": (
        "kl_loss_curve.png",
        KL_LOSS_CURVE_TAGS,
        "GRPO reference KL curves",
        "Unweighted KL divergence",
        OPTIMIZER_STEP_AXIS_LABEL,
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
    vector_shape: tuple[int, ...] | None = None
    for event in events:
        step = int(event.step)
        if step in observed_steps:
            raise AdvantageHeatmapError(
                f"TensorBoard tensor tag {tag!r} has duplicate step {step}"
            )
        observed_steps.add(step)

        tensor = np.asarray(tensor_util.make_ndarray(event.tensor_proto))
        valid_legacy = (
            tensor.ndim == 2 and tensor.shape[0] == 1 and tensor.shape[1] > 0
        )
        valid_same_mode = (
            tensor.ndim == 4
            and tensor.shape[0] == 1
            and tensor.shape[1] == 3
            and tensor.shape[2] == 10
            and tensor.shape[3] > 0
        )
        if not (valid_legacy or valid_same_mode):
            raise AdvantageHeatmapError(
                f"step {step} advantage tensor must have shape [1,G] or "
                "[1,3,10,N], "
                f"got {tensor.shape}"
            )
        current_shape = tuple(int(value) for value in tensor.shape[1:])
        if vector_shape is None:
            vector_shape = current_shape
        elif current_shape != vector_shape:
            raise AdvantageHeatmapError(
                "advantage tensors must use one consistent rollout shape; "
                f"expected {vector_shape}, got {current_shape} at step {step}"
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
        records.append((step, tensor.reshape(-1)))

    records.sort(key=lambda record: record[0])
    steps = np.asarray([step for step, _ in records], dtype=np.int64)
    values = np.stack([row for _, row in records], axis=0)
    return steps, values


def load_advantage_vectors(
    tb_dir: Path,
    tag: str = ADVANTAGE_VECTOR_TAG,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and flatten legacy ``[1,G]`` or same-mode ``[1,3,10,N]`` tensors."""

    return _load_advantage_vectors_from_accumulator(
        _load_event_accumulator(tb_dir), tag
    )


def _render_advantage_heatmap(
    steps: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    *,
    rollout_axis_label: str,
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
    y_tick_count = min(values.shape[1], 12)
    y_tick_indices = np.unique(
        np.linspace(0, values.shape[1] - 1, num=y_tick_count, dtype=np.int64)
    )
    ax.set_yticks(y_tick_indices, [f"s{index}" for index in y_tick_indices])
    ax.set_xlabel(rollout_axis_label)
    ax.set_ylabel("Flattened vehicle-mode trajectory slot")
    ax.set_title("Same-mode GRPO advantage by update state and trajectory slot")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Fixed-scale truncated advantage")
    fig.text(
        0.5,
        0.015,
        "Slot order is vehicle-major, then mode, then sampled trajectory.",
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


def _validate_rollout_axis_label(value: str) -> str:
    if value not in ROLLOUT_AXIS_LABELS:
        raise AdvantageHeatmapError(
            "rollout_axis_label must be Fresh sampling attempt or "
            "Accepted update state"
        )
    return value


def generate_advantage_heatmap(
    tb_dir: Path,
    output_path: Path,
    *,
    rollout_axis_label: str = FRESH_ROLLOUT_AXIS_LABEL,
) -> Path:
    """Generate a vehicle-mode-trajectory-by-update advantage heatmap."""

    steps, values = load_advantage_vectors(tb_dir)
    return _render_advantage_heatmap(
        steps,
        values,
        output_path,
        rollout_axis_label=_validate_rollout_axis_label(rollout_axis_label),
    )


def _load_scalar_series(
    accumulator: event_accumulator.EventAccumulator,
    tags: tuple[str, ...],
    *,
    duplicate_step_tags: frozenset[str] = frozenset(),
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
            if step in observed_steps and tag not in duplicate_step_tags:
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


def _validate_terminal_guard_duplicate(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    tags: tuple[str, ...],
) -> None:
    expected_steps = series[tags[0]][0]
    for tag in tags[1:]:
        if not np.array_equal(series[tag][0], expected_steps):
            raise AdvantageHeatmapError(
                "terminal stability-guard scalar tags must use aligned steps: "
                + ", ".join(tags)
            )

    unique_steps, counts = np.unique(expected_steps, return_counts=True)
    duplicate_steps = unique_steps[counts > 1]
    valid_duplicate = (
        len(duplicate_steps) == 1
        and int(duplicate_steps[0]) == int(expected_steps[-1])
        and int(counts[unique_steps == duplicate_steps[0]][0]) == 2
        and np.array_equal(
            np.flatnonzero(expected_steps == duplicate_steps[0]),
            np.asarray([len(expected_steps) - 2, len(expected_steps) - 1]),
        )
    )
    if not valid_duplicate:
        raise AdvantageHeatmapError(
            "allowed terminal stability-guard logs must contain exactly two "
            "records at one final duplicate step"
        )

    guard_values = series["stability_guard/rejected"][1][-2:]
    if not np.array_equal(guard_values, np.asarray([0.0, 1.0])):
        raise AdvantageHeatmapError(
            "the final duplicate step must have stability_guard/rejected "
            "values 0 then 1"
        )


def _render_scalar_curve(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    tags: tuple[str, ...],
    *,
    title: str,
    ylabel: str,
    xlabel: str,
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
    ax.set_xlabel(xlabel)
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


def _render_policy_stability_curve(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    update_axis_label: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    panels = (
        (
            axes[0, 0],
            POST_UPDATE_KL_TAGS,
            "Post-update reference KL",
            "KL divergence",
        ),
        (
            axes[0, 1],
            ADAPTER_DRIFT_TAGS,
            "Maximum mode-adapter drift",
            "Relative drift",
        ),
        (
            axes[1, 0],
            ACTIVE_SIGNAL_TAGS,
            "Signal and no-signal modes",
            "Mode count",
        ),
        (
            axes[1, 1],
            GUARD_SIGNAL_TAGS,
            "Stability guard and zero-gradient update",
            "Count / indicator",
        ),
    )
    for ax, tags, title, ylabel in panels:
        for tag in tags:
            steps, values = series[tag]
            marker = "o" if len(steps) <= 100 else None
            ax.plot(
                steps,
                values,
                marker=marker,
                markersize=3.0,
                linewidth=1.4,
                label=tag,
            )
        ax.set_xlabel(update_axis_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.grid(color="#D9D9D9", linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5)
    axes[0, 0].axhline(0.25, color="#E15759", linestyle="--", linewidth=1.0)
    axes[0, 1].axhline(0.02, color="#E15759", linestyle="--", linewidth=1.0)
    fig.suptitle("GRPO policy-update stability", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _validate_aligned_scalar_steps(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    reward_tags: tuple[str, ...],
) -> None:
    for tags in (reward_tags, *_SCALAR_STEP_ALIGNMENT_GROUPS):
        expected_steps = series[tags[0]][0]
        for tag in tags[1:]:
            if not np.array_equal(series[tag][0], expected_steps):
                raise AdvantageHeatmapError(
                    "TensorBoard scalar tags must use aligned steps: "
                    + ", ".join(tags)
                )


def generate_grpo_plots(
    tb_dir: Path,
    output_dir: Path,
    *,
    require_frozen_pretrain_reward: bool = False,
    rollout_axis_label: str = FRESH_ROLLOUT_AXIS_LABEL,
    allow_terminal_guard_duplicate: bool = False,
) -> Mapping[str, Path]:
    """Generate GRPO plots, accepting legacy reward logs unless strict."""

    rollout_axis_label = _validate_rollout_axis_label(rollout_axis_label)
    if (
        allow_terminal_guard_duplicate
        and rollout_axis_label != ACCEPTED_ROLLOUT_AXIS_LABEL
    ):
        raise AdvantageHeatmapError(
            "allow_terminal_guard_duplicate requires Accepted update state"
        )
    accumulator = _load_event_accumulator(tb_dir)
    steps, advantages = _load_advantage_vectors_from_accumulator(
        accumulator, ADVANTAGE_VECTOR_TAG
    )
    available_scalar_tags = set(
        accumulator.Tags().get("scalars", ())
    )
    has_frozen_pretrain_reward = (
        FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG in available_scalar_tags
    )
    if require_frozen_pretrain_reward and not has_frozen_pretrain_reward:
        raise AdvantageHeatmapError(
            "TensorBoard is missing required scalar tag: "
            f"{FROZEN_PRETRAIN_RAW_PROXY_REWARD_TAG}"
        )
    reward_tags = (
        REWARD_CURVE_TAGS
        if has_frozen_pretrain_reward
        else LEGACY_REWARD_CURVE_TAGS
    )
    curve_specs = dict(_CURVE_SPECS)
    rollout_axis_curve_keys = ["reward_curve", "validation_reward_curve"]
    if rollout_axis_label == ACCEPTED_ROLLOUT_AXIS_LABEL:
        rollout_axis_curve_keys.extend(("grpo_loss_curve", "kl_loss_curve"))
    for key in rollout_axis_curve_keys:
        filename, tags, title, ylabel, _ = curve_specs[key]
        curve_specs[key] = (
            filename,
            tags,
            title,
            ylabel,
            rollout_axis_label,
        )
    reward_filename, _, reward_title, reward_ylabel, reward_xlabel = (
        curve_specs["reward_curve"]
    )
    curve_specs["reward_curve"] = (
        reward_filename,
        reward_tags,
        reward_title,
        reward_ylabel,
        reward_xlabel,
    )
    all_scalar_tags = tuple(
        tag
        for _, tags, _, _, _ in curve_specs.values()
        for tag in tags
    )
    all_scalar_tags += POLICY_STABILITY_CURVE_TAGS
    update_tags = tuple(
        dict.fromkeys(
            (
                *reward_tags,
                *GRPO_LOSS_CURVE_TAGS,
                *KL_LOSS_CURVE_TAGS,
                *POLICY_STABILITY_CURVE_TAGS,
            )
        )
    )
    scalar_series = _load_scalar_series(
        accumulator,
        all_scalar_tags,
        duplicate_step_tags=(
            frozenset(update_tags)
            if allow_terminal_guard_duplicate
            else frozenset()
        ),
    )
    if allow_terminal_guard_duplicate:
        _validate_terminal_guard_duplicate(scalar_series, update_tags)
    _validate_aligned_scalar_steps(
        scalar_series,
        reward_tags=reward_tags,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {
        "advantage_heatmap": _render_advantage_heatmap(
            steps,
            advantages,
            output_dir / "advantage_vector_heatmap.png",
            rollout_axis_label=rollout_axis_label,
        )
    }
    for key, (filename, tags, title, ylabel, xlabel) in curve_specs.items():
        outputs[key] = _render_scalar_curve(
            scalar_series,
            tags,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            output_path=output_dir / filename,
        )
    outputs["policy_stability_curve"] = _render_policy_stability_curve(
        scalar_series,
        output_dir / "policy_stability_curve.png",
        update_axis_label=(
            rollout_axis_label
            if rollout_axis_label == ACCEPTED_ROLLOUT_AXIS_LABEL
            else OPTIMIZER_STEP_AXIS_LABEL
        ),
    )
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot GRPO training diagnostics from TensorBoard events."
    )
    parser.add_argument("--tensorboard-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--rollout-axis-label",
        choices=ROLLOUT_AXIS_LABELS,
        default=FRESH_ROLLOUT_AXIS_LABEL,
    )
    parser.add_argument(
        "--allow-terminal-guard-duplicate",
        action="store_true",
        help=(
            "Allow one aligned final duplicate accepted-update step when "
            "stability_guard/rejected transitions from 0 to 1."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    for name, output_path in generate_grpo_plots(
        args.tensorboard_dir,
        args.output_dir,
        rollout_axis_label=args.rollout_axis_label,
        allow_terminal_guard_duplicate=args.allow_terminal_guard_duplicate,
    ).items():
        print(f"{name}={output_path}")


if __name__ == "__main__":
    main()
