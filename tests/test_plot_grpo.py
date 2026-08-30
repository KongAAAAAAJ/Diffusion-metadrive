from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from torch.utils.tensorboard import SummaryWriter

import evaluation.plot_grpo as plot_grpo
from evaluation.plot_grpo import (
    ADVANTAGE_VECTOR_TAG,
    GRPO_LOSS_CURVE_TAGS,
    KL_LOSS_CURVE_TAGS,
    REWARD_CURVE_TAGS,
    VALIDATION_REWARD_CURVE_TAGS,
    AdvantageHeatmapError,
    generate_advantage_heatmap,
    generate_grpo_plots,
    load_advantage_vectors,
)


def _write_tensor_events(
    tb_dir: Path,
    events: list[tuple[int, np.ndarray]],
    *,
    tag: str = ADVANTAGE_VECTOR_TAG,
) -> None:
    with SummaryWriter(log_dir=str(tb_dir)) as writer:
        for step, value in events:
            writer.add_tensor(tag, torch.as_tensor(value), global_step=step)


def test_load_advantage_vectors_preserves_more_than_ten_complete_events(
    tmp_path: Path,
) -> None:
    tb_dir = tmp_path / "tb"
    written_steps = np.asarray(
        [91, 7, 63, 14, 70, 28, 105, 42, 21, 84, 49, 35, 98, 56, 77],
        dtype=np.int64,
    )
    events = [
        (
            int(step),
            np.asarray(
                [[step + 0.125, -step - 0.25, step / 3.0, -step / 7.0]],
                dtype=np.float32,
            ),
        )
        for step in written_steps
    ]
    _write_tensor_events(tb_dir, events)

    steps, values = load_advantage_vectors(tb_dir)

    expected_order = np.argsort(written_steps)
    expected_values = np.concatenate(
        [events[index][1] for index in expected_order], axis=0
    )
    np.testing.assert_array_equal(steps, written_steps[expected_order])
    np.testing.assert_array_equal(values, expected_values)
    assert values.shape == (15, 4)
    assert values.dtype == np.float32


def test_generate_advantage_heatmap_uses_group_by_step_orientation_and_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tb_dir = tmp_path / "tb"
    steps = [5, 11, 20]
    values = np.asarray(
        [
            [-2.5, -1.0, 0.25, 1.5],
            [-0.5, 0.0, 0.75, 2.0],
            [-1.5, -0.25, 1.0, 2.25],
        ],
        dtype=np.float32,
    )
    _write_tensor_events(
        tb_dir,
        [(step, values[index : index + 1]) for index, step in enumerate(steps)],
    )

    captured: dict[str, object] = {}
    original_imshow = Axes.imshow

    def record_imshow(
        self: Axes,
        image_values: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        captured["axis"] = self
        captured["values"] = np.asarray(image_values).copy()
        captured["kwargs"] = kwargs.copy()
        return original_imshow(self, image_values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", record_imshow)
    output_path = tmp_path / "plots" / "advantage_heatmap.png"

    result = generate_advantage_heatmap(tb_dir, output_path)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    np.testing.assert_array_equal(captured["values"], values.T)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["cmap"] == "RdBu_r"
    assert kwargs["vmin"] == -2.5
    assert kwargs["vmax"] == 2.5
    assert kwargs["origin"] == "lower"

    axis = captured["axis"]
    assert isinstance(axis, Axes)
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "5",
        "11",
        "20",
    ]
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "g0",
        "g1",
        "g2",
        "g3",
    ]
    assert axis.get_xlabel() == "Optimizer step"
    assert any(
        "no identity across steps" in text.get_text()
        for text in axis.figure.texts
    )


def test_load_advantage_vectors_rejects_missing_tag(tmp_path: Path) -> None:
    tb_dir = tmp_path / "tb"
    _write_tensor_events(
        tb_dir,
        [(1, np.zeros((1, 4), dtype=np.float32))],
        tag="advantage/other",
    )

    with pytest.raises(AdvantageHeatmapError, match="was not found"):
        load_advantage_vectors(tb_dir)


@pytest.mark.parametrize(
    "bad_shape",
    [
        (4,),
        (2, 4),
        (1, 0),
    ],
)
def test_load_advantage_vectors_rejects_bad_shape(
    tmp_path: Path, bad_shape: tuple[int, ...]
) -> None:
    tb_dir = tmp_path / "tb"
    _write_tensor_events(
        tb_dir,
        [(3, np.zeros(bad_shape, dtype=np.float32))],
    )

    with pytest.raises(AdvantageHeatmapError, match=r"shape \[1,G\]"):
        load_advantage_vectors(tb_dir)


def test_load_advantage_vectors_rejects_inconsistent_group_size(
    tmp_path: Path,
) -> None:
    tb_dir = tmp_path / "tb"
    _write_tensor_events(
        tb_dir,
        [
            (1, np.zeros((1, 4), dtype=np.float32)),
            (2, np.zeros((1, 3), dtype=np.float32)),
        ],
    )

    with pytest.raises(AdvantageHeatmapError, match="consistent group size"):
        load_advantage_vectors(tb_dir)


def test_load_advantage_vectors_rejects_duplicate_step(tmp_path: Path) -> None:
    tb_dir = tmp_path / "tb"
    _write_tensor_events(
        tb_dir,
        [
            (8, np.zeros((1, 4), dtype=np.float32)),
            (8, np.ones((1, 4), dtype=np.float32)),
        ],
    )

    with pytest.raises(AdvantageHeatmapError, match="duplicate step 8"):
        load_advantage_vectors(tb_dir)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_load_advantage_vectors_rejects_non_finite_values(
    tmp_path: Path, invalid_value: float
) -> None:
    tb_dir = tmp_path / "tb"
    values = np.zeros((1, 4), dtype=np.float32)
    values[0, 2] = invalid_value
    _write_tensor_events(tb_dir, [(12, values)])

    with pytest.raises(AdvantageHeatmapError, match="only finite values"):
        load_advantage_vectors(tb_dir)


def _write_complete_grpo_events(tb_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    training_steps = np.arange(1, 16, dtype=np.int64)
    validation_steps = np.asarray([5, 10, 15], dtype=np.int64)
    scalar_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag_index, tag in enumerate(
        (
            *REWARD_CURVE_TAGS,
            *VALIDATION_REWARD_CURVE_TAGS,
            *GRPO_LOSS_CURVE_TAGS,
            *KL_LOSS_CURVE_TAGS,
        )
    ):
        steps = (
            validation_steps
            if tag.startswith("validation/")
            else training_steps
        )
        values = (
            0.1 * steps.astype(np.float32)
            + np.float32(tag_index)
            - np.float32(2.0)
        )
        scalar_series[tag] = (steps, values)

    with SummaryWriter(log_dir=str(tb_dir)) as writer:
        for step in training_steps:
            advantage = torch.tensor(
                [[-1.5, -0.5, 0.5, 1.5]], dtype=torch.float32
            ) + float(step) / 100.0
            writer.add_tensor(
                ADVANTAGE_VECTOR_TAG,
                advantage,
                global_step=int(step),
            )
        for tag, (steps, values) in scalar_series.items():
            for step, value in zip(steps, values):
                writer.add_scalar(tag, float(value), global_step=int(step))
    return scalar_series


def test_generate_grpo_plots_round_trips_tags_steps_and_writes_five_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tb_dir = tmp_path / "tb"
    expected_series = _write_complete_grpo_events(tb_dir)
    output_dir = tmp_path / "plots"

    size_guidance_calls: list[dict[str, int]] = []
    original_accumulator = plot_grpo.event_accumulator.EventAccumulator

    def record_accumulator(*args: object, **kwargs: object) -> object:
        size_guidance_calls.append(dict(kwargs["size_guidance"]))
        return original_accumulator(*args, **kwargs)

    monkeypatch.setattr(
        plot_grpo.event_accumulator,
        "EventAccumulator",
        record_accumulator,
    )

    plot_calls: list[
        tuple[Axes, str, np.ndarray, np.ndarray]
    ] = []
    original_plot = Axes.plot

    def record_plot(
        self: Axes,
        x: object,
        y: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        plot_calls.append(
            (
                self,
                str(kwargs["label"]),
                np.asarray(x).copy(),
                np.asarray(y).copy(),
            )
        )
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)

    outputs = generate_grpo_plots(tb_dir, output_dir)

    assert set(outputs) == {
        "advantage_heatmap",
        "reward_curve",
        "validation_reward_curve",
        "grpo_loss_curve",
        "kl_loss_curve",
    }
    assert {key: path.name for key, path in outputs.items()} == {
        "advantage_heatmap": "advantage_vector_heatmap.png",
        "reward_curve": "reward_curve.png",
        "validation_reward_curve": "validation_reward_curve.png",
        "grpo_loss_curve": "grpo_loss_curve.png",
        "kl_loss_curve": "kl_loss_curve.png",
    }
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 1_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    assert size_guidance_calls == [
        {
            plot_grpo.event_accumulator.TENSORS: 0,
            plot_grpo.event_accumulator.SCALARS: 0,
        }
    ]
    assert len(plot_calls) == 11
    calls_by_tag = {tag: (axis, steps, values) for axis, tag, steps, values in plot_calls}
    assert set(calls_by_tag) == set(expected_series)
    for tag, (expected_steps, expected_values) in expected_series.items():
        axis, actual_steps, actual_values = calls_by_tag[tag]
        np.testing.assert_array_equal(actual_steps, expected_steps)
        np.testing.assert_allclose(actual_values, expected_values, rtol=0, atol=1e-6)
        assert axis.get_xlabel() == "Optimizer step"

    reward_axes = {calls_by_tag[tag][0] for tag in REWARD_CURVE_TAGS}
    validation_reward_axes = {
        calls_by_tag[tag][0] for tag in VALIDATION_REWARD_CURVE_TAGS
    }
    grpo_axes = {calls_by_tag[tag][0] for tag in GRPO_LOSS_CURVE_TAGS}
    kl_axes = {calls_by_tag[tag][0] for tag in KL_LOSS_CURVE_TAGS}
    assert (
        len(reward_axes)
        == len(validation_reward_axes)
        == len(grpo_axes)
        == len(kl_axes)
        == 1
    )
    assert reward_axes.isdisjoint(validation_reward_axes)
    assert next(iter(reward_axes)).get_ylabel() == "Raw tau_d reward"
    validation_reward_axis = next(iter(validation_reward_axes))
    assert validation_reward_axis.get_ylabel() == "Raw tau_d reward / gain"
    assert (
        validation_reward_axis.get_title()
        == "GRPO validation reward and gain curves"
    )
    assert next(iter(grpo_axes)).get_ylabel() == "Loss"
    assert next(iter(kl_axes)).get_ylabel() == "Unweighted KL divergence"


def _write_all_scalar_tags(
    writer: SummaryWriter,
    *,
    duplicate_tag: str | None = None,
    non_finite_tag: str | None = None,
) -> None:
    for tag in (
        *REWARD_CURVE_TAGS,
        *VALIDATION_REWARD_CURVE_TAGS,
        *GRPO_LOSS_CURVE_TAGS,
        *KL_LOSS_CURVE_TAGS,
    ):
        value = np.nan if tag == non_finite_tag else 0.25
        writer.add_scalar(tag, value, global_step=1)
        if tag == duplicate_tag:
            writer.add_scalar(tag, 0.5, global_step=1)


def test_generate_grpo_plots_rejects_missing_validation_scalar_tag(
    tmp_path: Path,
) -> None:
    tb_dir = tmp_path / "tb"
    with SummaryWriter(log_dir=str(tb_dir)) as writer:
        writer.add_tensor(
            ADVANTAGE_VECTOR_TAG,
            torch.zeros((1, 4), dtype=torch.float32),
            global_step=1,
        )
        for tag in (
            *REWARD_CURVE_TAGS,
            *VALIDATION_REWARD_CURVE_TAGS,
            *GRPO_LOSS_CURVE_TAGS,
            *KL_LOSS_CURVE_TAGS,
        ):
            if tag != "validation/reward_gain":
                writer.add_scalar(tag, 0.25, global_step=1)

    with pytest.raises(AdvantageHeatmapError, match="missing required scalar tags"):
        generate_grpo_plots(tb_dir, tmp_path / "plots")


def test_generate_grpo_plots_rejects_duplicate_validation_scalar_step(
    tmp_path: Path,
) -> None:
    tb_dir = tmp_path / "tb"
    with SummaryWriter(log_dir=str(tb_dir)) as writer:
        writer.add_tensor(
            ADVANTAGE_VECTOR_TAG,
            torch.zeros((1, 4), dtype=torch.float32),
            global_step=1,
        )
        _write_all_scalar_tags(
            writer,
            duplicate_tag="validation/reward_gain",
        )

    with pytest.raises(AdvantageHeatmapError, match="duplicate step 1"):
        generate_grpo_plots(tb_dir, tmp_path / "plots")


def test_generate_grpo_plots_rejects_non_finite_validation_scalar(
    tmp_path: Path,
) -> None:
    tb_dir = tmp_path / "tb"
    with SummaryWriter(log_dir=str(tb_dir)) as writer:
        writer.add_tensor(
            ADVANTAGE_VECTOR_TAG,
            torch.zeros((1, 4), dtype=torch.float32),
            global_step=1,
        )
        _write_all_scalar_tags(
            writer,
            non_finite_tag="validation/pretrain_reward",
        )

    with pytest.raises(AdvantageHeatmapError, match="must be finite"):
        generate_grpo_plots(tb_dir, tmp_path / "plots")


def test_generate_grpo_plots_rejects_misaligned_validation_scalar_steps(
    tmp_path: Path,
) -> None:
    tb_dir = tmp_path / "tb"
    with SummaryWriter(log_dir=str(tb_dir)) as writer:
        writer.add_tensor(
            ADVANTAGE_VECTOR_TAG,
            torch.zeros((1, 4), dtype=torch.float32),
            global_step=1,
        )
        _write_all_scalar_tags(writer)
        writer.add_scalar("validation/reward_gain", 0.5, global_step=2)

    with pytest.raises(AdvantageHeatmapError, match="aligned optimizer steps"):
        generate_grpo_plots(tb_dir, tmp_path / "plots")


def test_grpo_plot_cli_uses_tensorboard_and_output_directories() -> None:
    args = plot_grpo._build_parser().parse_args(
        [
            "--tensorboard-dir",
            "/tmp/source_tb",
            "--output-dir",
            "/tmp/output_plots",
        ]
    )

    assert args.tensorboard_dir == Path("/tmp/source_tb")
    assert args.output_dir == Path("/tmp/output_plots")
