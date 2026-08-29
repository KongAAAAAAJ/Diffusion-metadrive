from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from torch.utils.tensorboard import SummaryWriter

from evaluation.plot_grpo_advantage_heatmap import (
    ADVANTAGE_VECTOR_TAG,
    AdvantageHeatmapError,
    generate_advantage_heatmap,
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
