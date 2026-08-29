from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from evaluation.plot_grpo_reward_boxplots import (
    GROUPED_REWARD_BOXPLOT_FILENAME,
    GROUPED_REWARD_COLUMNS,
    MODEL_COLUMN,
    MODEL_COLORS,
    MODEL_ORDER,
    REWARD_COLUMNS,
    SINGLE_REWARD_COLUMNS,
    RewardBoxplotError,
    generate_reward_boxplots,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    for model, offset in (("stage1_a", 0.0), ("grpo_open", 0.25)):
        for step in range(4):
            row: dict[str, object] = {
                "model": model,
                "scenario": "S5_hard_brake_lead",
                "seed": 31,
                "step": step,
            }
            for index, reward in enumerate(REWARD_COLUMNS):
                row[reward] = offset + 0.1 * step - 0.05 * index
            rows.append(row)
    return rows


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    selected_fields = fieldnames or [
        "model",
        "scenario",
        "seed",
        "step",
        *REWARD_COLUMNS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_generate_reward_boxplots_writes_group_and_five_single_pngs_with_raw_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "step_rewards.csv"
    output_dir = tmp_path / "charts"
    _write_csv(csv_path, _rows())
    output_dir.mkdir()
    legacy_grouped_paths = [
        output_dir / f"{reward}_boxplot.png" for reward in GROUPED_REWARD_COLUMNS
    ]
    for path in legacy_grouped_paths:
        path.write_bytes(b"legacy")

    scatter_calls: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
    saved_grids: dict[str, tuple[int, set[tuple[int, int]]]] = {}
    original_scatter = Axes.scatter
    original_savefig = Figure.savefig

    def record_scatter(
        self: Axes,
        x: object,
        y: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        scatter_calls.append((np.asarray(x), np.asarray(y), kwargs.copy()))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)

    def record_savefig(
        self: Figure,
        output_path: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        grid_shapes = {
            (
                ax.get_subplotspec().get_gridspec().nrows,
                ax.get_subplotspec().get_gridspec().ncols,
            )
            for ax in self.axes
        }
        saved_grids[Path(output_path).name] = (len(self.axes), grid_shapes)
        return original_savefig(self, output_path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_savefig)
    outputs = generate_reward_boxplots(csv_path, output_dir)

    expected_names = {GROUPED_REWARD_BOXPLOT_FILENAME}
    expected_names.update(
        f"{reward}_boxplot.png" for reward in SINGLE_REWARD_COLUMNS
    )
    assert len(outputs) == len(expected_names) == 6
    assert {path.name for path in outputs} == expected_names
    for path in outputs:
        assert path.is_file()
        assert path.stat().st_size > 1_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not any(path.exists() for path in legacy_grouped_paths)

    assert saved_grids[GROUPED_REWARD_BOXPLOT_FILENAME] == (4, {(2, 2)})
    for reward in SINGLE_REWARD_COLUMNS:
        assert saved_grids[f"{reward}_boxplot.png"] == (1, {(1, 1)})

    assert len(scatter_calls) == len(REWARD_COLUMNS) * len(MODEL_ORDER) == 18
    assert sum(len(y) for _, y, _ in scatter_calls) == 72
    for call_index, (x, y, kwargs) in enumerate(scatter_calls):
        model_index = call_index % len(MODEL_ORDER)
        model = MODEL_ORDER[model_index]
        position = model_index + 1
        assert x.shape == y.shape == (4,)
        assert np.all(x >= position - 0.14)
        assert np.all(x <= position + 0.14)
        assert not np.allclose(x, position)
        assert kwargs["alpha"] == 0.30
        assert kwargs["s"] == 16
        assert kwargs["color"] == MODEL_COLORS[model]
        assert kwargs["edgecolors"] == "none"
        assert kwargs["zorder"] == 3


def test_reward_boxplots_reject_missing_required_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_column.csv"
    fieldnames = [MODEL_COLUMN, *REWARD_COLUMNS[:-1]]
    _write_csv(csv_path, _rows(), fieldnames=fieldnames)

    with pytest.raises(RewardBoxplotError, match="missing required columns"):
        generate_reward_boxplots(csv_path, tmp_path / "charts")


def test_reward_boxplots_reject_missing_model(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_model.csv"
    rows = [row for row in _rows() if row[MODEL_COLUMN] == "stage1_a"]
    _write_csv(csv_path, rows)

    with pytest.raises(RewardBoxplotError, match="missing models: grpo_open"):
        generate_reward_boxplots(csv_path, tmp_path / "charts")


@pytest.mark.parametrize("invalid_value", ["nan", "inf", "-inf"])
def test_reward_boxplots_reject_non_finite_reward(
    tmp_path: Path, invalid_value: str
) -> None:
    csv_path = tmp_path / f"non_finite_{invalid_value}.csv"
    rows = _rows()
    rows[0]["total_reward"] = invalid_value
    _write_csv(csv_path, rows)

    with pytest.raises(RewardBoxplotError, match="must be finite"):
        generate_reward_boxplots(csv_path, tmp_path / "charts")


def test_reward_boxplots_reject_unknown_model(tmp_path: Path) -> None:
    csv_path = tmp_path / "unknown_model.csv"
    rows = _rows()
    rows[0][MODEL_COLUMN] = "other"
    _write_csv(csv_path, rows)

    with pytest.raises(RewardBoxplotError, match="unsupported model"):
        generate_reward_boxplots(csv_path, tmp_path / "charts")
