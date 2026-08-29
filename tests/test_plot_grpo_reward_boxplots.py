from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evaluation.plot_grpo_reward_boxplots import (
    MODEL_COLUMN,
    REWARD_COLUMNS,
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


def test_generate_reward_boxplots_writes_nine_valid_pngs(tmp_path: Path) -> None:
    csv_path = tmp_path / "step_rewards.csv"
    output_dir = tmp_path / "charts"
    _write_csv(csv_path, _rows())

    outputs = generate_reward_boxplots(csv_path, output_dir)

    assert len(outputs) == len(REWARD_COLUMNS) == 9
    assert {path.name for path in outputs} == {
        f"{reward}_boxplot.png" for reward in REWARD_COLUMNS
    }
    for path in outputs:
        assert path.is_file()
        assert path.stat().st_size > 1_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


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
