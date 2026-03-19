from __future__ import annotations

from pathlib import Path

from metadrive.policy.diffusion_policy.train_transfuser import create_next_run_dir


def test_create_next_run_dir_uses_existing_run_count(tmp_path: Path):
    output_root = tmp_path / "diffusion"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoints").mkdir()
    (output_root / "tb").mkdir()

    run_1 = create_next_run_dir(output_root)
    assert run_1.name == "run_1"
    assert run_1.exists()

    run_2 = create_next_run_dir(output_root)
    assert run_2.name == "run_2"
    assert run_2.exists()


def test_create_next_run_dir_ignores_non_run_directories(tmp_path: Path):
    output_root = tmp_path / "diffusion"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_1").mkdir()
    (output_root / "open_loop_eval").mkdir()
    (output_root / "closed_loop").mkdir()

    run_dir = create_next_run_dir(output_root)
    assert run_dir.name == "run_2"
