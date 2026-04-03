import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "metadrive"
        / "policy"
        / "diffusion_policy"
        / "run_dir_utils.py"
    )
    spec = importlib.util.spec_from_file_location("run_dir_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_numbered_run_dir_starts_at_run_1(tmp_path: Path):
    module = _load_module()
    run_dir = module.create_numbered_run_dir(tmp_path)

    assert run_dir == tmp_path / "run_1"
    assert run_dir.is_dir()


def test_create_numbered_run_dir_counts_existing_run_directories(tmp_path: Path):
    module = _load_module()
    (tmp_path / "run_1").mkdir()
    (tmp_path / "run_3").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "run_tmp").mkdir()

    run_dir = module.create_numbered_run_dir(tmp_path)

    assert run_dir == tmp_path / "run_4"
    assert run_dir.is_dir()
