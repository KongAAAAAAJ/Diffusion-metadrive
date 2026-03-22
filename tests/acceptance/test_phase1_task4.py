from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_phase1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase1_verify_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_verify_phase1_file_has_no_syntax_error():
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


def test_parse_args_supports_episodes_and_render():
    module = _load_module()
    args = module.parse_args(["--episodes", "5", "--render", "0"])
    assert args.episodes == 5
    assert args.render == 0


def test_parse_args_supports_max_steps():
    module = _load_module()
    args = module.parse_args(["--max-steps", "100"])
    assert args.max_steps == 100


def test_script_imports_platoon_env_and_idm_policy():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "from envs.platoon_env import PlatoonEnv" in source
    assert "from metadrive.policy.idm_policy import IDMPolicy" in source


def test_main_accepts_argv_parameter():
    module = _load_module()
    signature = inspect.signature(module.main)
    assert "argv" in signature.parameters
