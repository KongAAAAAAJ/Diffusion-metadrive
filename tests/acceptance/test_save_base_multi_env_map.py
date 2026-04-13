from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_script_module():
    script_path = Path("scripts/save_base_multi_env_map.py")
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    base_multi_env_module = types.ModuleType("metadrive.envs.diffusion_envs.base_multi_env")
    base_multi_env_module.BaseMultiEnv = _FakeEnv
    register("metadrive.envs.diffusion_envs.base_multi_env", base_multi_env_module)

    draw_module = types.ModuleType("metadrive.utils.draw_top_down_map")
    draw_module.draw_top_down_map = lambda current_map, resolution: [[1]]
    register("metadrive.utils.draw_top_down_map", draw_module)

    spec = importlib.util.spec_from_file_location("save_base_multi_env_map", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in stubbed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


class _FakeEnv:
    last_config = None
    close_calls = 0

    @staticmethod
    def default_config():
        return {
            "use_hybrid_map": True,
            "route_preset": "mainline",
            "hybrid_map_blocks_config": [
                {"block_id": "s0", "id": "s", "parent_block_id": "root", "parent_socket_index": 0, "length": 42}
            ],
        }

    def __init__(self, config):
        type(self).last_config = dict(config)
        self.current_map = object()

    def reset(self):
        return {"agent0": {}}, {}

    def close(self):
        type(self).close_calls += 1


class _FakePlt:
    saved_paths = []

    @classmethod
    def imshow(cls, image, cmap=None):
        return image, cmap

    @classmethod
    def axis(cls, mode):
        return mode

    @classmethod
    def savefig(cls, path, dpi=200, bbox_inches="tight"):
        cls.saved_paths.append((str(path), dpi, bbox_inches))
        Path(path).write_bytes(b"fake-map")

    @classmethod
    def close(cls):
        return None


def test_save_base_multi_env_map_creates_png_and_closes_env(tmp_path, monkeypatch):
    module = _load_script_module()
    output_path = tmp_path / "maps" / "base_map.png"

    monkeypatch.setattr(module, "BaseMultiEnv", _FakeEnv)
    monkeypatch.setattr(module, "draw_top_down_map", lambda current_map, resolution: [[1, 2], [3, 4]])
    monkeypatch.setattr(module, "plt", _FakePlt)

    _FakeEnv.close_calls = 0
    _FakePlt.saved_paths = []
    exit_code = module.main(["--output", str(output_path), "--resolution", "256"])

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake-map"
    assert _FakeEnv.close_calls == 1
    assert _FakePlt.saved_paths[0][0] == str(output_path)
    assert _FakeEnv.last_config["hybrid_map_blocks_config"] == _FakeEnv.default_config()["hybrid_map_blocks_config"]


def test_save_base_multi_env_map_passes_seed_and_hybrid_overrides(tmp_path, monkeypatch):
    module = _load_script_module()
    output_path = tmp_path / "map.png"

    monkeypatch.setattr(module, "BaseMultiEnv", _FakeEnv)
    monkeypatch.setattr(module, "draw_top_down_map", lambda current_map, resolution: [[0]])
    monkeypatch.setattr(module, "plt", _FakePlt)

    exit_code = module.main(
        [
            "--output",
            str(output_path),
            "--resolution",
            "512",
            "--start-seed",
            "7",
            "--num-scenarios",
            "3",
            "--use-hybrid-map",
            "1",
        ]
    )

    assert exit_code == 0
    assert _FakeEnv.last_config["start_seed"] == 7
    assert _FakeEnv.last_config["num_scenarios"] == 3
    assert _FakeEnv.last_config["use_hybrid_map"] is True
    assert _FakeEnv.last_config["traffic_density"] == 0.0
    assert _FakeEnv.last_config["random_traffic"] is False


def test_save_base_multi_env_map_passes_route_preset_override(tmp_path, monkeypatch):
    module = _load_script_module()
    output_path = tmp_path / "map.png"

    monkeypatch.setattr(module, "BaseMultiEnv", _FakeEnv)
    monkeypatch.setattr(module, "draw_top_down_map", lambda current_map, resolution: [[0]])
    monkeypatch.setattr(module, "plt", _FakePlt)

    exit_code = module.main(
        [
            "--output",
            str(output_path),
            "--route-preset",
            "ramp_merge",
        ]
    )

    assert exit_code == 0
    assert _FakeEnv.last_config["route_preset"] == "ramp_merge"


def test_save_base_multi_env_map_passes_blocks_config_override(tmp_path, monkeypatch):
    module = _load_script_module()
    output_path = tmp_path / "map.png"
    blocks_config = [
        {"block_id": "s0", "id": "s", "parent_block_id": "root", "parent_socket_index": 0, "length": 80},
        {"block_id": "c0", "id": "c", "parent_block_id": "s0", "parent_socket_index": 0, "length": 50, "radius": 45, "angle": 90, "dir": 1},
    ]

    monkeypatch.setattr(module, "BaseMultiEnv", _FakeEnv)
    monkeypatch.setattr(module, "draw_top_down_map", lambda current_map, resolution: [[0]])
    monkeypatch.setattr(module, "plt", _FakePlt)

    exit_code = module.main(
        [
            "--output",
            str(output_path),
            "--hybrid-map-blocks-config",
            json.dumps(blocks_config),
        ]
    )

    assert exit_code == 0
    assert _FakeEnv.last_config["hybrid_map_blocks_config"] == blocks_config
