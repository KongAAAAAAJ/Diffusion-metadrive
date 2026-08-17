from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import evaluation.bev_evaluation_artifacts as artifacts
from evaluation.bev_evaluation_artifacts import (
    ClosedLoopArtifactWriter,
    EvaluationArtifactError,
    OpenLoopArtifactCollector,
    _capture_metadrive_topdown_frame,
)


class _FakeFrameCanvas:
    def pos2pix(self, x: float, y: float) -> tuple[float, float]:
        return (float(x), float(y))


class _FakeTopDownRenderer:
    def __init__(self, surface: object) -> None:
        self.position = None
        self._screen_canvas = surface
        self._frame_canvas = _FakeFrameCanvas()

    @staticmethod
    def _world_to_screen_position(
        point: np.ndarray, offset: tuple[float, float]
    ) -> tuple[float, float]:
        return (
            float(point[0]) - float(offset[0]),
            float(point[1]) - float(offset[1]),
        )


class _FirstRenderCreatesRendererEnv:
    def __init__(self, surface: object) -> None:
        self.engine = None
        self.surface = surface
        self.render_calls: list[dict[str, object]] = []

    def render(self, **kwargs: object) -> None:
        self.render_calls.append(dict(kwargs))
        self.top_down_renderer = _FakeTopDownRenderer(self.surface)


def test_open_loop_artifacts_include_legacy_metrics_tables_and_images(
    tmp_path: Path,
) -> None:
    batch = {
        "bev": torch.zeros((1, 3, 8, 16, 16), dtype=torch.uint8),
        "expert_trajectory": torch.zeros((1, 3, 8, 3), dtype=torch.float32),
        "gt_mode": torch.tensor([[0, 1, 2]], dtype=torch.int64),
    }
    selected = torch.zeros((1, 3, 8, 3), dtype=torch.float32)
    selected[:, :, :, 0] = torch.arange(8, dtype=torch.float32)
    candidates = selected[:, :, None].repeat(1, 1, 10, 1, 1)
    logits = torch.zeros((1, 3, 10), dtype=torch.float32)
    logits[0, 0, 0] = 2.0
    logits[0, 1, 1] = 2.0
    logits[0, 2, 2] = 2.0
    output = {
        "selected_mode": torch.tensor([[0, 1, 2]], dtype=torch.int64),
        "mode_logits": logits,
        "selected_trajectory": selected,
        "trajectory_candidates": candidates,
    }

    collector = OpenLoopArtifactCollector(tmp_path, save_visualizations=True)
    collector.add_batch(batch, output)
    report = collector.finalize()

    metrics = report["metrics"]
    assert metrics["mode_accuracy"] == pytest.approx(1.0)
    assert metrics["lateral_accuracy"] == pytest.approx(1.0)
    assert metrics["ade_mean"] > 0.0
    assert metrics["mode_confusion"]["KEEP_HIGH"]["KEEP_HIGH"] == 1
    assert (tmp_path / "open_loop_samples.json").is_file()
    assert (tmp_path / "open_loop_samples.csv").is_file()
    assert (tmp_path / "images" / "sample_00000.png").is_file()
    assert (tmp_path / "trajectory_plots" / "sample_00000.png").is_file()


def test_closed_loop_artifacts_include_step_audit_control_plots_and_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_video(path: Path, frames: object, fps: int) -> None:
        assert frames
        assert fps == 10
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"diagnostic-video")

    monkeypatch.setattr(artifacts, "_write_video", fake_video)
    renderer_frame = np.full((24, 24, 3), 73, dtype=np.uint8)
    capture_calls: list[dict[str, object]] = []

    def fake_capture(
        env: object,
        overlays: object,
        **kwargs: object,
    ) -> np.ndarray:
        capture_calls.append({"env": env, "overlays": overlays, "kwargs": kwargs})
        return renderer_frame.copy()

    monkeypatch.setattr(artifacts, "_capture_metadrive_topdown_frame", fake_capture)
    writer = ClosedLoopArtifactWriter(tmp_path, "stage1_a")
    episode_id = writer.start_episode(("S5_hard_brake_lead", "R1_entry_straight"), 17)
    pre = np.asarray(((0.0, 0.0, 0.0), (-8.0, 0.0, 0.0), (-16.0, 0.0, 0.0)))
    post = pre.copy()
    post[:, 0] += 0.2
    trajectories = np.zeros((3, 8, 3), dtype=np.float32)
    trajectories[:, :, 0] = np.arange(1, 9, dtype=np.float32)
    writer.capture_topdown_frame(
        episode_id,
        env=object(),
        step_index=0,
        pre_poses=pre,
        trajectories=trajectories,
    )
    writer.record_step(
        episode_id,
        step_index=0,
        dt_s=0.1,
        pre_poses=pre,
        post_poses=post,
        trajectories=trajectories,
        selected_modes=(0, 1, 2),
        rewards={"agent0": 1.0, "agent1": 0.5, "agent2": 0.25},
        controls={agent_id: np.asarray((0.1, 0.2)) for agent_id in artifacts.AGENT_IDS},
        bev=np.zeros((3, 8, 16, 16), dtype=np.uint8),
    )
    writer.finish_episode(episode_id)
    report = writer.finalize()

    summary = json.loads(Path(report["control_error_summary"]).read_text())
    assert summary["num_role_steps"] == 3
    root = tmp_path / "stage1_a"
    assert (root / "trajectory_data" / episode_id / "step_00000.json").is_file()
    assert (root / "step_records.jsonl").is_file()
    assert (root / "coordinate_audit" / f"{episode_id}.jsonl").is_file()
    assert (root / "combined_traj_frames" / episode_id / "step_00000.png").is_file()
    assert (root / "trajectory_plot_frames" / episode_id / "step_00000.png").is_file()
    np.testing.assert_array_equal(
        np.asarray(
            Image.open(root / "combined_traj_frames" / episode_id / "step_00000.png")
        ),
        renderer_frame,
    )
    assert capture_calls[0]["kwargs"]["camera_position"] == pytest.approx((-8.0, 0.0))
    overlays = capture_calls[0]["overlays"]
    assert len(overlays) == 27
    line_overlays = [value for value in overlays if value["type"] == "line"]
    circle_overlays = [value for value in overlays if value["type"] == "circle"]
    assert len(line_overlays) == 3
    assert len(circle_overlays) == 24
    for role, line in enumerate(line_overlays):
        np.testing.assert_allclose(line["world_points"][0], pre[role, :2])
    assert (root / "step_images" / episode_id / "step_00000.png").is_file()
    assert (root / "videos_2d" / f"{episode_id}.mp4").is_file()
    assert (root / "videos_trajectory_plot" / f"{episode_id}.mp4").is_file()
    assert (root / "videos_semantic_bev" / f"{episode_id}.mp4").is_file()
    assert (root / "control_errors" / f"{episode_id}_lat_error.png").is_file()
    assert (root / "control_errors" / f"{episode_id}_speed_error.png").is_file()
    assert (root / "control_errors" / f"{episode_id}_trajectory_compare.png").is_file()
    assert report["trajectory_plot_frames"].endswith("trajectory_plot_frames")
    assert report["trajectory_plot_videos"].endswith("videos_trajectory_plot")


def test_metadrive_topdown_draws_on_renderer_before_extracting_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pygame
    from metadrive.engine.top_down_renderer import WorldSurface

    if not pygame.get_init():
        pygame.init()
    surface = pygame.Surface((64, 64))
    surface.fill((1, 2, 3))
    env = _FirstRenderCreatesRendererEnv(surface)
    events: list[str] = []
    original_lines = pygame.draw.lines
    original_circle = pygame.draw.circle

    def draw_lines(*args: object, **kwargs: object) -> object:
        events.append("line")
        return original_lines(*args, **kwargs)

    def draw_circle(*args: object, **kwargs: object) -> object:
        events.append("circle")
        return original_circle(*args, **kwargs)

    def extract_frame(canvas: object) -> np.ndarray:
        events.append("extract")
        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    monkeypatch.setattr(pygame.draw, "lines", draw_lines)
    monkeypatch.setattr(pygame.draw, "circle", draw_circle)
    monkeypatch.setattr(WorldSurface, "to_cv2_image", extract_frame)
    frame = _capture_metadrive_topdown_frame(
        env,
        (
            {
                "type": "line",
                "world_points": np.asarray(((0.0, 0.0), (5.0, 0.0))),
                "color": (255, 0, 0),
                "width_px": 3,
            },
            {
                "type": "circle",
                "world_point": np.asarray((5.0, 0.0)),
                "color": (255, 0, 0),
                "radius_px": 5,
            },
        ),
        camera_position=(0.0, 0.0),
        screen_size=64,
        film_size=300,
    )

    assert events == ["line", "circle", "extract"]
    assert frame.shape == (64, 64, 3)
    assert frame.flags["C_CONTIGUOUS"]
    assert env.render_calls == [
        {
            "mode": "top_down",
            "window": False,
            "screen_size": (64, 64),
            "film_size": (300, 300),
            "target_agent_heading_up": False,
            "camera_position": (0.0, 0.0),
        }
    ]


def test_metadrive_topdown_rejects_missing_renderer() -> None:
    class MissingRendererEnv:
        engine = None

        @staticmethod
        def render(**kwargs: object) -> None:
            return None

    with pytest.raises(EvaluationArtifactError, match="top_down_renderer"):
        _capture_metadrive_topdown_frame(
            MissingRendererEnv(),
            (),
            camera_position=(0.0, 0.0),
            screen_size=64,
            film_size=300,
        )
