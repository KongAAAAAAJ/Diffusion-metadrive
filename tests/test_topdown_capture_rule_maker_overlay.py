import numpy as np

from models.diffusion.test_transfuser_policy import _capture_topdown_frame_with_overlay


class _FakeFrameCanvas:
    def pos2pix(self, x, y):
        return (float(x), float(y))


class _FakeTopDownRenderer:
    def __init__(self, surface):
        self.position = None
        self._screen_canvas = surface
        self._frame_canvas = _FakeFrameCanvas()

    def _world_to_screen_position(self, point, offset):
        return (float(point[0]) - float(offset[0]), float(point[1]) - float(offset[1]))


class _FakeTopDownEnv:
    def __init__(self, surface):
        self.agents = {}
        self.top_down_renderer = _FakeTopDownRenderer(surface)
        self.engine = None
        self.render_calls = []

    def render(self, **kwargs):
        self.render_calls.append(kwargs)
        return None


def test_capture_topdown_frame_forwards_rule_maker_debug(monkeypatch):
    import pygame
    import tools.topdown_view as topdown_view

    if not pygame.get_init():
        pygame.init()
    surface = pygame.Surface((16, 16))
    surface.fill((1, 2, 3))
    env = _FakeTopDownEnv(surface)
    debug = {
        "formation_locked": False,
        "risk_triggered": True,
        "agent_ids": ["agent0"],
        "best_actions": {"agent0": 0},
    }
    calls = []

    def _fake_overlay(frame, overlay_env, rule_maker_debug):
        calls.append((frame.copy(), overlay_env, rule_maker_debug))
        return frame

    monkeypatch.setattr(topdown_view, "overlay_rule_maker_debug", _fake_overlay)

    frame = _capture_topdown_frame_with_overlay(
        env,
        overlay_polylines=[],
        screen_size=16,
        film_size=16,
        camera_position=(0.0, 0.0),
        rule_maker_debug=debug,
    )

    assert frame is not None
    assert calls
    assert calls[0][1] is env
    assert calls[0][2] is debug


def test_capture_topdown_frame_keeps_rule_maker_debug_optional(monkeypatch):
    import pygame
    import tools.topdown_view as topdown_view

    if not pygame.get_init():
        pygame.init()
    surface = pygame.Surface((16, 16))
    surface.fill((1, 2, 3))
    env = _FakeTopDownEnv(surface)
    calls = []

    def _fake_overlay(frame, overlay_env, rule_maker_debug):
        calls.append(rule_maker_debug)
        return frame

    monkeypatch.setattr(topdown_view, "overlay_rule_maker_debug", _fake_overlay)

    frame = _capture_topdown_frame_with_overlay(
        env,
        overlay_polylines=[],
        screen_size=16,
        film_size=16,
        camera_position=(0.0, 0.0),
    )

    assert isinstance(frame, np.ndarray)
    assert calls == [None]


def test_capture_topdown_frame_returns_opencv_writable_contiguous_array():
    import cv2
    import pygame

    if not pygame.get_init():
        pygame.init()
    surface = pygame.Surface((16, 16))
    surface.fill((1, 2, 3))
    env = _FakeTopDownEnv(surface)
    debug = {
        "formation_locked": False,
        "risk_triggered": True,
        "agent_ids": ["agent0"],
        "best_actions": {"agent0": 0},
    }

    frame = _capture_topdown_frame_with_overlay(
        env,
        overlay_polylines=[],
        screen_size=16,
        film_size=16,
        camera_position=(0.0, 0.0),
        rule_maker_debug=debug,
    )

    assert frame.flags["C_CONTIGUOUS"]
    cv2.putText(frame, "ok", (1, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
