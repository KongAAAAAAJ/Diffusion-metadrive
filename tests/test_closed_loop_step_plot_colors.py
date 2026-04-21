from metadrive.policy.diffusion_policy.mode_visualization import mode_color
from metadrive.policy.diffusion_policy.test_transfuser_policy import _plot_color_for_mode_index


def test_closed_loop_step_plot_uses_preview_mode_colors() -> None:
    mode_names = ["KEEP_LANE_HIGH", "LANE_CHANGE_LEFT_MEDIUM", "EMERGENCY_STOP"]

    assert _plot_color_for_mode_index(0, mode_names) == mode_color("KEEP_LANE_HIGH")
    assert _plot_color_for_mode_index(1, mode_names) == mode_color("LANE_CHANGE_LEFT_MEDIUM")
    assert _plot_color_for_mode_index(2, mode_names) == mode_color("EMERGENCY_STOP")


def test_closed_loop_step_plot_unknown_mode_falls_back_to_stable_palette() -> None:
    assert _plot_color_for_mode_index(3, ["UNKNOWN"]) != _plot_color_for_mode_index(4, ["UNKNOWN"])
