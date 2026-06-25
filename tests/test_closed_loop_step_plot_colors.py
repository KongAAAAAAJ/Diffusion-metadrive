from models.diffusion.mode_visualization import mode_color
from models.diffusion.test_transfuser_policy import _plot_color_for_mode_index


def test_closed_loop_step_plot_uses_preview_mode_colors() -> None:
    mode_names = ["KEEP_HIGH", "LEFT_LC_MEDIUM", "STOP"]

    assert _plot_color_for_mode_index(0, mode_names) == mode_color("KEEP_HIGH")
    assert _plot_color_for_mode_index(1, mode_names) == mode_color("LEFT_LC_MEDIUM")
    assert _plot_color_for_mode_index(2, mode_names) == mode_color("STOP")


def test_closed_loop_step_plot_unknown_mode_falls_back_to_stable_palette() -> None:
    assert _plot_color_for_mode_index(3, ["UNKNOWN"]) != _plot_color_for_mode_index(4, ["UNKNOWN"])
