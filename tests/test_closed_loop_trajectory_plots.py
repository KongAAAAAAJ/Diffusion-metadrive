import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _install_test_stubs() -> None:
    cv2_module = types.ModuleType("cv2")
    cv2_module.COLOR_RGB2BGR = 0
    cv2_module.cvtColor = lambda image, code: image
    cv2_module.imwrite = lambda path, image: Path(path).write_bytes(b"fake-image") or True
    sys.modules["cv2"] = cv2_module

    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.load = lambda *args, **kwargs: {}
    sys.modules["torch"] = torch_module

    dataset_module = types.ModuleType("metadrive.envs.diffusion_envs.base_multi_env")
    dataset_module.DatasetCollectEnv = object
    sys.modules["metadrive.envs.diffusion_envs.base_multi_env"] = dataset_module

    callback_module = types.ModuleType("metadrive.policy.diffusion_policy.transfuser_callback")
    callback_module.render_closed_loop_prediction = lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8)
    sys.modules["metadrive.policy.diffusion_policy.transfuser_callback"] = callback_module

    config_module = types.ModuleType("metadrive.policy.diffusion_policy.transfuser_config")
    config_module.build_transfuser_config = lambda *args, **kwargs: types.SimpleNamespace(plan_anchor_path="")
    config_module.transfuser_config_to_dict = lambda config: {}
    sys.modules["metadrive.policy.diffusion_policy.transfuser_config"] = config_module

    policy_module = types.ModuleType("metadrive.policy.diffusion_policy.transfuser_policy")
    policy_module.TransfuserPolicy = object
    sys.modules["metadrive.policy.diffusion_policy.transfuser_policy"] = policy_module

    run_dir_module = types.ModuleType("metadrive.policy.diffusion_policy.run_dir_utils")
    run_dir_module.create_numbered_run_dir = lambda path: Path(path) / "run_1"
    sys.modules["metadrive.policy.diffusion_policy.run_dir_utils"] = run_dir_module


def _load_module():
    _install_test_stubs()
    module_path = (
        Path(__file__).resolve().parents[1]
        / "metadrive"
        / "policy"
        / "diffusion_policy"
        / "test_transfuser_policy.py"
    )
    spec = importlib.util.spec_from_file_location("test_transfuser_policy_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_plot_view_bounds_keeps_nearby_road_boundaries_but_excludes_far_ones():
    module = _load_module()
    actual_positions = [np.array([0.0, 0.0]), np.array([10.0, 0.0])]
    planned_trajectories = [(0, [np.array([2.0, 0.5]), np.array([8.0, 0.5])])]
    multimodal_trajectories = [(0, np.array([[[1.0, 1.0], [9.0, 1.0]]], dtype=np.float64), 0)]
    road_boundaries = [
        np.array([[1.0, -2.0], [9.0, -2.0]], dtype=np.float64),
        np.array([[100.0, 100.0], [110.0, 100.0]], dtype=np.float64),
    ]

    xlim, ylim = module._compute_plot_view_bounds(
        actual_positions,
        planned_trajectories,
        multimodal_trajectories,
        road_boundaries=road_boundaries,
        padding=1.0,
        road_margin=2.0,
    )

    assert xlim[0] <= 0.0 and xlim[1] >= 10.0
    assert ylim[0] <= -3.0
    assert ylim[1] < 50.0


def test_build_step_trajectory_plot_path_uses_expected_directory_layout(tmp_path: Path):
    module = _load_module()

    path = module._build_step_trajectory_plot_path(tmp_path, episode_idx=2, step_idx=15)

    assert path == tmp_path / "step_trajectory_plots" / "episode_002" / "step_00015.png"


def test_save_step_trajectory_plot_writes_image(tmp_path: Path):
    module = _load_module()
    output_path = module._build_step_trajectory_plot_path(tmp_path, episode_idx=0, step_idx=1)
    step_record = module.StepTrajectoryPlotRecord(
        step_idx=1,
        ego_position=np.array([0.0, 0.0], dtype=np.float64),
        selected_trajectory=np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        multimodal_trajectories=np.array(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[1.0, 1.0], [2.0, 1.0]],
            ],
            dtype=np.float64,
        ),
        selected_mode_idx=0,
    )
    road_boundaries = [
        np.array([[0.0, -1.0], [3.0, -1.0]], dtype=np.float64),
        np.array([[50.0, 50.0], [60.0, 50.0]], dtype=np.float64),
    ]

    module._save_step_trajectory_plot(
        step_record=step_record,
        road_boundaries=road_boundaries,
        output_path=output_path,
        episode_idx=0,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
