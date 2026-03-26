from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.platoon_env import PlatoonEnv
from evaluation.reward_terms import compute_team_reward
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon
from train.closedloop_executor import ClosedLoopExecutor, ParallelClosedLoopExecutor
from train.joint_group import build_joint_groups, extract_joint_trajectories, select_top_k_candidates
from train.ma_grpo_trainer import MultiAgentGRPOTrainer

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


DEFAULT_SINGLE_CKPT = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt"
DEFAULT_CONFIG_PATH = "configs/train/platoon_grpo_v2.yaml"
DEFAULT_RL_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion_rl")


class StubTrajectoryHead(nn.Module):
    def __init__(self, ego_fut_mode: int = 3, num_poses: int = 8):
        super().__init__()
        self.ego_fut_mode = ego_fut_mode
        self._num_poses = num_poses
        self.plan_anchor = nn.Parameter(torch.zeros(ego_fut_mode, num_poses, 3), requires_grad=False)

    def norm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def denorm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def bezier_xyyaw(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)


class ToyPlanner(nn.Module):
    def __init__(
        self,
        num_agents: int = 3,
        ego_fut_mode: int = 3,
        init_scale: float = 0.78,
        init_bias: float = 0.18,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.model = nn.Module()
        self.model._trajectory_head = StubTrajectoryHead(ego_fut_mode=ego_fut_mode)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        with torch.no_grad():
            template_xy = self._template_xy(torch.device("cpu"))
            template = torch.cat([template_xy, torch.zeros_like(template_xy[..., :1])], dim=-1)
            self.model._trajectory_head.plan_anchor.copy_(template.unsqueeze(0).repeat(ego_fut_mode, 1, 1))

    def _template_xy(self, device: torch.device) -> torch.Tensor:
        base = torch.linspace(0.0, 7.0, 8, device=device).unsqueeze(-1)
        x = base * self.scale + self.bias
        y = torch.zeros_like(base)
        return torch.cat([x, y], dim=-1)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        outputs = {}
        for agent_id in batch.keys():
            device = batch[agent_id]["status"].device
            xy = self._template_xy(device)
            outputs[agent_id] = torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)
        return outputs

    def extract_rl_context(self, batch: dict):
        contexts = {}
        for agent_id in batch.keys():
            device = batch[agent_id]["status"].device
            contexts[agent_id] = {"_stub": torch.zeros(1, device=device)}
        return contexts, list(batch.keys())

    def predict_denoised_traj(self, noisy_traj_norm: torch.Tensor, timestep: torch.Tensor, context: dict) -> torch.Tensor:
        device = noisy_traj_norm.device
        template = self._template_xy(device)
        template = template.view(1, 1, 8, 2).expand(noisy_traj_norm.shape[0], noisy_traj_norm.shape[1], -1, -1)
        return template


class ToyEnv:
    def __init__(self, num_agents: int, mode: str):
        self.num_agents = num_agents
        self.mode = mode

    def reset(self):
        obs = {}
        for i in range(self.num_agents):
            obs[f"agent{i}"] = {
                "camera": torch.zeros(3, 256, 1024, dtype=torch.float32),
                "lidar": torch.zeros(1, 256, 256, dtype=torch.float32),
                "status": torch.zeros(8, dtype=torch.float32),
                "formation_relation_state": torch.zeros(12, dtype=torch.float32),
            }
        return obs

    def close(self) -> None:
        return None

    def get_state(self) -> dict:
        return {"last_mode": self.mode, "num_agents": self.num_agents}

    def set_state(self, state: dict) -> None:
        del state
        return None

    def get_current_obs(self):
        return self.reset()

    def step(self, actions: dict[str, np.ndarray]):
        obs = self.reset()
        reward = {agent_id: 0.0 for agent_id in obs.keys()}
        terminated = {agent_id: False for agent_id in obs.keys()}
        truncated = {agent_id: False for agent_id in obs.keys()}
        terminated["__all__"] = False
        truncated["__all__"] = False
        info = {agent_id: {"control_mode": "trajectory"} for agent_id in obs.keys()}
        return obs, reward, terminated, truncated, info

    def evaluate_trajectory_group(self, agent_id: str, trajectories: torch.Tensor):
        del agent_id
        step_infos = []
        crash_flags = []
        out_flags = []
        target_progress = 5.9 if self.mode == "toy-single" else 5.7
        crash_threshold = 5.1 if self.mode == "toy-single" else 5.0
        for idx, traj in enumerate(trajectories):
            final_x = float(traj[-1, 0].item())
            lateral_mean = float(np.abs(traj[:, 1]).mean().item())
            formation_error = abs(target_progress - final_x) + 0.2 * lateral_mean
            crash = bool(final_x < crash_threshold or lateral_mean > 0.8)
            min_gap = max(2.0, 12.0 - 1.5 * formation_error)
            info = {
                "progress": final_x / 8.0,
                "formation_error": formation_error,
                "min_gap": min_gap,
                "jerk": 0.02 * idx + 0.01 * lateral_mean,
                "delta_steering": 0.01 * idx + 0.01 * lateral_mean,
                "crash": crash,
                "out_of_road": False,
            }
            step_infos.append([info for _ in range(8)])
            crash_flags.append(crash)
            out_flags.append(False)
        return {"step_infos": step_infos, "crash_flags": crash_flags, "out_of_road_flags": out_flags}


def _build_writer(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    if SummaryWriter is not None:
        return SummaryWriter(log_dir=str(log_dir))
    event_file = log_dir / f"events.out.tfevents.{int(time.time())}"
    event_file.touch()
    return None


def _log_metrics(writer, metrics: dict, step: int):
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            writer.add_scalar(f"train/{key}", float(value), step)


def _resolve_log_dir(mode: str, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    if mode == "platoon-closedloop":
        return Path("logs/platoon_closedloop_rl")
    if mode == "platoon":
        return Path("logs/platoon_rl")
    return Path("logs/toy_single_rl")


def _allocate_rl_run_paths(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    run_dirs = [child for child in root.iterdir() if child.is_dir() and child.name.startswith("run_")]
    run_dir = root / f"run_{len(run_dirs) + 1}"
    return run_dir, run_dir / "checkpoints", run_dir / "tb"


def _resolve_output_dirs(mode: str, requested_checkpoint_dir: str | None, requested_log_dir: str | None) -> tuple[Path, Path]:
    if mode in {"platoon", "platoon-closedloop"} and requested_checkpoint_dir is None and requested_log_dir is None:
        _, checkpoint_dir, log_dir = _allocate_rl_run_paths(DEFAULT_RL_OUTPUT_ROOT)
        return checkpoint_dir, log_dir

    if requested_checkpoint_dir is not None:
        checkpoint_dir = Path(requested_checkpoint_dir)
    else:
        checkpoint_dir = Path("checkpoints/platoon_rl")

    log_dir = _resolve_log_dir(mode, requested_log_dir)
    return checkpoint_dir, log_dir


def _resolve_summary_path(mode: str, num_agents: int) -> Path:
    if mode == "toy-single":
        name = "toy-single_summary.json"
    elif mode == "platoon-closedloop":
        name = "real_platoon_closedloop_summary.json"
    elif num_agents == 1:
        name = "real_single_summary.json"
    else:
        name = "real_platoon_summary.json"
    return Path("outputs/phase5") / name


def _legacy_summary_path(mode: str, num_agents: int) -> Path | None:
    if mode == "toy-single":
        return Path("outputs/phase5/toy-single_summary.json")
    if mode == "platoon-closedloop":
        return Path("outputs/phase5/platoon-closedloop_summary.json")
    if num_agents > 1:
        return Path("outputs/phase5/platoon_summary.json")
    return None


def _save_checkpoint(model: nn.Module, checkpoint_dir: Path, step: int, metrics: dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        checkpoint_dir / f"step_{step}.ckpt",
    )


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing_enable"):
            try:
                module.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(module, "gradient_checkpointing"):
            try:
                setattr(module, "gradient_checkpointing", True)
            except Exception:
                pass


def _apply_freeze_config(model: nn.Module, config: dict):
    freeze_backbone = bool(config.get("freeze_backbone", True))
    freeze_tf_decoder = bool(config.get("freeze_tf_decoder", False))
    freeze_trajectory_head = bool(config.get("freeze_trajectory_head", False))

    backbone_keywords = ["_backbone", "_bev_upscale", "_bev_downscale", "_keyval_embedding", "_keyval_"]
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        should_freeze = False
        if freeze_backbone and any(keyword in name for keyword in backbone_keywords):
            should_freeze = True
        if freeze_tf_decoder and "_tf_decoder" in name:
            should_freeze = True
        if freeze_trajectory_head and "_trajectory_head" in name:
            should_freeze = True
        if "relation_encoder" in name or "_status_encoding" in name:
            should_freeze = False

        param.requires_grad_(not should_freeze)
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    if freeze_backbone:
        frozen_bn_count = 0
        for name, module in model.named_modules():
            if any(keyword in name for keyword in backbone_keywords):
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
                    module.track_running_stats = False
                    frozen_bn_count += 1
        print(f"[freeze] Froze {frozen_bn_count} BatchNorm layers in backbone")

    ratio = 100.0 * trainable / max(total, 1)
    print(f"[freeze] Total params: {total:,}, Trainable: {trainable:,} ({ratio:.1f}%)")


def _adapt_learning_rate(optimizer: torch.optim.Optimizer, metrics: dict, config: dict) -> float:
    kl = float(metrics["kl"])
    kl_target = float(config.get("kl_target", 8.0))
    kl_high = kl_target * 1.5
    kl_low = kl_target * 0.5
    lr_max = float(config.get("lr", 5e-5))
    lr_min = 1e-6
    lr_decay = float(config.get("lr_decay_factor", 0.8))
    lr_grow = float(config.get("lr_grow_factor", 1.05))
    latest_lr = lr_min
    for group in optimizer.param_groups:
        current_lr = float(group["lr"])
        if kl > kl_high:
            group["lr"] = max(current_lr * lr_decay, lr_min)
        elif kl < kl_low:
            group["lr"] = min(current_lr * lr_grow, lr_max)
        latest_lr = float(group["lr"])
    return latest_lr


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", type=str, choices=["toy-single", "platoon", "platoon-closedloop"], default="toy-single")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_SINGLE_CKPT)
    parser.add_argument("--num-agents", type=int, default=None)
    return parser.parse_args(argv)


def _build_platoon_env(config: dict, render: bool, num_agents: int):
    return PlatoonEnv(
        {
            "observation_mode": "multimodal",
            "use_render": bool(render),
            "num_agents": num_agents,
            "horizon": int(config.get("horizon", 100)),
            "traffic_density": float(config.get("traffic_density", 0.04)),
            "num_scenarios": int(config.get("num_scenarios", 1)),
            "use_hybrid_map": bool(config.get("use_hybrid_map", True)),
            "hybrid_map_sequence": str(config.get("hybrid_map_sequence", "SSXCOCSS")),
        }
    )


def _build_joint_training_inputs(trainer: MultiAgentGRPOTrainer, rollouts: dict):
    """在闭环训练模式下，基于每个智能体采样得到的多 anchor 轨迹，自动组合多组联合轨迹，并用闭环执行器评估每组轨迹的团队奖励"""

    if not bool(trainer.config.get("use_closedloop", False)) or not hasattr(trainer.env, "get_state"):
        return None, None, {}

    # 选出每个智能体的 top-k 候选轨迹
    per_agent_candidates = {}
    top_k = int(trainer.config.get("joint_top_k", 2))
    num_joint_groups = int(trainer.config.get("num_joint_groups", 8))
    for agent_id in rollouts["agent_ids"]:
        per_agent_candidates[agent_id] = select_top_k_candidates(
            rollouts[agent_id]["reward_per_anchor"],
            rollouts[agent_id]["crash_per_anchor"],
            top_k=top_k,
        )

    # 构建多组联合轨迹，并通过闭环执行评估每组轨迹的团队奖励
    joint_groups = build_joint_groups(per_agent_candidates, num_groups=num_joint_groups)
    joint_trajectories = [extract_joint_trajectories(group, rollouts) for group in joint_groups]

    num_closedloop_workers = int(trainer.config.get("closedloop_workers", 0))
    if num_closedloop_workers > 1:
        executor = getattr(trainer, "_parallel_executor", None)
        if executor is None:
            executor = ParallelClosedLoopExecutor(
                env=trainer.env,
                env_config=trainer.config.get("_env_config", {}),
                reward_config=trainer.reward_config,
                num_workers=num_closedloop_workers,
            )
            trainer._parallel_executor = executor
    else:
        executor = ClosedLoopExecutor(trainer.env, trainer.reward_config)

    exec_results = executor.execute_joint_groups(joint_trajectories)
    team_rewards = [compute_team_reward(result["step_infos"], trainer.reward_config) for result in exec_results]
    return joint_groups, team_rewards, dict(getattr(executor, "last_profile", {}) or {})


def _series(metrics_trace: list[dict], key: str, default: float = 0.0) -> list[float]:
    values = []
    for metrics in metrics_trace:
        value = metrics.get(key, default)
        values.append(float(value))
    return values


PROFILE_KEYS = [
    "profile_step_total",
    "profile_collect",
    "profile_joint",
    "profile_joint_restore_reset",
    "profile_joint_restore_set_state",
    "profile_joint_restore_total",
    "profile_joint_step_execution",
    "profile_update",
    "profile_env_step",
    "profile_logging",
    "profile_checkpoint",
]


def _profile_stamp() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _summarize_profile(metrics_trace: list[dict]) -> tuple[dict, list[str]]:
    profile_summary: dict[str, float] = {}
    recommendations: list[str] = []
    if not metrics_trace:
        return profile_summary, recommendations

    key_aliases = {
        "profile_step_total": "step_total",
        "profile_collect": "collect",
        "profile_joint": "joint",
        "profile_joint_restore_reset": "joint_restore_reset",
        "profile_joint_restore_set_state": "joint_restore_set_state",
        "profile_joint_restore_total": "joint_restore_total",
        "profile_joint_step_execution": "joint_step_execution",
        "profile_update": "update",
        "profile_env_step": "env_step",
        "profile_logging": "logging",
        "profile_checkpoint": "checkpoint",
    }

    per_key_values: dict[str, list[float]] = {}
    for key in PROFILE_KEYS:
        values = _series(metrics_trace, key)
        per_key_values[key] = values
        alias = key_aliases[key]
        profile_summary[f"{alias}_mean"] = float(sum(values) / max(len(values), 1))
        profile_summary[f"{alias}_max"] = float(max(values))
        profile_summary[f"{alias}_total"] = float(sum(values))

    step_mean = profile_summary.get("step_total_mean", 0.0)
    stage_means = {
        key_aliases[key]: profile_summary[f"{key_aliases[key]}_mean"]
        for key in PROFILE_KEYS
        if key != "profile_step_total"
    }
    dominant_stage = max(stage_means, key=stage_means.get) if stage_means else "collect"
    dominant_mean = stage_means.get(dominant_stage, 0.0)
    dominant_ratio = dominant_mean / max(step_mean, 1e-8)
    profile_summary["dominant_stage_mean"] = float(dominant_mean)
    profile_summary["dominant_stage_ratio"] = float(dominant_ratio)

    if dominant_stage == "joint":
        recommendations.append("Closed-loop joint evaluation dominates runtime; reduce num_joint_groups, joint_top_k, or closed-loop horizon first.")
    if profile_summary.get("joint_restore_reset_mean", 0.0) > profile_summary.get("joint_step_execution_mean", 0.0):
        recommendations.append("env.reset() inside closed-loop state restore is more expensive than rolling out the joint trajectory; prioritize removing full reset() from candidate evaluation.")
    if dominant_stage == "update":
        recommendations.append("Model update dominates runtime; consider reducing group_size, ddim_steps, or enabling more aggressive parameter freezing.")
    if dominant_stage == "collect":
        recommendations.append("Trajectory sampling dominates runtime; consider fewer anchors/groups, fewer DDIM steps, or a lighter planner backbone.")
    if dominant_stage == "env_step":
        recommendations.append("Environment stepping dominates runtime; consider lighter scenarios, fewer agents, or reducing simulator-side rendering/sensor overhead.")
    if profile_summary.get("logging_mean", 0.0) > 0.05:
        recommendations.append("Logging overhead is measurable; disable verbose profiling/logging outside one-off performance analysis runs.")
    if profile_summary.get("checkpoint_mean", 0.0) > 0.1:
        recommendations.append("Checkpoint writing is noticeable; increase checkpoint_interval during profiling or long experiments.")
    if not recommendations:
        recommendations.append("No single stage is overwhelmingly dominant; optimize the largest mean stage first and then re-profile.")
    return profile_summary, recommendations


def _print_profile_step(step_idx: int, total_steps: int, metrics: dict) -> None:
    print(
        f"[profile][step {step_idx + 1}/{total_steps}] "
        f"total={metrics['profile_step_total']:.3f}s "
        f"collect={metrics['profile_collect']:.3f}s "
        f"joint={metrics['profile_joint']:.3f}s "
        f"joint_reset={metrics['profile_joint_restore_reset']:.3f}s "
        f"joint_set_state={metrics['profile_joint_restore_set_state']:.3f}s "
        f"joint_step_exec={metrics['profile_joint_step_execution']:.3f}s "
        f"update={metrics['profile_update']:.3f}s "
        f"env={metrics['profile_env_step']:.3f}s "
        f"log={metrics['profile_logging']:.3f}s "
        f"ckpt={metrics['profile_checkpoint']:.3f}s"
    )


def _print_profile_summary(profile_summary: dict, recommendations: list[str]) -> None:
    if not profile_summary:
        return
    print(
        "[profile][summary] "
        f"step_mean={profile_summary['step_total_mean']:.3f}s "
        f"collect_mean={profile_summary['collect_mean']:.3f}s "
        f"joint_mean={profile_summary['joint_mean']:.3f}s "
        f"joint_reset_mean={profile_summary['joint_restore_reset_mean']:.3f}s "
        f"joint_set_state_mean={profile_summary['joint_restore_set_state_mean']:.3f}s "
        f"joint_step_exec_mean={profile_summary['joint_step_execution_mean']:.3f}s "
        f"update_mean={profile_summary['update_mean']:.3f}s "
        f"env_mean={profile_summary['env_step_mean']:.3f}s "
        f"log_mean={profile_summary['logging_mean']:.3f}s "
        f"ckpt_mean={profile_summary['checkpoint_mean']:.3f}s "
        f"dominant={max(profile_summary.get('dominant_stage_ratio', 0.0) * 100.0, 0.0):.1f}%"
    )
    for recommendation in recommendations:
        print(f"[profile][suggestion] {recommendation}")


def build_runtime(
    mode: str,
    config_path: str,
    steps: int,
    render: bool,
    checkpoint_dir: str,
    log_dir: str | None,
    ckpt_path: str,
    num_agents: int | None = None,
    run_training: bool = False,
):
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    if num_agents is not None:
        resolved_num_agents = int(num_agents)
    elif mode == "toy-single":
        resolved_num_agents = 1
    else:
        resolved_num_agents = int(config.get("num_agents", 3))

    if mode == "toy-single":
        config = {
            **config,
            "lr": float(config.get("lr", 5e-5)),
            "kl_target": float(config.get("kl_target", 8.0)),
            "ddim_eta": float(config.get("ddim_eta", 0.02)),
            "group_size": min(int(config.get("group_size", 4)), 2),
            "max_grad_norm": float(config.get("max_grad_norm", 10.0)),
            "lr_decay_factor": float(config.get("lr_decay_factor", 0.8)),
            "lr_grow_factor": float(config.get("lr_grow_factor", 1.05)),
            "use_closedloop": False,
        }
        env = ToyEnv(num_agents=resolved_num_agents, mode=mode)
        model = ToyPlanner(num_agents=resolved_num_agents, init_scale=0.78, init_bias=0.18)
        _apply_freeze_config(model, config)
        ref_model = copy.deepcopy(model)
    elif mode in {"platoon", "platoon-closedloop"}:
        config = {
            **config,
            "lr": float(config.get("lr", 5e-5)),
            "kl_target": float(config.get("kl_target", 8.0)),
            "ddim_eta": float(config.get("ddim_eta", 0.02)),
            "max_grad_norm": float(config.get("max_grad_norm", 10.0)),
            "group_size": min(int(config.get("group_size", 4)), 2),
            "lr_decay_factor": float(config.get("lr_decay_factor", 0.8)),
            "lr_grow_factor": float(config.get("lr_grow_factor", 1.05)),
            "max_env_steps_per_rollout": int(config.get("max_env_steps_per_rollout", 20)),
            "use_closedloop": bool(mode == "platoon-closedloop"),
            "closedloop_workers": int(config.get("closedloop_workers", 0)),
        }
        config["_env_config"] = {
            "observation_mode": "multimodal",
            "use_render": False,
            "num_agents": resolved_num_agents,
            "horizon": int(config.get("horizon", 100)),
            "traffic_density": float(config.get("traffic_density", 0.04)),
            "num_scenarios": int(config.get("num_scenarios", 1)),
            "use_hybrid_map": bool(config.get("use_hybrid_map", True)),
            "hybrid_map_sequence": str(config.get("hybrid_map_sequence", "SSXCOCSS")),
        }
        env = _build_platoon_env(config, render=render, num_agents=resolved_num_agents)
        anchor_path = Path("metadrive/exp_dataset/metadrive_anchors_ppo.npy")
        if not anchor_path.exists():
            anchor_path = Path("metadrive/exp_dataset/metadrive_anchors.npy")
        config_tf = build_transfuser_config("small", plan_anchor_path=str(anchor_path))

        # 将单车预训练权重迁移到编队模型中
        model = PlatoonDiffusionPlanner(config_tf, num_vehicles=resolved_num_agents)
        model = migrate_single_to_platoon(str(ckpt_path), model)

        # 冻结感知backbone
        _apply_freeze_config(model, config)

        # 为模型及其子模块启用梯度检查点，以节省显存
        _enable_gradient_checkpointing(model)

        # 复制一份，作为RL训练中的参考模型（不参与更新，仅用于计算KL散度）
        ref_model = copy.deepcopy(model)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ref_model = ref_model.to(device)

    config["total_steps"] = int(steps or config.get("total_steps", 5))
    trainer = MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=env, config=config)
    resolved_checkpoint_dir, resolved_log_dir = _resolve_output_dirs(mode, checkpoint_dir, log_dir)
    summary_path = _resolve_summary_path(mode, resolved_num_agents)
    legacy_summary_path = _legacy_summary_path(mode, resolved_num_agents)

    resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "config": config,
        "trainer": trainer,
        "env": env,
        "model": model,
        "ref_model": ref_model,
        "steps": int(steps or config.get("total_steps", 5)),
        "mode": mode,
        "num_agents": resolved_num_agents,
        "log_dir": resolved_log_dir,
        "checkpoint_dir": resolved_checkpoint_dir,
        "summary_path": summary_path,
        "legacy_summary_path": legacy_summary_path,
    }
    if not run_training:
        return runtime

    writer = _build_writer(resolved_log_dir)
    metrics_trace = []
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_peak_gb = 0.0
    has_oom = False
    obs = trainer.env.reset()
    trainer.current_obs = obs
    trainer.env_step_counter = 0
    if bool(config.get("use_closedloop", False)) and int(config.get("closedloop_workers", 0)) > 1:
        parallel_executor = getattr(trainer, "_parallel_executor", None)
        if parallel_executor is None:
            parallel_executor = ParallelClosedLoopExecutor(
                env=trainer.env,
                env_config=trainer.config.get("_env_config", {}),
                reward_config=trainer.reward_config,
                num_workers=int(config.get("closedloop_workers", 0)),
            )
            trainer._parallel_executor = parallel_executor
        parallel_executor.start()
    try:
        for step_idx in range(runtime["steps"]):
            profile_metrics = {key: 0.0 for key in PROFILE_KEYS}
            step_start = _profile_stamp()
            try:
                collect_start = step_start
                
                # 轨迹开环评估
                rollouts = trainer.collect_group_samples(group_size=int(config.get("group_size", 4)), obs=obs)
                profile_metrics["profile_collect"] = _profile_stamp() - collect_start

                joint_start = _profile_stamp()
                joint_groups, team_rewards, joint_profile = _build_joint_training_inputs(trainer, rollouts)
                profile_metrics["profile_joint"] = _profile_stamp() - joint_start
                profile_metrics["profile_joint_restore_reset"] = float(joint_profile.get("all_restore_reset", 0.0))
                profile_metrics["profile_joint_restore_set_state"] = float(joint_profile.get("all_restore_set_state", 0.0))
                profile_metrics["profile_joint_restore_total"] = float(joint_profile.get("all_restore_total", 0.0))
                profile_metrics["profile_joint_step_execution"] = float(joint_profile.get("all_step_execution", 0.0))

                update_start = _profile_stamp()
                metrics = trainer.update(rollouts, joint_groups=joint_groups, team_rewards=team_rewards)
                profile_metrics["profile_update"] = _profile_stamp() - update_start

                env_start = _profile_stamp()
                obs = trainer.step_env_with_best(rollouts)
                profile_metrics["profile_env_step"] = _profile_stamp() - env_start
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    has_oom = True
                raise

            metrics.update(profile_metrics)

            logging_start = _profile_stamp()
            _log_metrics(writer, metrics, step_idx)
            profile_metrics["profile_logging"] = _profile_stamp() - logging_start

            metrics["lr"] = _adapt_learning_rate(trainer.optimizer, metrics, config)

            if torch.cuda.is_available():
                gpu_peak_gb = max(gpu_peak_gb, torch.cuda.max_memory_allocated() / (1024 ** 3))
            interval = int(config.get("checkpoint_interval", 100))
            checkpoint_start = _profile_stamp()
            if mode in {"platoon", "platoon-closedloop"} and interval > 0 and (step_idx + 1) % interval == 0:
                _save_checkpoint(model, resolved_checkpoint_dir, step_idx + 1, metrics)
            profile_metrics["profile_checkpoint"] = _profile_stamp() - checkpoint_start

            metrics.update(profile_metrics)
            metrics["profile_step_total"] = _profile_stamp() - step_start
            metrics_trace.append(metrics)
            _print_profile_step(step_idx, runtime["steps"], metrics)
        profile_summary, profile_recommendations = _summarize_profile(metrics_trace)
        _print_profile_summary(profile_summary, profile_recommendations)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        parallel_executor = getattr(trainer, "_parallel_executor", None)
        if parallel_executor is not None:
            parallel_executor.stop()
            trainer._parallel_executor = None
        env.close()

    profile_summary, profile_recommendations = _summarize_profile(metrics_trace)
    summary = {
        "mode": mode,
        "num_agents": resolved_num_agents,
        "steps": runtime["steps"],
        "loss": _series(metrics_trace, "loss"),
        "rl_loss": _series(metrics_trace, "rl_loss"),
        "ref_reg_loss": _series(metrics_trace, "ref_reg_loss"),
        "beta_reg": _series(metrics_trace, "beta_reg"),
        "kl": _series(metrics_trace, "kl"),
        "mean_reward": _series(metrics_trace, "mean_reward"),
        "formation_error": _series(metrics_trace, "formation_error"),
        "collision_rate": _series(metrics_trace, "collision_rate"),
        "grad_norm": _series(metrics_trace, "grad_norm"),
        "team_reward_mean": _series(metrics_trace, "team_reward_mean"),
        "lambda_local": _series(metrics_trace, "lambda_local", default=float(config.get("lambda_local", 0.7))),
        "lambda_team": _series(metrics_trace, "lambda_team", default=float(config.get("lambda_team", 0.3))),
        "profile_step_total": _series(metrics_trace, "profile_step_total"),
        "profile_collect": _series(metrics_trace, "profile_collect"),
        "profile_joint": _series(metrics_trace, "profile_joint"),
        "profile_joint_restore_reset": _series(metrics_trace, "profile_joint_restore_reset"),
        "profile_joint_restore_set_state": _series(metrics_trace, "profile_joint_restore_set_state"),
        "profile_joint_restore_total": _series(metrics_trace, "profile_joint_restore_total"),
        "profile_joint_step_execution": _series(metrics_trace, "profile_joint_step_execution"),
        "profile_update": _series(metrics_trace, "profile_update"),
        "profile_env_step": _series(metrics_trace, "profile_env_step"),
        "profile_logging": _series(metrics_trace, "profile_logging"),
        "profile_checkpoint": _series(metrics_trace, "profile_checkpoint"),
        "profile_summary": profile_summary,
        "profile_recommendations": profile_recommendations,
        "gpu_peak_gb": float(gpu_peak_gb),
        "has_nan_loss": any(not torch.isfinite(torch.tensor(m["loss"])) for m in metrics_trace),
        "has_oom": bool(has_oom),
        "log_dir": str(resolved_log_dir),
        "checkpoint_dir": str(resolved_checkpoint_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if legacy_summary_path is not None and legacy_summary_path != summary_path:
        legacy_summary_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    args = parse_args(argv)
    return build_runtime(
        mode=args.mode,
        config_path=args.config,
        steps=args.steps,
        render=bool(args.render),
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        ckpt_path=args.ckpt,
        num_agents=args.num_agents,
        run_training=True,
    )


if __name__ == "__main__":
    main()
