from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if not hasattr(np, "bool8"):  # pragma: no cover - ray 2.4 expects this legacy alias
    np.bool8 = np.bool_

from envs.selector_platoon_env import SelectorPlatoonEnv
from models.selector.rllib_selector_model import IntentSelectorRLlibModel
from train.selector_callbacks import PlatoonFormationCallbacks

try:  # pragma: no cover - optional dependency in local dev
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.rllib.policy.policy import PolicySpec
    from ray.tune.registry import register_env
    from ray.tune.logger import UnifiedLogger

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    ray = None  # type: ignore
    PPOConfig = None  # type: ignore
    ModelCatalog = None  # type: ignore
    PolicySpec = None  # type: ignore
    register_env = None  # type: ignore
    UnifiedLogger = None  # type: ignore
    _RAY_AVAILABLE = False


DEFAULT_CONFIG_PATH = "configs/train/selector.yaml"
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/selector")
DEFAULT_SINGLE_CKPT = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_6/checkpoints/diffusion-epoch=52.ckpt"
ENV_CONFIG_KEYS = {
    "num_agents",
    "traffic_density",
    "num_scenarios",
    "use_hybrid_map",
    "hybrid_map_blocks_config",
    "use_render",
    "observation_mode",
    "horizon",
    "vehicle_config",
    "map",
    "map_config",
    "random_spawn_lane_index",
    "agent_configs",
    "sensors",
    "start_seed",
    "force_seed_spawn_manager",
}
SELECTOR_ENV_KEYS = {
    "pretrained_ckpt",
    "anchor_path",
    "model_size",
    "ego_fut_mode",
    "num_modes",
    "planner_device",
    "reward_config",
    "lambda_local",
    "lambda_team",
    "agent_context_dim",
    "candidate_summary_dim",
    "mode_embedding_dim",
}


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_to_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_custom_metric(result: dict[str, Any], key: str) -> float | None:
    metrics = result.get("custom_metrics", {})
    if not isinstance(metrics, dict):
        return None
    for candidate in (key, f"{key}_mean"):
        value = _as_float(metrics.get(candidate))
        if value is not None:
            return value
    return None


def _extract_learner_stat(result: dict[str, Any], key: str) -> float | None:
    info = result.get("info", {})
    if not isinstance(info, dict):
        return None
    learner = info.get("learner", {})
    if not isinstance(learner, dict):
        return None
    for policy_payload in learner.values():
        if not isinstance(policy_payload, dict):
            continue
        learner_stats = policy_payload.get("learner_stats", {})
        if not isinstance(learner_stats, dict):
            continue
        value = _as_float(learner_stats.get(key))
        if value is not None:
            return value
    return None


def _extract_result_metric(result: dict[str, Any], key: str) -> float | None:
    value = _as_float(result.get(key))
    if value is not None:
        return value
    value = _extract_custom_metric(result, key)
    if value is not None:
        return value
    return _extract_learner_stat(result, key)


def _format_metric(label: str, value: float | None, fmt: str = ".3f") -> str:
    if value is None:
        return f"{label}=n/a"
    return f"{label}={value:{fmt}}"


def build_iteration_log_line(iteration: int, max_iterations: int, result: dict[str, Any]) -> str:
    env_steps = _as_float(result.get("num_env_steps_sampled"))
    if env_steps is None:
        env_steps = _as_float(result.get("timesteps_total"))
    agent_steps = _as_float(result.get("num_agent_steps_sampled"))
    if agent_steps is None:
        agent_steps = _as_float(result.get("agent_timesteps_total"))
    reward_mean = _as_float(result.get("episode_reward_mean"))
    episode_len_mean = _as_float(result.get("episode_len_mean"))
    episodes_total = _as_float(result.get("episodes_total"))
    loss = _extract_learner_stat(result, "total_loss")
    policy_loss = _extract_learner_stat(result, "policy_loss")
    value_loss = _extract_learner_stat(result, "vf_loss")
    entropy = _extract_learner_stat(result, "entropy")
    formation_error = _extract_custom_metric(result, "formation_error_mean")
    crash_rate = _extract_custom_metric(result, "crash_rate")
    team_reward = _extract_custom_metric(result, "team_reward_mean")
    intent_entropy = _extract_custom_metric(result, "intent_entropy")
    env_wait = _as_float(result.get("sampler_perf", {}).get("mean_env_wait_ms")) if isinstance(result.get("sampler_perf", {}), dict) else None

    fields = [
        f"[train][iter {iteration}/{max_iterations}]",
        _format_metric("env_steps", env_steps, ".0f"),
        _format_metric("agent_steps", agent_steps, ".0f"),
        _format_metric("episodes", episodes_total, ".0f"),
        _format_metric("ep_reward", reward_mean),
        _format_metric("ep_len", episode_len_mean, ".1f"),
        _format_metric("loss", loss),
        _format_metric("policy_loss", policy_loss),
        _format_metric("value_loss", value_loss),
        _format_metric("entropy", entropy),
        _format_metric("formation", formation_error),
        _format_metric("crash", crash_rate),
        _format_metric("team_reward", team_reward),
        _format_metric("intent_entropy", intent_entropy),
        _format_metric("env_wait", env_wait),
    ]
    return " ".join(fields)


def build_training_complete_log_line(summary: dict[str, Any], elapsed_seconds: float) -> str:
    checkpoints = summary.get("checkpoints", [])
    checkpoint_count = len(checkpoints) if isinstance(checkpoints, list) else 0
    target_env_steps = summary.get("target_env_steps")
    iterations = summary.get("iterations")
    return (
        "[train][done] "
        f"elapsed={elapsed_seconds:.1f}s "
        f"iterations={iterations} "
        f"target_env_steps={target_env_steps if target_env_steps is not None else 'n/a'} "
        f"checkpoints={checkpoint_count} "
        f"run_dir={summary.get('run_dir', 'n/a')}"
    )


def require_rllib() -> None:
    if not _RAY_AVAILABLE:
        raise RuntimeError(
            "Ray/RLlib is not installed in this environment. "
            "Install ray[rllib]==2.4.0 before running the selector trainer."
        )


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _allocate_run_paths(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    run_dirs = [child for child in root.iterdir() if child.is_dir() and child.name.startswith("run_")]
    run_dir = root / f"run_{len(run_dirs) + 1}"
    return run_dir, run_dir / "checkpoints", run_dir / "tb"


def build_env_config(cfg: dict[str, Any]) -> dict[str, Any]:
    env_cfg = {key: cfg[key] for key in ENV_CONFIG_KEYS | SELECTOR_ENV_KEYS if key in cfg}
    env_cfg.setdefault("pretrained_ckpt", cfg.get("pretrained_ckpt", DEFAULT_SINGLE_CKPT))
    env_cfg.setdefault("num_agents", int(cfg.get("num_agents", 3)))
    env_cfg.setdefault("num_modes", int(cfg.get("num_modes", cfg.get("ego_fut_mode", 8))))
    env_cfg.setdefault("K", int(env_cfg["num_modes"]))
    env_cfg.setdefault("use_render", bool(cfg.get("use_render", False)))
    env_cfg.setdefault("observation_mode", str(cfg.get("observation_mode", "multimodal")))
    env_cfg.setdefault("planner_device", str(cfg.get("planner_device", "cpu")))
    env_cfg.setdefault("reward_config", dict(cfg.get("reward_config", {})))
    env_cfg.setdefault("lambda_local", float(cfg.get("lambda_local", 0.5)))
    env_cfg.setdefault("lambda_team", float(cfg.get("lambda_team", 0.5)))
    env_cfg.setdefault("rllib_reset_compat", False)
    return env_cfg


def build_rllib_config(cfg: dict[str, Any]):
    """
    1. 注册环境和模型
    2. 构建PPO算法配置
    """
    require_rllib()  # 检查RLlib依赖是否可用

    env_cfg = build_env_config(cfg)  # 从训练配置中构建环境配置

    temp_env = SelectorPlatoonEnv(env_cfg)  # 将platoonEnv包装成selector环境实例，适配动作空间、观察空间、奖励等
    observation_space = temp_env.single_observation_space
    action_space = temp_env.single_action_space

    if hasattr(temp_env, "close"):  # 拿到observation_space和action_space后，及时关闭临时环境释放资源
        temp_env.close()

    # 在RLlib中注册环境和模型，构建PPO算法配置
    register_env("selector_platoon_env", lambda config: SelectorPlatoonEnv(config))
    ModelCatalog.register_custom_model("intent_selector_model", IntentSelectorRLlibModel)

    # 定义训练配置，包含环境、模型、优化器、rollout、资源和多智能体设置等
    ppo_cfg = (
        PPOConfig()
        .environment("selector_platoon_env", env_config=env_cfg, disable_env_checking=True)
        .framework("torch")
        .training(
            lr=float(cfg.get("lr", 3e-4)),
            gamma=float(cfg.get("gamma", 0.99)),
            lambda_=float(cfg.get("lambda_gae", 0.95)),
            clip_param=float(cfg.get("clip_param", 0.2)),
            entropy_coeff=float(cfg.get("entropy_coeff", 0.01)),
            vf_loss_coeff=float(cfg.get("vf_loss_coeff", 0.5)),
            train_batch_size=int(cfg.get("train_batch_size", 4000)),
            sgd_minibatch_size=int(cfg.get("sgd_minibatch_size", 256)),
            num_sgd_iter=int(cfg.get("num_sgd_iter", 10)),
            model={
                "custom_model": "intent_selector_model",
                "_disable_preprocessor_api": True,
                "custom_model_config": {
                    "hidden": tuple(cfg.get("hidden", (256, 256))),
                    "critic_obs_key": "global_state",
                },
            },
        )
        .rollouts(
            num_rollout_workers=int(cfg.get("num_rollout_workers", 2)),
            num_envs_per_worker=int(cfg.get("num_envs_per_worker", 1)),
            rollout_fragment_length=cfg.get("rollout_fragment_length", "auto") if cfg.get("rollout_fragment_length") == "auto" else int(cfg.get("rollout_fragment_length", 200)),
            batch_mode=str(cfg.get("batch_mode", "truncate_episodes")),
        )
        .resources(num_gpus=float(cfg.get("num_gpus", 0)))
        .multi_agent(
            policies={
                "shared_selector": PolicySpec(
                    observation_space=observation_space,
                    action_space=action_space,
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_selector",
        )
        .callbacks(PlatoonFormationCallbacks)
    )
    return ppo_cfg, env_cfg


def estimate_env_steps_per_iteration(cfg: dict[str, Any]) -> int:
    """估算 RLlib 每次训练迭代（iteration）中采集的环境步数"""
    rfl = cfg.get("rollout_fragment_length", 200)
    train_batch_size = int(cfg.get("train_batch_size", 4000))
    if str(rfl) == "auto":
        return train_batch_size
    rollout_fragment_length = max(1, int(rfl))
    num_rollout_workers = max(1, int(cfg.get("num_rollout_workers", 2)))
    num_envs_per_worker = max(1, int(cfg.get("num_envs_per_worker", 1)))
    fragment_total = rollout_fragment_length * num_rollout_workers * num_envs_per_worker
    return max(fragment_total, train_batch_size)


def resolve_max_iterations(cfg: dict[str, Any]) -> int:
    """Return max training iterations, derived from total_env_steps / steps_per_iter."""
    total_env_steps = cfg.get("total_env_steps")
    if total_env_steps is None:
        return max(1, int(cfg.get("max_iterations", 500)))
    return max(1, math.ceil(int(total_env_steps) / estimate_env_steps_per_iteration(cfg)))


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    require_rllib()

    train_start = time.perf_counter()

    # 分配输出目录
    output_root = Path(cfg.get("output_root", DEFAULT_OUTPUT_ROOT))
    run_dir, checkpoint_dir, log_dir = _allocate_run_paths(output_root)
    best_checkpoint_dir = run_dir / "best_checkpoint"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ppo_cfg, env_cfg = build_rllib_config(cfg)  # *1.构建RLlib算法配置和环境配置

    def _logger_creator(config):
        return UnifiedLogger(config, str(log_dir), loggers=None)  # 修改log保存地址

    if hasattr(ppo_cfg, "debugging"):
        ppo_cfg = ppo_cfg.debugging(logger_creator=_logger_creator)

    estimated_env_steps_per_iteration = estimate_env_steps_per_iteration(cfg)
    max_iterations = resolve_max_iterations(cfg)
    checkpoint_freq = max(1, int(cfg.get("checkpoint_freq", 20)))
    best_checkpoint_metric = str(cfg.get("best_checkpoint_metric", "episode_reward_mean"))
    best_checkpoint_mode = str(cfg.get("best_checkpoint_mode", "max")).lower()
    last_result: dict[str, Any] | None = None
    checkpoint_paths: list[str] = []
    best_checkpoint_path: str | None = None
    best_checkpoint_value: float | None = None
    best_checkpoint_iteration: int | None = None
    algo = None

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
    try:
        algo = ppo_cfg.build()  # *2.构建算法实例
        for iteration in range(1, max_iterations + 1):
            result = algo.train()  # *3.执行训练迭代，采集数据、更新模型、评估等
            last_result = dict(result)
            print(build_iteration_log_line(iteration, max_iterations, last_result), flush=True)
            metric_value = _extract_result_metric(last_result, best_checkpoint_metric)
            is_better = False
            if metric_value is not None:
                if best_checkpoint_value is None:
                    is_better = True
                elif best_checkpoint_mode == "min":
                    is_better = metric_value < best_checkpoint_value
                else:
                    is_better = metric_value > best_checkpoint_value
            if is_better:
                if best_checkpoint_path is not None:
                    best_path = Path(best_checkpoint_path)
                    if best_path.exists():
                        shutil.rmtree(best_path)
                checkpoint_path = algo.save(str(best_checkpoint_dir))
                best_checkpoint_path = str(checkpoint_path)
                best_checkpoint_value = metric_value
                best_checkpoint_iteration = iteration
            if iteration % checkpoint_freq == 0 or iteration == max_iterations:  # *4.定期保存模型检查点和训练总结
                checkpoint_path = algo.save(str(checkpoint_dir))
                checkpoint_paths.append(str(checkpoint_path))
        summary = {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "best_checkpoint_dir": str(best_checkpoint_dir),
            "log_dir": str(log_dir),
            "env_config": env_cfg,
            "iterations": max_iterations,
            "target_env_steps": cfg.get("total_env_steps"),
            "estimated_env_steps_per_iteration": estimated_env_steps_per_iteration,
            "checkpoints": checkpoint_paths,
            "best_checkpoint_metric": best_checkpoint_metric,
            "best_checkpoint_mode": best_checkpoint_mode,
            "best_checkpoint_path": best_checkpoint_path,
            "best_checkpoint_value": best_checkpoint_value,
            "best_checkpoint_iteration": best_checkpoint_iteration,
            "last_result": last_result or {},
            "elapsed_seconds": time.perf_counter() - train_start,
        }
        summary = _to_json_safe(summary)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(build_training_complete_log_line(summary, float(summary["elapsed_seconds"])), flush=True)
        return summary
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RLlib MAPPO selector for platoon driving.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = _load_config(args.config)
    if args.pretrained_ckpt is not None:
        cfg["pretrained_ckpt"] = args.pretrained_ckpt
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    if args.max_iterations is not None:
        cfg["max_iterations"] = args.max_iterations
    if args.total_env_steps is not None:
        cfg["total_env_steps"] = args.total_env_steps
    if args.checkpoint_freq is not None:
        cfg["checkpoint_freq"] = args.checkpoint_freq

    if args.dry_run:
        env_cfg = build_env_config(cfg)
        print(json.dumps(env_cfg, indent=2, sort_keys=True))
        return env_cfg
    return run_training(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
