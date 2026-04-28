from __future__ import annotations

import argparse
import copy
import heapq
import json
import multiprocessing
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from envs.mode_selection_sb3_env import ModeSelectionSB3Env
from models.mode_selection.sb3_mode_cls_policy import export_plan_cls_delta, require_sb3


DEFAULT_CONFIG_PATH = "configs/train/mode_cls_ppo.yaml"
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/mode_cls_ppo")
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def load_config(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def create_next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(match.group(1))
        for item in output_root.iterdir()
        if item.is_dir() and (match := _RUN_DIR_RE.match(item.name))
    ]
    run_dir = output_root / f"run_{max(existing) + 1 if existing else 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


class TopKCheckpointKeeper:
    def __init__(self, k: int):
        self.k = int(k)
        self.heap: list[tuple[float, str]] = []

    def update(self, score: float, path: Path) -> None:
        heapq.heappush(self.heap, (float(score), str(path)))
        if len(self.heap) > self.k:
            _, remove_path = heapq.heappop(self.heap)
            shutil.rmtree(remove_path, ignore_errors=True)


def _extract_trained_plan_cls_branch(model):
    core = getattr(getattr(model, "policy", None), "mode_cls_core", None)
    if core is None:
        raise RuntimeError("SB3 policy does not expose mode_cls_core; cannot export plan_cls_branch delta.")
    return core.plan_cls_branch


def resolve_device(name: str | None) -> torch.device:
    requested = str(name or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_planner(config: Mapping[str, Any], env_config: Mapping[str, Any]):
    from metadrive.policy.diffusion_policy.transfuser_config import (
        build_transfuser_config,
        diffusion_model_config_to_overrides,
        load_diffusion_model_config,
    )
    from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
    from models.platoon.weight_migration import migrate_single_to_platoon

    model_cfg = load_diffusion_model_config(str(config.get("model_config_path", "configs/diffusion/model.yaml")))
    tf_config = build_transfuser_config(
        str(model_cfg.get("model_size", "small")),
        **diffusion_model_config_to_overrides(model_cfg),
    )
    planner = PlatoonDiffusionPlanner(tf_config, num_vehicles=int(config.get("num_agents", 3)))
    pretrained_ckpt = str(config.get("pretrained_ckpt", "") or "")
    if pretrained_ckpt:
        planner = migrate_single_to_platoon(pretrained_ckpt, planner)
    planner.freeze_for_mode_selection().to(resolve_device(str(env_config.get("planner_device", "cpu"))))
    return planner


def run_oracle_diagnostic(env: ModeSelectionSB3Env, run_dir: Path) -> None:
    # Lightweight diagnostic: validates candidate export on reset and records argmax/masks.
    obs, _ = env.reset()
    report = {
        "pretrained_argmax_mode": env._last_export["pretrained_argmax_mode"].tolist(),
        "valid_mode_counts": obs["agent_mode_masks"].sum(axis=1).astype(int).tolist(),
        "candidate_endpoint_sample": env._last_export["trajectory_candidates"][:, :, -1, :2].tolist(),
    }
    (run_dir / "oracle_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_training(config: Mapping[str, Any], output_root: Path, total_timesteps: int) -> Path:
    # Ensure subprocesses (SubprocVecEnv) use spawn, not fork, to avoid CUDA/Panda3D segfaults.
    multiprocessing.set_start_method("spawn", force=True)

    MaskablePPO, _Policy = require_sb3()
    from stable_baselines3.common.callbacks import BaseCallback

    run_dir = create_next_run_dir(output_root)

    # ── env config ──────────────────────────────────────────────────────────
    env_config = dict(config.get("env_config", {}))
    env_config["reward_config"] = dict(config.get("reward_config", {}))
    env_config.setdefault("num_agents", int(config.get("num_agents", 3)))
    env_config.setdefault("num_modes", int(config.get("num_modes", 16)))
    env_config.setdefault("cls_feature_dim", int(config.get("cls_feature_dim", 160)))
    env_config.setdefault("global_state_dim", int(config.get("global_state_dim", 60)))

    # debug logging: disabled by default in production; enable via debug_logging=true
    if config.get("debug_logging", False):
        env_config.setdefault("debug_log_path", str(run_dir / "mode_selection_debug.jsonl"))
    else:
        env_config["debug_log_path"] = ""   # empty → _write_debug_log is a no-op

    # ── device setup ────────────────────────────────────────────────────────
    planner_device_name = str(env_config.get("planner_device", "cpu"))
    ppo_device_name = str(config.get("ppo_device", planner_device_name))
    planner_device = resolve_device(planner_device_name)
    ppo_device = resolve_device(ppo_device_name)
    print(f"[mode_cls_ppo] cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"[mode_cls_ppo] planner_device={planner_device}", flush=True)
    print(f"[mode_cls_ppo] ppo_policy_device={ppo_device}", flush=True)
    if "cuda" in ppo_device_name.lower() and not torch.cuda.is_available():
        print("[mode_cls_ppo] WARNING: ppo_device=cuda requested but CUDA not available, falling back to CPU", flush=True)
        ppo_device = torch.device("cpu")

    # ── trainable cls branch (extracted in main process) ────────────────────
    main_planner = build_planner(config, env_config)
    trainable_plan_cls_branch = copy.deepcopy(
        main_planner.model._trajectory_head.diff_decoder.layers[-1].task_decoder.plan_cls_branch
    )
    for parameter in trainable_plan_cls_branch.parameters():
        parameter.requires_grad_(True)

    # ── build env(s) ────────────────────────────────────────────────────────
    num_envs = int(config.get("num_envs", 1))
    if num_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        # Factory config excludes the main-process planner object (not picklable across
        # processes); each subprocess builds its own frozen planner.
        factory_cfg = {k: v for k, v in env_config.items() if k != "planner"}

        def _make_env_fn(env_id: int):
            def _fn():
                from stable_baselines3.common.monitor import Monitor
                cfg = dict(factory_cfg)
                cfg["seed_offset"] = env_id   # stagger scenario start across envs
                cfg["debug_log_path"] = ""    # always disabled in multi-env mode
                cfg["planner"] = build_planner(config, cfg)
                return Monitor(ModeSelectionSB3Env(cfg))
            return _fn

        env = SubprocVecEnv([_make_env_fn(i) for i in range(num_envs)])
        print(f"[mode_cls_ppo] parallel_envs={num_envs} (SubprocVecEnv)", flush=True)

        # Oracle diagnostic on a temporary single env (SubprocVecEnv can't run it directly)
        tmp_cfg = dict(factory_cfg)
        tmp_cfg["planner"] = main_planner
        tmp_env = ModeSelectionSB3Env(tmp_cfg)
        run_oracle_diagnostic(tmp_env, run_dir)
        tmp_env.close()
    else:
        from stable_baselines3.common.monitor import Monitor
        env_config["planner"] = main_planner
        env = Monitor(ModeSelectionSB3Env(env_config))
        run_oracle_diagnostic(env.env, run_dir)
        print("[mode_cls_ppo] parallel_envs=1 (single env)", flush=True)

    # ── checkpoint callback ──────────────────────────────────────────────────
    keeper = TopKCheckpointKeeper(int(config.get("ckpt_top_k", 3)))
    ckpt_root = run_dir / "checkpoints"
    pretrained_ckpt = str(config.get("pretrained_ckpt", ""))
    ckpt_interval = int(config.get("checkpoint_interval_steps", 20000))

    class TopKModeClsCheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._last_ckpt_step = 0

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps - self._last_ckpt_step < ckpt_interval:
                return
            self._last_ckpt_step = self.num_timesteps
            ep_rewards = [float(item["r"]) for item in getattr(self.model, "ep_info_buffer", []) if "r" in item]
            score = float(np.mean(ep_rewards)) if ep_rewards else float(self.num_timesteps)
            ckpt_dir = ckpt_root / f"step_{self.num_timesteps:08d}_score_{score:.4f}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(ckpt_dir / "sb3_model"))
            export_plan_cls_delta(
                _extract_trained_plan_cls_branch(self.model),
                ckpt_dir / "plan_cls_branch_delta.pt",
                {
                    "pretrained_ckpt": pretrained_ckpt,
                    "num_modes": int(config.get("num_modes", 16)),
                    "score": score,
                    "timesteps": int(self.num_timesteps),
                },
            )
            keeper.update(score, ckpt_dir)
            print(
                f"[mode_cls_ppo] checkpoint saved: {ckpt_dir.name} (score={score:.4f})",
                flush=True,
            )

    # ── model ────────────────────────────────────────────────────────────────
    model = MaskablePPO(
        _Policy,
        env,
        policy_kwargs={
            "plan_cls_branch": trainable_plan_cls_branch,
            "num_agents": int(config.get("num_agents", 3)),
            "num_modes": int(config.get("num_modes", 16)),
            "cls_feature_dim": int(config.get("cls_feature_dim", 160)),
            "global_state_dim": int(config.get("global_state_dim", 60)),
            "lambda_kl": float(config.get("lambda_kl", 0.02)),
        },
        learning_rate=float(config.get("learning_rate", 3e-4)),
        n_steps=int(config.get("n_steps", 32)),
        batch_size=int(config.get("batch_size", 32)),
        n_epochs=int(config.get("n_epochs", 4)),
        gamma=float(config.get("gamma", 0.99)),
        verbose=1,
        device=str(ppo_device),
        tensorboard_log=str(run_dir / "tb"),
    )
    model.learn(total_timesteps=int(total_timesteps), progress_bar=False, callback=TopKModeClsCheckpointCallback())

    # ── final checkpoint (always saved regardless of interval) ───────────────
    ckpt_dir = ckpt_root / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_dir / "sb3_model"))
    export_plan_cls_delta(
        _extract_trained_plan_cls_branch(model),
        ckpt_dir / "plan_cls_branch_delta.pt",
        {
            "pretrained_ckpt": str(config.get("pretrained_ckpt", "")),
            "num_modes": int(config.get("num_modes", 16)),
            "note": "Final SB3 checkpoint; top-k checkpoints are stored in sibling step_* directories.",
        },
    )
    (run_dir / "summary.json").write_text(json.dumps({"total_timesteps": int(total_timesteps)}, indent=2), encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CTDE mode-classification PPO with SB3 MaskablePPO.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pretrained-ckpt", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--total-env-steps", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=0)
    parser.add_argument("--planner-device", default="")
    parser.add_argument("--use-render", type=int, choices=(0, 1), default=-1)
    parser.add_argument("--scenario-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.pretrained_ckpt:
        config["pretrained_ckpt"] = args.pretrained_ckpt
    if args.num_agents > 0:
        config["num_agents"] = int(args.num_agents)
        config.setdefault("env_config", {})["num_agents"] = int(args.num_agents)
    if args.planner_device:
        config.setdefault("env_config", {})["planner_device"] = args.planner_device
    if args.use_render >= 0:
        config.setdefault("env_config", {})["use_render"] = bool(args.use_render)
    if args.scenario_ids:
        scenario_ids = [item.strip() for item in args.scenario_ids.split(",") if item.strip()]
        if scenario_ids:
            config.setdefault("env_config", {})["scenario_ids"] = scenario_ids
    total_steps = int(args.total_env_steps or config.get("total_timesteps", 200))
    run_dir = run_training(config, Path(args.output_root), total_steps)
    print(f"[mode_cls_ppo] outputs: {run_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
