from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from evaluation.preview_and_evaluation import run_scenario
from models.controller.LQRFollowerController import _solve_lqr
from scenarios.definitions import SCENARIO_BY_ID


SCENARIO_IDS = (
    "S5_hard_brake_lead",
    "S6_background_merge_in",
    "S7_ego_merge_from_ramp",
    "S8_ego_exit_to_ramp",
    "S9_narrow_channel_negotiation",
)
Q1_VALUES = (0.5, 1.0, 2.0, 4.0, 8.0)
Q2_VALUES = (0.5, 1.0, 2.0, 4.0)
R_VALUES = (0.05, 0.1, 0.2, 0.5)


def _finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _rms(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float("nan")


def reference_lat_k(q1: float, q2: float, r: float, speed_km_h: float = 30.0, wheelbase_m: float = 3.0) -> list[float]:
    v = max(abs(float(speed_km_h) / 3.6), 0.5)
    L = max(float(wheelbase_m), 1.0)
    A = np.array([[0.0, v], [0.0, 0.0]], dtype=np.float64)
    B = np.array([[0.0], [v / L]], dtype=np.float64)
    Q = np.diag([float(q1), float(q2)])
    R = np.array([[float(r)]], dtype=np.float64)
    return np.asarray(_solve_lqr(A, B, Q, R), dtype=np.float64).reshape(-1).tolist()


def collect_episode_lqr_stats(payload: dict) -> dict:
    lat_errors: list[float] = []
    heading_errors: list[float] = []
    clipped_steers: list[float] = []
    pdms_formation_lat: list[float] = []
    k_values: list[list[float]] = []
    failed = False

    steps = list(payload.get("steps") or [])
    if len(steps) < 20:
        failed = True

    previous_steer_by_agent: dict[str, float] = {}
    delta_steers: list[float] = []
    for step in steps:
        for info in (step.get("info") or {}).values():
            if isinstance(info, dict) and bool(info.get("crash") or info.get("crash_vehicle") or info.get("out_of_road")):
                failed = True
        pdms = step.get("pdms") or {}
        for agent_id, debug in (step.get("control_debug") or {}).items():
            if not isinstance(debug, dict) or debug.get("mode") != "follower_lqr":
                continue
            lat_errors.append(float(debug.get("lat_error", np.nan)))
            heading_errors.append(float(debug.get("heading_error", np.nan)))
            steer = float(debug.get("clipped_steering", np.nan))
            clipped_steers.append(steer)
            prev = previous_steer_by_agent.get(agent_id)
            if prev is not None and np.isfinite(prev) and np.isfinite(steer):
                delta_steers.append(steer - prev)
            previous_steer_by_agent[agent_id] = steer
            k_lat = debug.get("K_lat")
            if isinstance(k_lat, list) and len(k_lat) >= 2:
                k_values.append([float(k_lat[0]), float(k_lat[1])])
            agent_pdms = pdms.get(agent_id, {}) if isinstance(pdms, dict) else {}
            if isinstance(agent_pdms, dict) and agent_pdms.get("formation_lat") is not None:
                pdms_formation_lat.append(float(agent_pdms["formation_lat"]))

    saturation_count = sum(1 for steer in clipped_steers if np.isfinite(steer) and abs(float(steer)) >= 0.999)
    saturation_ratio = float(saturation_count / max(len(clipped_steers), 1))
    rms_lat_error = _rms(lat_errors)
    rms_heading_error = _rms(heading_errors)
    rms_delta_steer = _rms(delta_steers)
    mean_pdms_formation_lat = _finite_mean(pdms_formation_lat)

    failure_penalty = 10.0 if failed else 0.0
    cost = (
        (0.0 if not np.isfinite(rms_lat_error) else rms_lat_error / 1.5)
        + 0.5 * (0.0 if not np.isfinite(rms_heading_error) else rms_heading_error / 0.35)
        + 0.35 * (0.0 if not np.isfinite(rms_delta_steer) else rms_delta_steer / 0.15)
        + 2.0 * saturation_ratio
        + failure_penalty
        - 0.5 * (0.0 if not np.isfinite(mean_pdms_formation_lat) else mean_pdms_formation_lat)
    )

    k_arr = np.asarray(k_values, dtype=np.float64).reshape(-1, 2) if k_values else np.zeros((0, 2), dtype=np.float64)
    return {
        "cost": float(cost),
        "follower_lqr_steps": int(len(lat_errors)),
        "rms_lat_error": float(rms_lat_error) if np.isfinite(rms_lat_error) else None,
        "rms_heading_error": float(rms_heading_error) if np.isfinite(rms_heading_error) else None,
        "rms_delta_steer": float(rms_delta_steer) if np.isfinite(rms_delta_steer) else None,
        "steering_saturation_ratio": saturation_ratio,
        "mean_pdms_formation_lat": float(mean_pdms_formation_lat) if np.isfinite(mean_pdms_formation_lat) else None,
        "failure_penalty": failure_penalty,
        "failed": bool(failed),
        "K_lat_mean": k_arr.mean(axis=0).tolist() if k_arr.size else None,
        "K_lat_min": k_arr.min(axis=0).tolist() if k_arr.size else None,
        "K_lat_max": k_arr.max(axis=0).tolist() if k_arr.size else None,
    }


def aggregate_stats(rows: list[dict]) -> dict:
    keys = [
        "cost",
        "rms_lat_error",
        "rms_heading_error",
        "rms_delta_steer",
        "steering_saturation_ratio",
        "mean_pdms_formation_lat",
        "failure_penalty",
    ]
    result = {f"mean_{key}": _finite_mean(row.get(key) for row in rows) for key in keys}
    result["total_follower_lqr_steps"] = int(sum(int(row.get("follower_lqr_steps", 0) or 0) for row in rows))
    result["failed_runs"] = int(sum(1 for row in rows if row.get("failed")))
    k_values = [row.get("K_lat_mean") for row in rows if row.get("K_lat_mean") is not None]
    if k_values:
        k_arr = np.asarray(k_values, dtype=np.float64)
        result["K_lat_rollout_mean"] = k_arr.mean(axis=0).tolist()
        result["K_lat_rollout_min"] = k_arr.min(axis=0).tolist()
        result["K_lat_rollout_max"] = k_arr.max(axis=0).tolist()
    else:
        result["K_lat_rollout_mean"] = None
        result["K_lat_rollout_min"] = None
        result["K_lat_rollout_max"] = None
    return result


def _combo_id(q1: float, q2: float, r: float) -> str:
    return f"q1_{q1:g}_q2_{q2:g}_r_{r:g}".replace(".", "p")


def _episode_json_path(output_root: Path, scenario_id: str) -> Path:
    return output_root / scenario_id / "metrices" / "episode_0000" / "metrice.json"


def evaluate_combo(q1: float, q2: float, r: float, seeds: list[int], output_root: Path, episodes_per_scenario: int) -> dict:
    runs: list[dict] = []
    for seed in seeds:
        for scenario_id in SCENARIO_IDS:
            route = SCENARIO_BY_ID[scenario_id].allowed_local_routes[0]
            run_root = output_root / _combo_id(q1, q2, r) / f"seed_{seed}" / scenario_id
            run_scenario(
                scenario_id=scenario_id,
                local_route=route,
                num_agents=3,
                num_episodes=int(episodes_per_scenario),
                output_root=run_root,
                heading_up=False,
                traffic_density=0.10,
                start_seed=int(seed),
                fps=10,
                control_policy="adaptive",
                evaluate=True,
                lqr_lat_q1=float(q1),
                lqr_lat_q2=float(q2),
                lqr_lat_r=float(r),
            )
            for episode_idx in range(int(episodes_per_scenario)):
                path = run_root / scenario_id / "metrices" / f"episode_{episode_idx:04d}" / "metrice.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                stats = collect_episode_lqr_stats(payload)
                stats.update({"scenario_id": scenario_id, "route": route, "seed": int(seed), "episode": int(episode_idx)})
                runs.append(stats)
    summary = aggregate_stats(runs)
    return {
        "q1": float(q1),
        "q2": float(q2),
        "r": float(r),
        "reference_K_lat_30kmh": reference_lat_k(q1, q2, r),
        "runs": runs,
        **summary,
    }


def write_reports(results: list[dict], output_root: Path, top_k: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda row: float(row.get("mean_cost", float("inf"))))
    payload = {
        "scenario_ids": list(SCENARIO_IDS),
        "ranked_results": ranked,
        "best": ranked[0] if ranked else None,
    }
    (output_root / "lqr_lateral_sweep_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    columns = [
        "rank",
        "q1",
        "q2",
        "r",
        "mean_cost",
        "mean_rms_lat_error",
        "mean_rms_heading_error",
        "mean_rms_delta_steer",
        "mean_steering_saturation_ratio",
        "mean_mean_pdms_formation_lat",
        "failed_runs",
        "total_follower_lqr_steps",
        "reference_K_lat_30kmh",
        "K_lat_rollout_mean",
    ]
    with (output_root / "lqr_lateral_topk.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(ranked[: int(top_k)], start=1):
            writer.writerow({key: json.dumps(row.get(key)) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in columns if key != "rank"} | {"rank": rank})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune adaptive follower lateral LQR Q/R weights.")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/lqr_lateral_tune"))
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[59])
    parser.add_argument("--refine-seeds", type=int, nargs="+", default=[59, 60, 61])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-combos", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combos = list(itertools.product(Q1_VALUES, Q2_VALUES, R_VALUES))
    if args.max_combos is not None:
        combos = combos[: max(0, int(args.max_combos))]

    coarse_results = [
        evaluate_combo(q1, q2, r, list(args.seeds), Path(args.output_root), int(args.episodes_per_scenario))
        for q1, q2, r in combos
    ]
    top = sorted(coarse_results, key=lambda row: float(row.get("mean_cost", float("inf"))))[: int(args.top_k)]
    refined_results = [
        evaluate_combo(row["q1"], row["q2"], row["r"], list(args.refine_seeds), Path(args.output_root) / "refined", int(args.episodes_per_scenario))
        for row in top
    ]
    final_results = refined_results or coarse_results
    write_reports(final_results, Path(args.output_root), int(args.top_k))

    best = sorted(final_results, key=lambda row: float(row.get("mean_cost", float("inf"))))[0] if final_results else None
    if best is not None:
        print(
            "[lqr_tune] best "
            f"q1={best['q1']} q2={best['q2']} r={best['r']} "
            f"cost={best['mean_cost']:.4f} K={best['reference_K_lat_30kmh']}"
        )
    print(f"[lqr_tune] reports={Path(args.output_root)}")


if __name__ == "__main__":
    main()
