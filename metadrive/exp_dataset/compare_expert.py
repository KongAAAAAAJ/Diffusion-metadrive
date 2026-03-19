import numpy as np
from metadrive.envs.diffusion_envs.base_multi_env import TopDownStateMultiEnv


def run_episodes(expert_name: str, n_episodes: int = 10, render: bool = True) -> list[float]:
    """
    用指定 expert 在 TopDownStateMultiEnv 中运行 n_episodes 个 episode，
    返回每个 episode 的累计 reward 列表。
    """
    from metadrive.examples.ppo_expert import expert as ppo_expert
    from metadrive.policy.idm_policy import IDMPolicy

    env = TopDownStateMultiEnv()
    env.reset()

    def get_action(agent_id, vehicle, idm_cache):
        if expert_name == "ppo":
            return ppo_expert(vehicle, deterministic=True)
        if agent_id not in idm_cache:
            idm_cache[agent_id] = IDMPolicy(vehicle, random_seed=0)
        return idm_cache[agent_id].act()

    ep_rewards = []
    ep_reward = 0.0
    ep_count = 0
    step = 0
    idm_cache = {}

    env.reset()
    while True:
        actions = {
            aid: get_action(aid, env.agents[aid], idm_cache)
            for aid in env.agents.keys()
        }
        _, r, tm, _, info = env.step(actions)
        if render:
            env.render(
                mode="top_down",
                text={"expert": expert_name, "ep": ep_count + 1, "step": step},
            )
        ep_reward += sum(r.values())
        step += 1

        if tm["__all__"]:
            ep_count += 1
            agent_info = list(info.values())[0] if info else {}
            ep_rewards.append(ep_reward)
            print(
                f"  [{expert_name}] ep={ep_count:2d}  steps={step:4d}  "
                f"reward={ep_reward:7.2f}  "
                f"arrive={agent_info.get('arrive_dest', False)}  "
                f"crash={agent_info.get('crash', False)}  "
                f"oor={agent_info.get('out_of_road', False)}"
            )
            ep_reward = 0.0
            step = 0
            idm_cache.clear()
            if ep_count >= n_episodes:
                break
            env.reset()

    env.close()
    return ep_rewards


if __name__ == "__main__":
    N = 10          # 每种 expert 运行的 episode 数
    RENDER = False   # 是否显示 top-down 渲染

    results = {}
    for expert in ["ppo", "idm"]:
        print(f"\n{'='*50}")
        print(f"  Running expert: {expert.upper()}  ({N} episodes)")
        print(f"{'='*50}")
        rewards = run_episodes(expert, n_episodes=N, render=RENDER)
        results[expert] = rewards

    # ── 汇总对比 ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Comparison Summary ({N} episodes each)")
    print(f"{'='*50}")
    print(f"{'Expert':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Per-episode rewards")
    print(f"{'-'*70}")
    for expert, rewards in results.items():
        arr = np.array(rewards)
        per_ep = "  ".join(f"{v:.1f}" for v in rewards)
        print(
            f"{expert.upper():<8} {arr.mean():>8.2f} {arr.std():>8.2f} "
            f"{arr.min():>8.2f} {arr.max():>8.2f}  [{per_ep}]"
        )
