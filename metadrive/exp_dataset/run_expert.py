from metadrive.envs.diffusion_envs.base_multi_env import TopDownStateMultiEnv


if __name__ == "__main__":
    from metadrive.examples.ppo_expert import expert as ppo_expert

    env = TopDownStateMultiEnv()
    o, _ = env.reset()
    print("Agents:", list(o.keys()))
    obs_sample = list(o.values())[0]
    if isinstance(obs_sample, dict):
        print("Obs keys:", list(obs_sample.keys()))
        for k, v in obs_sample.items():
            print(f"  '{k}' shape: {v.shape}")
    else:
        print("Obs shape:", obs_sample.shape)

    # from metadrive.envs.diffusion_envs.utils import show_map
    # show_map(env)

    ep_reward = 0.0
    ep_count = 0
    step = 0
    while True:
        actions = {
            agent_id: ppo_expert(env.agents[agent_id], deterministic=True)
            for agent_id in env.agents.keys()
        }
        o, r, tm, tc, info = env.step(actions)
        env.render(mode="top_down", text={"step": step, "ep": ep_count})
        ep_reward += sum(r.values())
        step += 1

        if tm["__all__"]:
            ep_count += 1
            agent_info = list(info.values())[0] if info else {}
            print(
                f"[Episode {ep_count}] steps={step}, "
                f"reward={ep_reward:.2f}, "
                f"arrive_dest={agent_info.get('arrive_dest', False)}, "
                f"crash={agent_info.get('crash', False)}, "
                f"out_of_road={agent_info.get('out_of_road', False)}"
            )
            ep_reward = 0.0
            step = 0
            if ep_count >= 10:
                break
            o, _ = env.reset()

    env.close()
