# 切换工作目录到脚本所在目录
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.chdir(str(ROOT))

import numpy as np
import os
from scipy.linalg import solve_discrete_are
from highway_env.utils import lon_ctrl_visualize
from algorithms.base_policy import BaseController



class LQRPlatoonController(BaseController):
    """
    Distributed LQR for platoon longitudinal control.
    Each agent uses state-feedback u = -K x for spacing and velocity error dynamics.
    """
    def __init__(self, Q=None, R=None, dt=0.05, Ts=0.1):
        super().__init__()
        self.dt = dt
        self.Ts = Ts  # engine lag
        # Discretize model matrices for error dynamics x = [s_err; v_err; a]
        self.A = np.array([[1, dt, 0],
                           [0,  1, dt],
                           [0,  0, 1 - dt/self.Ts]])
        self.B = np.array([[0], [0], [dt / self.Ts]])
        # Cost matrices
        self.Q = Q if Q is not None else np.diag([1.0, 0.1, 0.01])
        self.R = R if R is not None else np.array([[1]])
        # Solve discrete-time algebraic Riccati
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        # LQR gain
        self.K = np.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)

    def act(self, env, obs):
        actions = []
        # unpack obs
        _obs = obs[0]
        s_errs, v_errs, a_errs = self.unpack_obs(_obs)
        for idx, veh in enumerate(env.controlled_vehicles):
            front, _ = env.road.neighbour_vehicles(veh)
            x = np.array([s_errs[idx], v_errs[idx], a_errs[idx]])
            # LQR control
            u = - (self.K @ x).item()
            u = np.clip(u, self.u_min, self.u_max)
            actions.append(u)
        return np.array(actions)


# Example interaction
if __name__ == '__main__':
    import highway_env
    import time
    import gymnasium as gym

    np.random.seed(1)

    config = {
        "env_idx": 6,
        "duration": 25,
        "observation": {
            "type": "MultiAgentObservation",  # 多智能体观测 tuple
            "observation_config": {
                "type": "OneHot_LonCtrlObservation",
                "use_comm_delay": False,  # 是否考虑通信延时
                "max_time_delay": 1.5,  # [s]
                "normalize": True,  # 当使用render_opt()进行测试时，关闭obs_normalization
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "LonCtrlContinuousAction",  # 编队纵向加速度
                "normalize": False,  # 使用归一化的action: [-1, 1]，不归一化[-5, 1]
            }
        },
        "simulation_frequency": 10,
        "policy_frequency": 10,
        "controlled_vehicles": 3,
    }
    env = gym.make(id="highway-marl-ctrl-v0", render_mode="rgb_array", config=config)
    option = {'env_type': 'train_env'}
    obs, info = env.reset(options=option)

    policy = LQRPlatoonController()

    done = False
    k = 0
    tt = []
    while not done:
        # t1 = time.time()
        actions = policy.act(env=env.env.env, obs=obs)
        # t2 = time.time()
        # tt.append(t2 - t1)
        # print(f"t = {t2 - t1}s")

        obs, reward, terminated, truncated, info = env.step(action=actions)
        k += 1
        env.render()
        done = terminated or truncated
    print(f"average running time: {np.array(tt).mean()}")  # 0.010s
    print(f"k = {k}")

    # 绘图
    # figs, axes = plt.subplots(4, 1)
    #
    # lines = ['ax_des', 'ax', 'vx', 'x']
    # for i, line in enumerate(lines):
    #     axes[i].plot(info["v_mes"][f"{line}"][:k, 0], label="car 1")
    #     axes[i].plot(info["v_mes"][f"{line}"][:k, 1], label="car 2")
    #     axes[i].plot(info["v_mes"][f"{line}"][:k, 2], label="car 3")
    #     axes[i].plot(info["v_mes"][f"{line}"][:k, 3], label="car front")
    #     axes[i].set_title(f"{line}")
    #     axes[i].set_xlabel("Time step")
    #     axes[i].set_ylabel("Value")
    #     axes[i].legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    fig_path = "F:\\IOTJ_Data_and_Plot\\results\\lqr"
    os.makedirs(fig_path, exist_ok=True)
    lon_ctrl_visualize(info, fig_path)
