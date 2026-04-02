# Phase 6: MAPPO Intent Selector for Platoon Formation

## Background

The Phase 5 GRPO approach directly fine-tuned the diffusion model with RL gradients,
which failed because:
- Action space was high-dimensional (full DDIM denoising chain)
- Platoon reward signal corrupted the single-vehicle pretrained distribution
- Non-stationarity: other agents' policies change during training, breaking Markov assumption for independent learners
- Credit assignment was unclear: joint formation reward could not be attributed to individual denoising steps

## New Architecture

```
PlatoonEnv (raw obs: camera, lidar, status, formation_state)
    │
    ▼
Pretrained DiffusionPlanner  ← frozen, per agent
    │  generates K candidate trajectories  (K, 8, 3)
    ▼
SelectorPlatoonEnv (wrapper)
    │  obs = { formation_state(12), traj_candidates(K,8,3) }
    │  action = intent index  i* ∈ {0, …, K-1}
    ▼
IntentSelector  ← MAPPO actor, shared weights across agents
    │  Centralized Critic sees joint obs + joint actions during training
    ▼
Execute trajectory[i*] in PlatoonEnv
    │
    ▼
Reward: formation quality + safety
```

Key properties:
- Pretrained model is **completely frozen** — its distribution is never corrupted
- Policy gradient acts only on the lightweight IntentSelector
- Centralized Critic (MAPPO) resolves credit assignment across agents
- Closed-loop training through RLlib rollout workers calling PlatoonEnv

---

## Implementation Plan

### Task 1 — DiffusionCandidateGenerator

**File**: `models/selector/candidate_generator.py`

Wrap the frozen pretrained model to expose a simple `generate(obs) -> (K, 8, 3)` interface.

```python
class DiffusionCandidateGenerator:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.model = load_pretrained_planner(checkpoint_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.anchors = np.load("metadrive/exp_dataset/metadrive_anchors.npy")  # (K, 8, 2)

    @torch.no_grad()
    def generate(self, obs: dict) -> np.ndarray:
        """
        Returns K candidate trajectories, one per semantic anchor.
        Shape: (K, 8, 3)  — 8 timesteps × (steering, accel, jerk)
        """
        features = obs_to_features(obs)
        trajs = []
        for anchor in self.anchors:
            traj = self.model.plan_from_anchor(features, anchor)  # (8, 3)
            trajs.append(traj)
        return np.stack(trajs)  # (K, 8, 3)
```

**Acceptance criteria**:
- `generate()` returns shape `(K, 8, 3)` with K = number of anchors (currently 9)
- Inference time < 100 ms on CPU for K=9 with 4 DDIM steps
- Weights are frozen: no gradient flows through this module

---

### Task 2 — SelectorPlatoonEnv (Environment Wrapper)

**File**: `envs/selector_env.py`

Gymnasium-compatible multi-agent env. Converts PlatoonEnv into a Discrete-action env
suitable for RLlib MAPPO.

```python
class SelectorPlatoonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        self.base_env = PlatoonEnv(config)
        self.generator = DiffusionCandidateGenerator(config["pretrained_ckpt"])
        self.K = config.get("K", 9)
        self.num_agents = config.get("num_agents", 3)
        self._agent_ids = [f"agent{i}" for i in range(self.num_agents)]

        # Observation per agent: formation_state(12) + flattened candidates(K*8*3)
        obs_dim = 12 + self.K * 8 * 3
        self.observation_space = gym.spaces.Dict({
            aid: gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            for aid in self._agent_ids
        })
        self.action_space = gym.spaces.Dict({
            aid: gym.spaces.Discrete(self.K)
            for aid in self._agent_ids
        })
        self._cached_trajs: dict = {}

    def reset(self, *, seed=None, options=None):
        raw_obs, info = self.base_env.reset()
        return self._build_selector_obs(raw_obs), info

    def step(self, actions: dict):
        # actions: Dict[agent_id -> int]
        traj_actions = {
            aid: self._cached_trajs[aid][actions[aid]]
            for aid in actions
        }
        raw_obs, _, terminated, truncated, info = self.base_env.low_level_step(traj_actions)
        reward = {aid: self._compute_reward(info[aid]) for aid in info}
        return self._build_selector_obs(raw_obs), reward, terminated, truncated, info

    def _build_selector_obs(self, raw_obs: dict) -> dict:
        selector_obs = {}
        for aid in self._agent_ids:
            if aid not in raw_obs:
                continue
            trajs = self.generator.generate(raw_obs[aid])  # (K, 8, 3)
            self._cached_trajs[aid] = trajs
            formation = raw_obs[aid]["formation_relation_state"]  # (12,)
            selector_obs[aid] = np.concatenate([
                formation,
                trajs.reshape(-1)   # K*8*3
            ]).astype(np.float32)
        return selector_obs

    @staticmethod
    def _compute_reward(agent_info: dict) -> float:
        r = 0.0
        r -= 0.5  * agent_info.get("formation_error", 0.0)
        r -= 5.0  * float(agent_info.get("crash", False))
        r -= 2.0  * float(agent_info.get("out_of_road", False))
        r += 0.05 * agent_info.get("progress", 0.0)
        r += 0.1  * min(agent_info.get("speed_km_h", 0.0) / 40.0, 1.0)
        return r
```

**Acceptance criteria**:
- `env.observation_space` and `env.action_space` are valid gym Dict spaces
- `reset()` and `step()` return 5-tuples with correct agent keys
- Reward is a dict keyed by agent IDs
- `"__all__"` key in terminated/truncated for RLlib episode termination

---

### Task 3 — IntentSelector (Policy Network)

**File**: `models/selector/intent_selector.py`

Lightweight MLP policy used as the MAPPO actor.

```python
class IntentSelector(nn.Module):
    """
    Input:  flat vector [formation_state(12) + traj_candidates(K*8*3)]
    Output: logits over K intents
    """
    def __init__(self, obs_dim: int, K: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)   # (B, K) logits
```

For MAPPO, the Centralized Critic has the same architecture but takes the
concatenated observations of all agents:

```python
class CentralizedCritic(nn.Module):
    """
    Input:  joint obs = concat of all agents' obs vectors  (num_agents * obs_dim,)
    Output: scalar value estimate
    """
    def __init__(self, joint_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint_obs: Tensor) -> Tensor:
        return self.net(joint_obs).squeeze(-1)  # (B,)
```

**Acceptance criteria**:
- `IntentSelector.forward(obs)` accepts shape `(B, obs_dim)` → `(B, K)`
- `CentralizedCritic.forward(joint_obs)` accepts shape `(B, num_agents*obs_dim)` → `(B,)`
- Both modules unit-tested with random inputs

---

### Task 4 — MAPPO Training Script

**File**: `train/train_selector.py`

Use RLlib's PPO with a custom model that implements CTDE (Centralized Training,
Decentralized Execution).

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray import tune

# Register custom model
ModelCatalog.register_custom_model("intent_selector", IntentSelectorRLlibModel)

config = (
    PPOConfig()
    .environment(
        env=SelectorPlatoonEnv,
        env_config={
            "pretrained_ckpt": args.pretrained_ckpt,
            "num_agents": 3,
            "K": 9,
            "num_scenarios": 100,
            "traffic_density": 0.04,
        },
    )
    .multi_agent(
        policies={"shared_selector": PolicySpec(
            model_config={"custom_model": "intent_selector"}
        )},
        policy_mapping_fn=lambda agent_id, *_: "shared_selector",
    )
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        train_batch_size=4000,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
    )
    .rollouts(num_rollout_workers=4, rollout_fragment_length=200)
    .resources(num_gpus=0)   # pretrained model runs on CPU in workers
    .callbacks(PlatoonFormationCallbacks)
)

tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 500},
    checkpoint_freq=20,
    local_dir="outputs/phase6_mappo",
)
```

**Centralized Critic via RLlib custom model**:

```python
class IntentSelectorRLlibModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # actor:  IntentSelector
        # critic: CentralizedCritic (takes full-obs from info["opponent_obs"])
        ...

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        self._value_input = input_dict["obs"]["state"]  # joint obs injected via callbacks
        return self.actor(obs), state

    def value_function(self):
        return self.critic(self._value_input)
```

Joint obs injection: RLlib's `CentralizedCriticMixin` pattern or custom `postprocess_trajectory`
callback that concatenates all agents' obs into the value input.

**Acceptance criteria**:
- Training starts without errors
- `mean_reward` is logged per iteration
- `formation_error` tracked in custom metrics
- Checkpoint saved every 20 iterations to `outputs/phase6_mappo/`

---

### Task 5 — PlatoonFormationCallbacks

**File**: `train/selector_callbacks.py`

Custom RLlib callbacks to log platoon-specific metrics.

```python
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class PlatoonFormationCallbacks(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        info = episode.last_info_for()
        if info:
            episode.user_data.setdefault("formation_errors", []).append(
                info.get("formation_error", 0.0)
            )
            episode.user_data.setdefault("crashes", []).append(
                float(info.get("crash", False))
            )

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        episode.custom_metrics["formation_error_mean"] = np.mean(
            episode.user_data.get("formation_errors", [0])
        )
        episode.custom_metrics["crash_rate"] = np.mean(
            episode.user_data.get("crashes", [0])
        )
```

**Acceptance criteria**:
- `formation_error_mean` and `crash_rate` appear in RLlib training logs
- Metrics are per-episode averages

---

### Task 6 — Config File

**File**: `configs/train/selector.yaml`

```yaml
pretrained_ckpt: "outputs/pretrained/diffusion_planner.ckpt"
num_agents: 3
K: 9                        # number of semantic anchors / intent candidates
num_scenarios: 100
traffic_density: 0.04

# MAPPO hyperparameters
lr: 3.0e-4
gamma: 0.99
lambda_gae: 0.95
clip_param: 0.2
train_batch_size: 4000
sgd_minibatch_size: 256
num_sgd_iter: 10
num_rollout_workers: 4
rollout_fragment_length: 200
max_iterations: 500
checkpoint_freq: 20

# Reward weights
w_formation: 0.5
w_crash: 5.0
w_out_of_road: 2.0
w_progress: 0.05
w_speed: 0.1
```

---

## File Map

### New files to create

| File | Role |
|------|------|
| `models/selector/__init__.py` | package init |
| `models/selector/candidate_generator.py` | frozen diffusion model wrapper |
| `models/selector/intent_selector.py` | actor + centralized critic |
| `envs/selector_env.py` | SelectorPlatoonEnv gym wrapper |
| `train/train_selector.py` | RLlib MAPPO training entry point |
| `train/selector_callbacks.py` | RLlib custom callbacks |
| `configs/train/selector.yaml` | training hyperparameters |

### Existing files — keep unchanged

| File | Role in new architecture |
|------|--------------------------|
| `envs/platoon_env.py` | inner environment, unchanged |
| `models/platoon/platoon_diffusion_planner.py` | loaded frozen into CandidateGenerator |
| `metadrive/exp_dataset/abstract_anchors.py` | generates anchor trajectories |
| `metadrive/exp_dataset/metadrive_anchors.npy` | pre-computed anchor set (K=9) |

### Deleted files (GRPO-specific, not reusable)

| File | Reason deleted |
|------|----------------|
| `train/ma_grpo_trainer.py` | GRPO trainer replaced by RLlib MAPPO |
| `train/train_selector.py` | RLlib MAPPO selector training entry |
| `train/joint_group.py` | GRPO joint group selection utility |
| `train/closedloop_executor.py` | GRPO group-evaluation executor |
| `models/diffusion/diffusion_rl_scheduler.py` | log_prob replay chain (GRPO-only) |
| `tests/acceptance/test_phase5_rl_training_fix.py` | tests for deleted GRPO fixes |

---

## Execution Order

```
Task 1  →  Task 2  →  Task 3  →  Task 4  →  Task 5  →  Task 6
candidate    env       network    training    callbacks   config
generator   wrapper    MLP        script
```

Tasks 1–3 are independent of RLlib; test them with unit tests before
wiring into RLlib in Tasks 4–6.

---

## Open Questions

1. **Anchor coverage**: The current 9 anchors were clustered from single-vehicle
   highway data. Verify that at least 2–3 anchors correspond to "follow leader at
   constant headway" behavior; if not, add platoon-specific anchors via
   `abstract_anchors.py`.

2. **Diffusion inference in workers**: Each RLlib rollout worker loads the
   pretrained model. With 4 workers on CPU, verify total RAM fits. Alternative:
   run diffusion model on a dedicated server process and pass candidates via
   shared memory.

3. **Joint obs construction for Centralized Critic**: RLlib does not natively
   broadcast all agents' obs to each agent's critic. Use
   `postprocess_trajectory` callback to inject concatenated joint obs, or use
   MARLLIB's built-in MAPPO which handles this automatically.
   MARLLIB option: `pip install marllib` and use `marllib.marl.build_algo("mappo")`
   instead of raw RLlib PPO — simplifies centralized critic wiring considerably.
