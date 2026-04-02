# RL Training Fix Plan — Phase 5

## Background

run_2 (8000 steps, platoon-closedloop) completely failed:
- collision_rate: 0.75 -> 1.0 (step 500 already 0.98)
- KL: 0 -> 220 (target=8, never controlled)
- grad_norm: saturated at max_grad_norm=10 throughout
- mean_reward: -19.7 -> -73.6
- formation_error: 5.5 -> 105.4

Root cause chain: ratio=1 (no policy constraint) + advantage non-negative only (no penalty signal) + MSE regularization (wrong metric) = unconstrained policy drift.

---

## P0: Must Fix (Training Cannot Converge Without These)

### P0.1 — Fix `compute_rl_loss` ratio calculation

**File**: `train/ma_grpo_trainer.py`, line 216-233

**Current (BROKEN)**:
```python
def compute_rl_loss(self, rollouts: dict, advantages: dict) -> Tensor:
    losses = []
    self._cached_current_preds = {}
    for agent_id in rollouts["agent_ids"]:
        replay_all, model_outputs = self.scheduler.replay_with_log_prob(
            self.model,
            rollouts["batch"],
            agent_id,
            rollouts[agent_id]["diffusion_chain"],
            return_model_outputs=True,
        )
        self._cached_current_preds[agent_id] = model_outputs
        advantage = advantages[agent_id]
        per_token_loss = -torch.exp(replay_all - replay_all.detach()) * advantage
        mask_nz = per_token_loss != 0
        rl_loss = (per_token_loss * mask_nz).sum() / mask_nz.sum().clamp(min=1)
        losses.append(rl_loss)
    return torch.stack(losses).mean()
```

**Bug**: `replay_all - replay_all.detach()` is always 0, so `exp(0) = 1`, ratio is constant 1. There is no importance sampling constraint. The model can update arbitrarily far from the reference policy.

**Fix**: Use the sampling-time log_prob stored in rollouts as the "old" log_prob baseline. The GRPO ratio should be `exp(log_pi_current - log_pi_old)`.

```python
def compute_rl_loss(self, rollouts: dict, advantages: dict) -> Tensor:
    losses = []
    self._cached_current_preds = {}
    for agent_id in rollouts["agent_ids"]:
        # Current policy log_prob (with grad)
        replay_all, model_outputs = self.scheduler.replay_with_log_prob(
            self.model,
            rollouts["batch"],
            agent_id,
            rollouts[agent_id]["diffusion_chain"],
            return_model_outputs=True,
        )
        self._cached_current_preds[agent_id] = model_outputs

        # Old policy log_prob (from sampling time, no grad)
        old_log_prob = rollouts[agent_id]["log_prob_all"].detach()  # [G, M, step_num]

        advantage = advantages[agent_id]  # [G, M, step_num]

        # Importance sampling ratio
        log_ratio = replay_all - old_log_prob
        ratio = torch.exp(log_ratio)

        # Clipped surrogate (PPO-style clip for stability)
        clip_eps = float(self.config.get("clip_eps", 0.2))
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        per_token_loss = -torch.min(surr1, surr2)

        mask_nz = advantage != 0
        rl_loss = (per_token_loss * mask_nz).sum() / mask_nz.sum().clamp(min=1)
        losses.append(rl_loss)
    return torch.stack(losses).mean()
```

**Config change** in `configs/train/platoon_grpo_v2.yaml`: add `clip_eps: 0.2`

**Tests**: Add unit test in `tests/` that verifies:
1. When model == ref_model (step 0), ratio should be ~1.0 everywhere
2. After one gradient step, ratio should deviate from 1.0 (not stay at 1.0)
3. Clipping should bound the ratio to [0.8, 1.2]

---

### P0.2 — Fix `compute_advantages` to retain negative advantages

**File**: `train/ma_grpo_trainer.py`, line 187-214

**Current (BROKEN)**:
```python
normalized = (reward - mean_r) / (std_r + 1e-4)
positive_mask = reward > mean_r
normalized = normalized.clamp(min=0.0) * positive_mask.float()
```

**Bug**: All below-mean rewards get advantage=0. The model only learns "what is good" but never "what is bad". Crash trajectories with reward slightly below mean get no penalty signal at all.

**Fix**: Keep standard GRPO two-sided normalization. Only apply crash mask separately.

```python
def compute_advantages(self, rollouts: dict) -> dict:
    advantages: dict = {}
    discount = torch.tensor(
        [self.advantage_gamma ** (self.scheduler.step_num - i - 1) for i in range(self.scheduler.step_num)],
        dtype=torch.float32,
        device=self.device,
    )
    for agent_id in rollouts["agent_ids"]:
        reward = rollouts[agent_id]["reward_per_anchor"]
        crash_mask = rollouts[agent_id]["crash_per_anchor"] | rollouts[agent_id]["out_per_anchor"]
        mean_r = reward.mean(dim=0, keepdim=True)
        std_r = reward.std(dim=0, unbiased=False, keepdim=True)
        normalized = (reward - mean_r) / (std_r + 1e-4)

        # Keep both positive and negative advantages (standard GRPO)
        # Do NOT clamp to 0 or mask positive only

        advantage = normalized.unsqueeze(-1) * discount.unsqueeze(0).unsqueeze(0)

        # Crash/out-of-road trajectories get a fixed negative advantage
        # to ensure they are strongly penalized regardless of distribution
        crash_penalty = float(self.config.get("crash_advantage_penalty", -2.0))
        advantage = torch.where(
            crash_mask.unsqueeze(-1),
            torch.full_like(advantage, crash_penalty),
            advantage,
        )
        advantages[agent_id] = advantage
    return advantages
```

**Config change**: add `crash_advantage_penalty: -2.0` to yaml

**Tests**: Verify that for a batch where half the rewards are above mean and half below, both positive and negative advantages exist in the output.

---

### P0.3 — Replace MSE ref_reg_loss with KL-based regularization

**File**: `train/ma_grpo_trainer.py`, line 251-279

**Current (BROKEN)**:
```python
step_losses.append(discount[step_idx] * F.mse_loss(curr, ref.detach(), reduction="mean"))
```

**Bug**: MSE between model outputs measures prediction-space distance, not probability distance. When the model outputs shift in a way that dramatically changes the trajectory distribution but keeps MSE small (e.g., a small change in a high-sensitivity region), the regularization doesn't activate. Conversely, MSE can be large for harmless changes. This is why KL exploded to 220 while the MSE loss appeared "controlled".

**Fix**: Use KL divergence between current and reference log_probs.

```python
def compute_ref_reg_loss(self, rollouts: dict) -> Tensor:
    losses = []
    for agent_id in rollouts["agent_ids"]:
        chain = rollouts[agent_id]["diffusion_chain"]

        # Current model log_prob (reuse cached from compute_rl_loss if available)
        current_log_prob = self.scheduler.replay_with_log_prob(
            self.model,
            rollouts["batch"],
            agent_id,
            chain,
        )  # [G, M, step_num]

        # Reference model log_prob
        ref_log_prob = self.scheduler.replay_with_log_prob(
            self.ref_model,
            rollouts["batch"],
            agent_id,
            chain,
        )  # [G, M, step_num]

        # Forward KL: E_current[log(current/ref)] = mean(current_log_prob - ref_log_prob)
        kl_per_step = current_log_prob - ref_log_prob.detach()
        kl_loss = kl_per_step.mean()
        losses.append(kl_loss)
    return torch.stack(losses).mean()
```

**Important**: `self.scheduler.replay_with_log_prob` for `self.ref_model` currently goes through `compute_ref_predictions` which only returns model_outputs, not log_probs. You need to call `replay_with_log_prob` directly with `self.ref_model`. Since ref_model is frozen (no grad), wrap in `torch.no_grad()`:

```python
with torch.no_grad():
    ref_log_prob = self.scheduler.replay_with_log_prob(
        self.ref_model,
        rollouts["batch"],
        agent_id,
        chain,
    )
```

**Also fix `_get_beta_reg`**: Since we now have proper KL in the loss, slow down the decay schedule. Change defaults:

```yaml
beta_reg_max: 0.5
beta_reg_min: 0.1
beta_reg_warmup_frac: 0.2
beta_reg_decay_frac: 0.6
```

**Tests**: Verify that when model == ref_model, ref_reg_loss is ~0. After perturbing model weights, ref_reg_loss should increase.

---

## P1: Serious Issues (Will Cause Suboptimal Training)

### P1.1 — Fix open-loop vs closed-loop crash signal mismatch in team_advantages

**File**: `train/ma_grpo_trainer.py`, line 281-320

**Current (BROKEN)**:
```python
for group_idx, group in enumerate(joint_groups):
    any_crash = False
    for agent_id, (g_idx, k_idx) in group.items():
        if bool(rollouts[agent_id]["crash_per_anchor"][g_idx, k_idx].item()):
            any_crash = True
            break
    if any_crash:
        normalized[group_idx] = -1.0
```

**Bug**: `crash_per_anchor` comes from the open-loop surrogate evaluation in `evaluate_trajectory_group()`, but `team_rewards` comes from the closed-loop execution in `closedloop_executor`. These two evaluations use completely different dynamics:
- Surrogate: constant-velocity extrapolation of other agents
- Closed-loop: all agents execute their selected trajectories simultaneously

A trajectory marked as "crash" in the surrogate may be safe in closed-loop (other agents moved), and vice versa. Using the surrogate crash flag to override the closed-loop team reward creates contradictory training signals.

**Fix**: Use the closed-loop execution results to determine crash status for team advantages.

Step 1: Modify `closedloop_executor._execute_single_group_standalone` to return per-agent crash flags (it already does via `crash_flags`).

Step 2: In `_build_joint_training_inputs` (train_platoon_rl.py), collect per-group crash info from exec_results:

```python
# After exec_results = executor.execute_joint_groups(joint_trajectories)
closedloop_crash_flags = []
for result in exec_results:
    any_crash = any(result["crash_flags"].values())
    closedloop_crash_flags.append(any_crash)
```

Step 3: Pass `closedloop_crash_flags` to `trainer.update()` and use it in `compute_team_advantages` instead of `crash_per_anchor`:

```python
def compute_team_advantages(self, rollouts, joint_groups, team_rewards,
                            closedloop_crash_flags=None) -> dict:
    # ... normalize team_rewards ...

    for group_idx, group in enumerate(joint_groups):
        if closedloop_crash_flags is not None and closedloop_crash_flags[group_idx]:
            normalized[group_idx] = -1.0
        # Remove the old crash_per_anchor check entirely
```

**Tests**: Create a test case where surrogate says crash but closed-loop says safe, verify team_advantage is positive (not -1.0).

---

### P1.2 — Fix `_ensure_agents_alive` ordering in closedloop_executor

**File**: `train/closedloop_executor.py`, line 207-225

**Current (BROKEN)**:
```python
for joint_group in joint_groups:
    self._ensure_agents_alive()    # may call env.reset()
    pre_restore = self._restore_state(saved_state)   # then set_state
    result = self._execute_single_group(joint_group)
```

**Bug**: If previous group terminated all agents, `_ensure_agents_alive()` calls `env.reset()` which creates entirely new agent objects. Then `set_state()` tries to map saved state to these new objects by name lookup, which may fail silently (agent objects skip restoration via `continue`). The group then executes with partially-restored or default-state agents.

**Fix**: Move `_ensure_agents_alive` inside `_restore_state`, and make state restoration atomic — if any agent fails to restore, raise an error.

```python
def _restore_state(self, saved_state: dict) -> dict[str, float]:
    profile = self._empty_restore_profile()
    restore_start = self._stamp()
    # Ensure agents exist before attempting state restoration
    self._ensure_agents_alive()
    self.env.set_state(saved_state)
    restore_end = self._stamp()
    profile["restore_set_state"] = restore_end - restore_start
    profile["restore_total"] = restore_end - restore_start
    return profile
```

And update `execute_joint_groups` to remove the separate `_ensure_agents_alive()` call:

```python
def execute_joint_groups(self, joint_groups):
    saved_state = self.env.get_state()
    results = []
    aggregate = self._empty_group_profile()
    aggregate["group_count"] = float(len(joint_groups))
    try:
        for joint_group in joint_groups:
            # _ensure_agents_alive is now inside _restore_state
            pre_restore = self._restore_state(saved_state)
            self._accumulate_restore(aggregate, "pre_restore", pre_restore)
            # ... rest unchanged ...
```

Also fix `_worker_loop` (line 100-115) with the same pattern: move `needs_reset` handling into `set_state` or always reset before `set_state`.

**Tests**: Create a test where first group terminates all agents, verify second group still executes correctly from saved state.

---

### P1.3 — Fix trajectory yaw always being zero

**File**: `models/diffusion/diffusion_rl_scheduler.py`, line 243-254

**Current**:
```python
sample_phys = th.denorm_odo(
    torch.cat([sample, torch.zeros_like(sample[..., :1])], dim=-1)
)[..., :2]  # takes only xy

if hasattr(th, "bezier_xyyaw"):
    trajectory = th.bezier_xyyaw(sample_phys)  # supposed to add yaw
else:
    trajectory = torch.cat(
        [sample_phys, torch.zeros_like(sample_phys[..., :1])], dim=-1
    )  # zero yaw fallback
```

**Bug**: `bezier_xyyaw` in `transfuser_model_v2.py` computes yaw from xy waypoint differences:
```python
def bezier_xyyaw(self, xy):
    # computes heading from consecutive waypoint differences
    dx = xy[..., 1:, 0] - xy[..., :-1, 0]
    dy = xy[..., 1:, 1] - xy[..., :-1, 1]
    yaw = torch.atan2(dy, dx)
    ...
```
This should produce non-zero yaw. But the `ToyPlanner.StubTrajectoryHead.bezier_xyyaw` just returns `cat([xy, zeros])`, and if the real model's `bezier_xyyaw` is not working correctly, all trajectories have yaw=0.

**Investigation required**: Read the actual `bezier_xyyaw` implementation in `transfuser_model_v2.py` and verify it produces correct yaw values. If it does, this is fine for the real model (only the toy model is affected). If it doesn't, the `_lateral_pd` function in platoon_env.py uses `waypoint[2]` (yaw) for steering, and zero yaw means all steering comes from `arctan2(y, x)` only — partial but not catastrophic.

**Fix (if bezier_xyyaw produces zero yaw)**: Compute heading from consecutive waypoint differences explicitly:

```python
# After sample_phys is computed [G, M, 8, 2]
dx = sample_phys[..., 1:, 0] - sample_phys[..., :-1, 0]  # [G, M, 7]
dy = sample_phys[..., 1:, 1] - sample_phys[..., :-1, 1]  # [G, M, 7]
yaw = torch.atan2(dy, dx)  # [G, M, 7]
# Pad first step with same heading as second step
yaw = torch.cat([yaw[..., :1], yaw], dim=-1)  # [G, M, 8]
trajectory = torch.cat([sample_phys, yaw.unsqueeze(-1)], dim=-1)  # [G, M, 8, 3]
```

**Tests**: Generate a non-straight trajectory and verify yaw is non-zero and consistent with waypoint directions.

---

### P1.4 — Fix `compute_team_advantages` non-negative clamp

**File**: `train/ma_grpo_trainer.py`, line 288-290

**Current**:
```python
normalized = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
normalized = normalized.clamp(min=0.0)  # Same bug as local advantages!
```

**Bug**: Same as P0.2 — team advantages also clamp to non-negative, losing half the signal.

**Fix**: Remove the clamp:
```python
normalized = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
# Do NOT clamp — keep both positive and negative team advantages
```

---

## P2: Important Improvements (Improve Training Quality)

### P2.1 — Fix `select_top_k_candidates` to be per-agent instead of global

**File**: `train/joint_group.py`, line 12-31

**Current**: Selects global top-k across the flattened `[G, M]` reward tensor. Since this function is called once per agent (see `_build_joint_training_inputs`), it actually already IS per-agent. However, the selection picks the top-k from the `[group_size, num_modes]` matrix, which means it may pick multiple candidates from the same group_idx but different mode_idx, or vice versa.

**Verify**: Check if `per_agent_candidates` in `_build_joint_training_inputs` is indeed populated per-agent. If yes, the current logic is correct (it's top-k per agent from that agent's G*M candidates). No change needed.

**If the bug IS confirmed**: change to select top-k by first taking the best mode per group, then taking top-k groups:

```python
def select_top_k_candidates(reward_per_anchor, crash_per_anchor, top_k=2):
    masked_reward = reward_per_anchor.clone()
    masked_reward[crash_per_anchor] = float('-inf')
    if bool(torch.isneginf(masked_reward).all().item()):
        masked_reward = reward_per_anchor.clone()

    # Best mode per group
    best_mode_per_group = masked_reward.argmax(dim=1)  # [G]
    best_reward_per_group = masked_reward.gather(1, best_mode_per_group.unsqueeze(1)).squeeze(1)  # [G]

    k = min(int(top_k), best_reward_per_group.numel())
    _, top_group_indices = torch.topk(best_reward_per_group, k)

    candidates = []
    for g_idx in top_group_indices.tolist():
        k_idx = int(best_mode_per_group[g_idx].item())
        candidates.append((g_idx, k_idx))
    return candidates
```

---

### P2.2 — Fix sampling/log_prob std_dev mismatch

**File**: `models/diffusion/diffusion_rl_scheduler.py` (DDIMSchedulerWithLogProb), line 88 and 113

**Current**:
- Sampling: `std_dev_t_mul = std_dev_t.clamp(min=0.04)` (line 88)
- Log_prob: `std_dev_for_log = std_dev_t.clamp(min=0.1)` (line 113)

**Bug**: The log_prob is computed with a wider Gaussian (sigma >= 0.1) than what was actually used for sampling (sigma >= 0.04). This means the log_prob is systematically biased — samples that are far from the mean get higher log_prob than they should, which distorts the RL gradient signal.

**Fix**: Use the same clamp value for both. Since `min=0.1` provides better numerical stability for log_prob computation, use `min=0.1` for sampling as well:

```python
# Line 88
if eta > 0:
    std_dev_t_mul = std_dev_t.clamp(min=0.1)  # was 0.04
else:
    std_dev_t_mul = torch.zeros_like(std_dev_t)
```

Or alternatively, if 0.04 is intentional for exploration, use 0.04 for log_prob too:
```python
# Line 113
std_dev_for_log = std_dev_t.clamp(min=0.04)  # was 0.1
```

**Recommendation**: Use `min=0.1` for both — the difference in exploration between 0.04 and 0.1 is negligible with `eta=0.02`, but the log_prob accuracy matters for the ratio calculation (especially after fixing P0.1).

---

### P2.3 — Add adaptive KL penalty (dual optimization)

**File**: `train/ma_grpo_trainer.py`, new method

Instead of relying solely on beta_reg schedule, add a KL-adaptive penalty that increases beta when KL exceeds target and decreases when KL is below target. This provides a safety net against KL explosion.

```python
def _adapt_beta_reg(self, current_kl: float) -> float:
    """Adapt beta_reg based on current KL vs target (Lagrangian dual update)."""
    kl_target = float(self.config.get("kl_target", 8.0))
    beta_adapt_rate = float(self.config.get("beta_adapt_rate", 0.01))

    scheduled_beta = self._get_beta_reg()

    if current_kl > kl_target * 1.5:
        # KL too high — increase beta
        self._beta_multiplier = getattr(self, '_beta_multiplier', 1.0) * (1.0 + beta_adapt_rate)
    elif current_kl < kl_target * 0.5:
        # KL too low — decrease beta (allow more exploration)
        self._beta_multiplier = getattr(self, '_beta_multiplier', 1.0) * (1.0 - beta_adapt_rate)
    self._beta_multiplier = max(0.1, min(getattr(self, '_beta_multiplier', 1.0), 10.0))

    return scheduled_beta * self._beta_multiplier
```

Call this in `update()` instead of `_get_beta_reg()`:
```python
# In update(), after computing kl:
beta_reg = self._adapt_beta_reg(float(kl.item()))
```

---

### P2.4 — Add early stopping on KL explosion

**File**: `train/train_platoon_rl.py`, inside the training loop

Add a safety check that stops training if KL exceeds a hard threshold:

```python
# After metrics = trainer.update(...)
if float(metrics.get("kl", 0.0)) > float(config.get("kl_hard_limit", 50.0)):
    print(f"[ABORT] KL={metrics['kl']:.1f} exceeds hard limit. Saving checkpoint and stopping.")
    _save_checkpoint(model, resolved_checkpoint_dir, step_idx + 1, metrics)
    break
```

**Config**: add `kl_hard_limit: 50.0` to yaml.

---

## Execution Order

```
Phase A (P0, blocking):
  P0.1 → P0.2 → P0.3  (sequential, each depends on correct loss formulation)

Phase B (P1, can parallel after Phase A):
  P1.1 + P1.2 + P1.3 + P1.4  (independent of each other)

Phase C (P2, can parallel after Phase A):
  P2.1 + P2.2 + P2.3 + P2.4  (independent of each other)
```

## Validation Plan

After all fixes, run a short 200-step training on `platoon-closedloop` mode and verify:

1. **KL stays controlled**: KL should stay in range [2, 15], NOT explode to 200+
2. **Collision rate decreases**: should trend downward or at least not increase to 1.0
3. **Ratio is bounded**: log `ratio.mean()` and `ratio.max()` — should be in [0.8, 1.2] range
4. **Negative advantages exist**: log `(advantage < 0).float().mean()` — should be ~0.3-0.5
5. **grad_norm not saturated**: should NOT be at max_grad_norm=10 every step

```bash
python train/train_platoon_rl.py \
  --mode platoon-closedloop \
  --steps 200 \
  --config configs/train/platoon_grpo_v2.yaml
```

## Config Summary (platoon_grpo_v2.yaml changes)

```yaml
# Add these new keys:
clip_eps: 0.2
crash_advantage_penalty: -2.0
kl_hard_limit: 50.0
beta_adapt_rate: 0.01

# Modify these existing keys:
beta_reg_max: 0.5       # was 1.0
beta_reg_min: 0.1       # unchanged
beta_reg_warmup_frac: 0.2  # was 0.3
beta_reg_decay_frac: 0.6   # was 0.4
```
