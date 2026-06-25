# Rear-End Guard Regulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rear-end guard to `HierarchicalExpertIDMPolicy` that significantly reduces `rear_end` collisions while preserving zero intersection collisions and improving `arrive_dest` in real `meta_drive` runs.

**Architecture:** Keep the existing steering, lane-change manager, and intersection regulator unchanged. Add a small stateful `RearEndGuardRegulator` that only makes IDM longitudinal control more conservative, wire it in before `IntersectionConflictRegulator`, expose diagnostics through `action_info`, and validate with both unit tests and real `run_expert.py` acceptance.

**Tech Stack:** Python, NumPy, pytest, MetaDrive runtime, `HierarchicalExpertIDMPolicy`

---

## File Map

- Create: `expert_dataset/hierarchical_expert/rear_end_guard.py`
  Purpose: implement `RearEndGuardRegulator`
- Modify: `expert_dataset/hierarchical_expert/hierarchical_policy.py`
  Purpose: instantiate regulator, apply it after IDM acceleration and before intersection regulation, reset it on episode reset, emit diagnostics
- Modify: `expert_dataset/hierarchical_expert/__init__.py`
  Purpose: export `RearEndGuardRegulator`
- Create: `expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py`
  Purpose: isolated unit tests for guard logic
- Modify: `expert_dataset/hierarchical_expert/tests/test_hierarchical_policy.py`
  Purpose: verify policy wiring, ordering, reset behavior, and diagnostics
- Modify: `expert_dataset/hierarchical_expert/tests/test_integration.py`
  Purpose: synthetic longitudinal safety regression coverage

## Fixed Runtime Baseline

Use this exact non-sandbox command for acceptance:

```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
/home/kong/anaconda3/envs/meta_drive/bin/python -m expert_dataset.run_expert \
  --expert-type idm \
  --episodes 10 \
  --render 0 \
  --print-obs-summary 0 \
  --print-idm-debug 0 \
  --print-episode-summary 1 \
  --print-run-summary 1
```

Current baseline to beat:

- `collision_rear_end = 4`
- `arrive_dest = 1`
- `out_of_road = 5`
- `collision_intersection = 0`

Target for this round:

- `collision_rear_end <= 2`
- `arrive_dest >= 2`
- `collision_intersection = 0`
- `out_of_road <= 6`
- `crash <= 4`
- `collision_lane_change <= 1`

---

### Task 1: Rear-End Guard Unit Tests

**Files:**
- Create: `expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py`
- Reference: `expert_dataset/hierarchical_expert/docs/rear-end-guard-design.md`

- [ ] **Step 1: Write the failing test for no-front-object passthrough**

```python
def test_guard_returns_idm_acc_when_front_object_missing():
    guard = RearEndGuardRegulator()
    ego = SimpleNamespace(speed=10.0)
    assert guard.adjust_acceleration(ego=ego, front_obj=None, front_dist=100.0, idm_acc=0.2) == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py -q
```
Expected: FAIL because module/class does not exist yet.

- [ ] **Step 3: Add failing tests for front-speed extraction, soft guard, hard guard, and reset**

Cover:
- front speed from `speed`
- front speed from `speed_km_h`
- front speed from `velocity_km_h`
- no closing speed preserves `idm_acc`
- when `front_obj is None`, diagnostics are:
  - `rear_end_guard_active is False`
  - `rear_end_guard_ttc is None`
  - `rear_end_guard_gap == front_dist`
- soft guard returns `min(idm_acc, SOFT_BRAKE)`
- hard guard returns `min(idm_acc, HARD_BRAKE)`
- stronger existing IDM brake is preserved
- rate limit applies across sequential calls
- `reset()` clears internal state

- [ ] **Step 4: Re-run test file to verify all new tests fail for the intended reasons**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py -q
```
Expected: FAIL on missing implementation, not on test syntax.

- [ ] **Step 5: Commit test-only red state**

```bash
git add expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py
git commit -m "test: add rear-end guard unit coverage"
```

### Task 2: Minimal Rear-End Guard Implementation

**Files:**
- Create: `expert_dataset/hierarchical_expert/rear_end_guard.py`
- Test: `expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py`

- [ ] **Step 1: Implement class skeleton and constants**

Add:

```python
class RearEndGuardRegulator:
    BASE_MIN_GAP = 6.0
    HEADWAY_GUARD = 1.6
    TTC_SOFT = 3.0
    TTC_HARD = 1.6
    SOFT_BRAKE = -1.8
    HARD_BRAKE = -4.0
    MAX_BRAKE = -4.5
    ACC_RATE_LIMIT = 1.0
```

and methods:
- `adjust_acceleration(...)`
- `_front_speed(front_obj)`
- `_limit_accel_change(accel_cmd)`
- `reset()`

- [ ] **Step 2: Implement minimal passthrough and speed extraction behavior**

Rules:
- if `front_obj is None`: return `idm_acc`
- if front speed unavailable: return `idm_acc`
- if `closing_speed <= 0`: return `idm_acc`

- [ ] **Step 3: Implement soft/hard TTC and dynamic-gap logic**

Use:

```python
safe_gap = BASE_MIN_GAP + HEADWAY_GUARD * v_ego
ttc = gap / max(closing_speed, eps)
```

Decision:
- hard: `a_guard = min(idm_acc, HARD_BRAKE)`
- soft: `a_guard = min(idm_acc, SOFT_BRAKE)`
- otherwise: `a_guard = idm_acc`

Important:
- if `idm_acc < MAX_BRAKE`, preserve `idm_acc`
- otherwise cap only the newly-added guard brake with `max(a_guard, MAX_BRAKE)`

- [ ] **Step 4: Implement rate limiting and reset**

Behavior:
- internal field `self._last_accel`
- first call sets it
- later calls clamp change to `ACC_RATE_LIMIT`
- `reset()` sets it back to `None`

- [ ] **Step 5: Run unit tests and make them pass**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py -q
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add expert_dataset/hierarchical_expert/rear_end_guard.py expert_dataset/hierarchical_expert/tests/test_rear_end_guard.py
git commit -m "feat: add rear-end guard regulator"
```

### Task 3: Policy Wiring and Diagnostics

**Files:**
- Modify: `expert_dataset/hierarchical_expert/hierarchical_policy.py`
- Modify: `expert_dataset/hierarchical_expert/__init__.py`
- Modify: `expert_dataset/hierarchical_expert/tests/test_hierarchical_policy.py`

- [ ] **Step 1: Add failing policy tests**

Add tests for:
- policy initializes `RearEndGuardRegulator`
- regulator is called after `self.acceleration(...)`
- regulator runs before `IntersectionConflictRegulator`
- final action remains `[steering, acceleration]`
- `reset()` clears rear-end guard state
- `action_info` contains:
  - `rear_end_guard_active`
  - `rear_end_guard_acc`
  - `rear_end_guard_ttc`
  - `rear_end_guard_gap`
- no-front-object path sets:
  - `rear_end_guard_active == False`
  - `rear_end_guard_ttc is None`
  - `rear_end_guard_gap == front_dist`

- [ ] **Step 2: Run policy tests to verify red**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests/test_hierarchical_policy.py -q
```
Expected: FAIL on missing wiring/diagnostics.

- [ ] **Step 3: Implement wiring in `hierarchical_policy.py`**

Required changes:
- import `RearEndGuardRegulator`
- in `__init__`:

```python
self.rear_end_guard = RearEndGuardRegulator()
```

- in `act()`:
  - compute `idm_acc = acc`
  - compute `rear_end_acc = self.rear_end_guard.adjust_acceleration(...)`
  - then pass `rear_end_acc` into `self.intersection_regulator.adjust_acceleration(...)`
- write diagnostics from rear-end guard into `action_info`

- [ ] **Step 4: Implement reset contract**

In `reset()`:

```python
self.rear_end_guard.reset()
```

- [ ] **Step 5: Export in `__init__.py`**

Add import and `__all__` entry for `RearEndGuardRegulator`.

- [ ] **Step 6: Run policy tests to verify green**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests/test_hierarchical_policy.py -q
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add expert_dataset/hierarchical_expert/hierarchical_policy.py expert_dataset/hierarchical_expert/__init__.py expert_dataset/hierarchical_expert/tests/test_hierarchical_policy.py
git commit -m "feat: wire rear-end guard into hierarchical policy"
```

### Task 4: Synthetic Integration Coverage

**Files:**
- Modify: `expert_dataset/hierarchical_expert/tests/test_integration.py`

- [ ] **Step 1: Add failing integration tests for longitudinal rear-end behavior**

Add scenarios for:
- ego approaches slower front vehicle and guard makes acceleration more conservative than baseline
- strong IDM brake is preserved, not weakened by guard
- repeated guarded steps do not oscillate violently
- existing intersection integration expectations still pass

- [ ] **Step 2: Run integration tests to verify red**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests/test_integration.py -q
```
Expected: FAIL on missing behavior or diagnostics.

- [ ] **Step 3: Make minimal implementation adjustments if tests expose gaps**

Only change:
- `rear_end_guard.py`
- `hierarchical_policy.py`

Do not change:
- lane-change planner
- tracker
- intersection regulator

- [ ] **Step 4: Run integration tests to verify green**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests/test_integration.py -q
```
Expected: PASS

- [ ] **Step 5: Run full hierarchical expert suite**

Run:
```bash
/home/kong/anaconda3/envs/meta_drive/bin/python -m pytest expert_dataset/hierarchical_expert/tests -q
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add expert_dataset/hierarchical_expert/tests/test_integration.py expert_dataset/hierarchical_expert/rear_end_guard.py expert_dataset/hierarchical_expert/hierarchical_policy.py
git commit -m "test: cover rear-end guard integration behavior"
```

### Task 5: Real Runtime Acceptance in `meta_drive`

**Files:**
- No code changes required in this task unless runtime failures directly identify a rear-end guard bug

- [ ] **Step 1: Run fixed 10-episode acceptance baseline with new code in non-sandbox**

Run:
```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
/home/kong/anaconda3/envs/meta_drive/bin/python -m expert_dataset.run_expert \
  --expert-type idm \
  --episodes 10 \
  --render 0 \
  --print-obs-summary 0 \
  --print-idm-debug 0 \
  --print-episode-summary 1 \
  --print-run-summary 1
```

- [ ] **Step 2: Record exact runtime summary**

Use the single `[Run Summary]` line emitted by `run_expert.py` as the authoritative source of truth.

Parse these exact fields from that line:
- `arrive_dest`
- `out_of_road`
- `crash`
- `lane_changes`
- `collision_rear_end`
- `collision_intersection`
- `collision_lane_change`

- [ ] **Step 3: Compare against required thresholds**

Pass only if:
- `collision_rear_end <= 2`
- `arrive_dest >= 2`
- `collision_intersection = 0`
- `out_of_road <= 6`
- `crash <= 4`
- `collision_lane_change <= 1`

- [ ] **Step 4: If runtime misses target, do one bounded tuning round**

Allowed tuning surface:
- constants in `rear_end_guard.py` only

Do not expand scope into:
- out-of-road fixes
- lane-change rewrites
- planner changes
- new environment diagnostics

- [ ] **Step 5: Re-run the same 10-episode command after tuning**

Expected: thresholds met or clear documented residual gap.

- [ ] **Step 6: Commit final runtime-backed tuning if applicable**

```bash
git add expert_dataset/hierarchical_expert/rear_end_guard.py
git commit -m "tune: improve rear-end guard runtime performance"
```

---

## Notes for Implementers

- Keep all new logic inside `expert_dataset/hierarchical_expert/`
- Do not edit `expert_dataset/expert_idm_policy.py`
- The guard must never make acceleration less conservative than IDM
- Preserve steering behavior exactly
- If real runtime acceptance passes early, stop and report before moving on to out-of-road work
