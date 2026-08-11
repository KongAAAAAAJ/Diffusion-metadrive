---
name: recall-bev-project-plan
description: Read, explain, review, or update the BEV platoon project's current machine-readable research plan and active hard gate. Use when the user says 项目计划, 最新计划, 后续计划, 实验计划, 训练计划, 项目路线图, 当前主线, 下一步是什么, Stage 1 计划, Stage 2 计划, 双域融合计划, GRPO 对比, or GRPO 消融, or asks which formal training/evaluation work remains. Do not use for implementing an already specified task unless plan or status context is also requested.
---

# Recall BEV Project Plan

Use this skill to answer plan and roadmap questions from the project's frozen machine-readable memory instead of reconstructing an outdated plan from conversation history.

## Read the Sources

1. Read `docs/project/project_plan.json` completely. Treat it as the authoritative research-plan memory.
2. Read `docs/project/project_state.json` completely. Use it to determine which artifact and hard gate are current.
3. Read an evidence file referenced by `project_state.json` only when the user's question requires its exact result, fingerprint, path, or gate outcome.
4. If the plan and state conflict, report the mismatch. Do not silently choose an older A/B, selector, or four-model route.

## Answer Plan Queries

For a general query such as “项目计划” or “下一步是什么”, return:

1. The current paper question and the three trajectory domains `tau_d`, `tau_cmd`, and `tau_a`.
2. The active hard gate and the immediately executable next action.
3. The Stage 1 formal scope: train Variant A only.
4. The Stage 2 comparison: `GRPO-Open` evaluates reward on `tau_cmd`; `GRPO-Exec` evaluates reward on surrogate-predicted `tau_a`.
5. The final three-model matrix and remaining ordered gates.
6. Any real blocker, especially the real-TruckSim surrogate gate.

Keep the response proportional to the request. Link the local memory and state files when useful.

## Preserve the Frozen Semantics

- Do not present Stage 1 Variant B as required formal training, a main baseline, or a final model.
- Do not restore MAPPO selector, camera/LiDAR input, teacher forcing, or sequential-role optimizer paths.
- Both Stage 2 branches must start from the same eligible Stage1-A checkpoint and use the same data, seeds, budgets, optimizer, KL/BC weights, reward components, and trajectory optimizer.
- The primary Stage 2 variable is only the reward domain: `R(tau_cmd)` versus `R(tau_a)`.
- GRPO probability and gradients remain tied to raw diffusion rollout `tau_d`.
- The chassis surrogate consumes `tau_cmd`, stays frozen during GRPO, and is not part of reward backpropagation.
- The G candidates are not separately executed in MetaDrive. Select one `tau_cmd`, then call `env.step()` exactly once.
- Synthetic surrogate data is engineering evidence only. Formal `GRPO-Exec` and paper conclusions require the real TruckSim fit, ID/OOD, and calibration gates.
- Round 14 cleanup starts only after both Stage 2 branches are semantically stable.

## Update the Plan

When the user explicitly changes the research plan:

1. Use `$manage-bev-project-workflow` and inspect the current Git state before editing.
2. Update `docs/project/project_plan.json` first.
3. Update the matching active gate, checkpoint matrix, next actions, and source-of-truth reference in `docs/project/project_state.json`.
4. Add or update an evidence report under `evaluation/` when the revision is material.
5. Validate JSON, run the plan consistency checks, run `git diff --check`, and report the working-tree state.

Do not change a frozen plan merely because implementation is inconvenient. A semantic revision requires explicit user direction.
