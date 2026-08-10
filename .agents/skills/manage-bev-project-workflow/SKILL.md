---
name: manage-bev-project-workflow
description: Orchestrate repository work for the BEV-only platoon diffusion project by loading project instructions and machine-readable state, selecting the relevant specialist skill or frozen runbook, enforcing hard gates, and updating state and evidence after execution. Use for implementation rounds, data collection, dataset verification, Stage 1 or GRPO training, checkpoint qualification, S5-S9 evaluation, cleanup, release, or project-status work in this repository. Do not use for a purely conceptual question that does not inspect or change project state.
---

# Manage BEV Project Workflow

Execute repository work through one state-driven, auditable workflow. Treat this skill as the orchestration layer; keep data, training, GRPO, and evaluation details in their specialist skills, frozen contracts, and runbooks.

## 1. Load instructions and state

1. Confirm the repository root and read `AGENTS.md` completely before acting.
2. Read `docs/project/project_state.json` completely.
3. Require the state file to identify at least:
   - schema version and update time;
   - current phase, round, and status;
   - active hard gate and blockers;
   - source-of-truth contracts and reports;
   - datasets and checkpoints with fingerprints and eligibility;
   - ordered next actions.
4. Treat `project_state.json` as an index, not a substitute for its referenced contracts. Read every referenced source required for the current task.
5. If the state file is absent, malformed, stale against Git, or contradicts a frozen contract, stop implementation. Report the mismatch and repair the state index from authoritative evidence before continuing. Never guess eligibility or rewrite metadata to bypass a gate.

## 2. Route the task

1. Classify the request as exactly one primary workflow: data collection, dataset verification, Stage 1 training, reward calibration, GRPO training, evaluation, semantic-preserving cleanup, or project administration.
2. Select the matching repository skill from `.agents/skills/` when it exists and read its `SKILL.md` completely.
3. When no specialist skill exists, use the frozen contract and runbook referenced by `project_state.json`; record the missing specialist skill as a maintainability gap without expanding the current task to create it.
4. Use only the minimum set of skills needed. State the selected skill order in commentary.
5. Reject work that violates the required phase order, consumes an ineligible checkpoint, mixes dataset fingerprints, modifies a frozen scenario or protocol during collection, or combines long-running collection/training with code changes.

## 3. Establish the execution boundary

Before modifying anything:

1. Inspect Git branch, HEAD, worktree status, active long-running jobs, and relevant artifact paths.
2. Preserve unrelated user changes and active collection/training outputs.
3. Resolve the current round's single completion boundary, mandatory tests, minimum runnable command, and stop conditions from the authoritative contract.
4. Run a short smoke before any long task. During a long task, only monitor logs, throughput, hardware use, checkpoints, and integrity signals; do not modify code or configuration.
5. If a hard gate fails, stop at that gate and report evidence. Do not relax safety limits, change scenario semantics, add compatibility fallbacks, or edit report metadata unless the user explicitly authorizes a new round that changes the contract.

## 4. Execute and verify

1. Make the smallest change that satisfies the selected workflow and current round.
2. Keep machine contracts strict: validate shapes, dtypes, finite values, hashes, fingerprints, eligibility flags, and scenario identities at boundaries.
3. Run focused unit tests first, then required integration or scenario gates.
4. Run `git diff --check` and inspect the final worktree.
5. Keep generated datasets, checkpoints, videos, and large logs outside Git; record their absolute paths and immutable identifiers in artifact metadata.
6. Never mark a round, dataset, checkpoint, or model eligible merely because code tests pass. Require every stated acceptance gate and evidence artifact.

## 5. Record the result

After execution, update the project record in this order:

1. Write or update the round's report under `evaluation/` with commands, measured results, failures, artifact paths, hashes, and remaining risks.
2. Update the relevant artifact manifest or frozen contract only when the task authorizes it.
3. Update `docs/project/project_state.json` last, after the evidence exists. Record:
   - new Git HEAD or explicit uncommitted state;
   - completed and failed gates;
   - dataset/checkpoint fingerprints and formal eligibility;
   - active blockers;
   - the next ordered action;
   - links to authoritative reports rather than duplicated narrative.
4. Validate JSON syntax and referenced paths. Run `git diff --check` again.
5. Do not store transient PIDs, secrets, verbose logs, or unsupported conclusions in project state.

## 6. Report to the user

End every execution with:

- outcome and whether the hard gate passed;
- files or modules changed;
- tests and measured results;
- minimum command to reproduce the result;
- artifact and report paths;
- remaining risks and the next permitted action;
- `git diff --check` result and concise worktree status.

If blocked, identify the exact failed gate, evidence, and required user decision. Do not describe incomplete work as accepted.
