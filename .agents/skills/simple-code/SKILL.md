---
name: simple-code
description: Enforce the repository's mandatory pre-change gate, minimal implementation discipline, verification loop, concise logging, and one-hard-gate-per-thread handoff. Use before modifying any source code, tests, configuration, scripts, build files, dependencies, interfaces, or architecture in this repository, including bug fixes, refactors, cleanup, scenario changes, and training or evaluation implementation.
---

# Simple Code

Apply this skill before the first edit. Keep it active through implementation, verification, and handoff.

## 1. Pass the pre-change gate

Before modifying files, state a compact gate record in commentary:

1. Define the concrete problem and observable success criteria.
2. List the minimum files or responsibilities expected to change.
3. Separate verified facts from unknowns. Inspect relevant code, contracts, documentation, type definitions, and existing dependencies instead of guessing.
4. Expose unresolved confusion and material tradeoffs. Stop for user direction only when a choice would materially change scope or behavior.
5. Name the focused tests, runnable smoke, and hard gate that will prove completion.
6. Confirm that no collection, training, or other long-running job conflicts with the planned edits.

Do not edit until this record is complete. If a frozen contract conflicts with the request, report the conflict; do not hide it behind compatibility code.

## 2. Design the smallest durable solution

- Solve only the current requirement. Do not add speculative features, configuration, extension points, indirection, or abstractions.
- Build the smallest end-to-end runnable path first. Add layers only after the existing product path is stable and a current requirement needs them.
- Preserve clear module responsibilities. Do not create a helper, utility class, wrapper, or abstraction for a one-time operation.
- Prefer capabilities already present in the repository or its mature dependencies. Read their documentation and types before concluding they are insufficient or adding a dependency.
- Use established library patterns and conventions when they reduce total complexity. For architectural decisions with lasting impact, inspect how mature maintained systems solve the same problem before choosing a design.
- Do not optimize for backward compatibility. Remove obsolete paths, shims, fallbacks, migrations, and their dead tests or documentation when they are superseded by the requested behavior.
- Do not introduce a temporary design that is already expected to be replaced. If the durable minimum is blocked, expose the blocker instead of installing a workaround.

## 3. Implement without defensive excess

- Modify only files required by the success criteria. Clean up only problems introduced by the current change.
- Trust internal code and framework invariants. Validate only at system boundaries such as user input, external APIs, network responses, persisted artifacts, and frozen machine contracts.
- Fail fast on violated assumptions.
- Never add broad exception handling, silent defaults, swallowed errors, `rescue nil`, or fallback behavior for impossible internal states.
- Keep contracts strict where the project workflow requires shapes, dtypes, hashes, fingerprints, eligibility, or scenario identities.
- Remove a deprecated path directly instead of routing through a compatibility layer.

## 4. Verify until the criterion passes

1. Run the smallest relevant unit or static check first.
2. Run the minimum end-to-end smoke.
3. Run the current scenario or hard-gate acceptance command when required.
4. If verification fails, diagnose the actual failure, make the smallest corrective edit, and repeat. Do not weaken the gate or change its metadata.
5. Run `git diff --check` and inspect concise worktree status before handoff.

Treat code tests as necessary but not sufficient for dataset, checkpoint, scenario, or model eligibility. Require the authoritative evidence gate.

## 5. Control logs and tool usage

- Redirect complete long-run output to an artifact log file outside Git.
- Poll long simulations infrequently and only at meaningful milestones. Read a compact progress line, failure summary, and the final 40-80 log lines; do not repeatedly read the full log.
- Inspect JSON with focused projections and inspect diffs or code with targeted ranges. Do not repeatedly emit complete JSON, full diffs, or hundreds of code lines.
- At thread start, select low or medium reasoning for monitoring, waiting, and simple evaluation. Reserve high reasoning for root-cause diagnosis. Reasoning effort cannot be switched reliably mid-thread, so split a diagnosis into a new thread when the required effort changes materially.

## 6. Stop at one scenario or hard gate

Treat completion of one scenario or one hard gate as the thread boundary:

1. Write the authoritative report and update project state only after evidence exists.
2. Provide a compact handoff containing outcome, changed files, tests, artifact/report paths, remaining risk, and next permitted action.
3. End the thread. Do not start the next scenario or hard gate.
4. In the new thread, load only `AGENTS.md`, the applicable skills, the latest report and state, and files directly relevant to the next gate.

Do not use this boundary to abandon a failing gate. Continue iterating within the current gate until it passes or a genuine blocker requires user authority.
