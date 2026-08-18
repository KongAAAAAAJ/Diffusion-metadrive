---
name: supplement-bev-dataset
description: Safely collect a targeted BEV dataset supplement, verify accepted episodes, and curate them into an existing joint-BEV/RiskEntry bundle with immutable source fingerprints, physical staging, atomic replacement, rollback, and auditable evidence. Use when a user asks to 补采、追加、合并或重平衡某场景/行为类别的episode并纳入大数据集. Do not use for ordinary standalone collection or model training.
---

# Supplement BEV Dataset

Treat supplement collection and dataset curation as two separate hard gates. Never write
supplement episodes directly into the destination bundle.

## 1. Establish the boundary

1. Read `AGENTS.md`, `.agents/skills/simple-code/SKILL.md`, and
   `docs/project/project_state.json` completely.
2. Inspect Git HEAD/worktree, running collectors or trainers, source roots, writer locks,
   pending transactions, available disk space, and filesystem boundaries.
3. Preserve unrelated edits and active artifacts. Separate code preparation from every
   long collection or copy run.
4. Freeze a machine-readable round contract containing:
   - target scenario and exact behavior category;
   - accepted episode and split quotas;
   - samples per episode and physical/safety acceptance predicates;
   - attempt limit, early-stop gate, seeds, and parallel worker count;
   - supplement root, destination root, and expected source fingerprints/index hashes;
   - curation selection/removal policy and formal-eligibility outcome.
5. Stop for user direction if any of these choices is missing or would change scenario,
   expert, safety, split, or training semantics.

## 2. Collect into an isolated bundle

1. Use a fresh supplement root and the existing collector/ordered-writer architecture.
2. Run parallel rollout workers only through the collector's supported concurrency path.
   Keep one ordered writer responsible for base, sidecar, and bundle transactions.
3. Count real rollouts, not scheduler skips, toward attempt gates. Commit base data only
   after the target category and all physical/safety predicates pass; retain rejected
   attempts as sidecar evidence when the contract requires it.
4. On an early-stop failure, maximum-attempt exhaustion, user stop, or pending transaction,
   stop cleanly and report it. Never relabel failures or edit code/configuration during the
   run.

## 3. Freeze the curation plan

After all writers stop:

1. Run the strict base, sidecar, and bundle verifiers on every source.
2. Bind the plan to each source fingerprint, bundle-index SHA256, scenario-contract SHA256,
   and selected episode metadata/payload hashes.
3. Select only base-committed episodes with complete matching sidecars and exact target
   evidence. Exclude pending and sidecar-only attempts unless the plan explicitly preserves
   them as failure evidence.
4. Freeze deterministic target episode IDs and splits. Check uniqueness by
   `(scenario_id, spawn_seed)` because seeds may intentionally repeat across scenarios.
5. Compute expected episode/sample/split/category counts and a canonical curation payload.
   Derive a new base fingerprint from that payload and then the bound sidecar fingerprint.
6. Present a dry-run manifest before any dataset write. Reject source drift, unknown pending
   state, symlinks, insufficient space, or an existing staging/backup path.

## 4. Build a physical staging bundle

1. Create a fixed-name sibling staging directory on the same filesystem as the destination.
2. Copy payload files physically; do not use hardlinks or symlinks.
3. Preserve each episode's real source scenario contract. Rewrite only the declared target
   episode index/split, new dataset bindings, and explicit curation provenance.
4. Regenerate component manifests, collection state, bundle index, payload inventory, and
   curation manifest. Represent removed or unused index positions explicitly; never fabricate
   committed data.
5. Keep formal training eligibility false unless a separate formal acceptance contract is
   fully measured and passed.

For the frozen S5 44.5k case, `tools/rebalance_s5_dataset.py` is the specialized reference
implementation. Do not apply its hard-coded IDs or hashes to another curation round.

## 5. Verify before installation

Require the staging bundle to pass all of the following:

- exact base/sidecar/index and split counts;
- exact target behavior counts and per-episode physical evidence;
- complete base-sidecar joins and declared mixed-contract provenance;
- finite arrays with required shapes/dtypes;
- copied payload SHA256 equality;
- no duplicate `(scenario_id, spawn_seed)`, pending transaction, symlink, or fingerprint
  mismatch;
- one read-only visualization smoke for a newly added episode when sidecar trajectories exist.

Do not weaken the ordinary single-contract verifier. Use a dedicated curated verifier with an
exact source-contract allowlist.

## 6. Install and clean up safely

Perform this stage only with explicit user authorization for in-place replacement:

1. Acquire non-blocking writer locks and recheck all source hashes.
2. Atomically rename destination to the fixed backup path.
3. Atomically rename staging to the destination path.
4. Run the same full verification again at the final path. On failure, move the failed new
   root aside and atomically restore the backup.
5. Release every writer/source lock before considering backup deletion. This is required on
   FUSE/external filesystems, where deleting an open lock file can create hidden files and
   leave a partial backup.
6. Delete the exact verified backup only when the user explicitly authorized permanent
   deletion. Re-resolve the path, require the expected sibling name, enumerate it, reject
   links, and use one normal deletion attempt.
7. If deletion fails, stop immediately and report the remaining path and size. Do not retry,
   broaden the target, switch shells, or use a stronger deletion primitive without new user
   direction.

## 7. Record evidence

Write the curation report first, then update `docs/project/project_state.json` last. Record
source and final fingerprints/hashes, selection and mapping, measured counts, verifier and
visualization results, eligibility, atomic-swap/rollback outcome, and truthful backup status.
Run `git diff --check` and report unrelated worktree changes separately.

## Hard stops

Stop rather than improvise when a writer is active, a source hash drifts, selected physical
evidence fails, disk space is insufficient, a pending transaction is ambiguous, staging or
backup already exists, final verification differs from staging, rollback fails, or deletion
fails.
