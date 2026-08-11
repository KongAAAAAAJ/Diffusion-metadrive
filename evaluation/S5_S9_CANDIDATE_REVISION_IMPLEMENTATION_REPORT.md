# S5--S9 candidate-v2 implementation report

Date: 2026-08-11
Branch: `s5-s9-scenario-revision`
Worktree: `/tmp/diffusion-metadrive-s5-s9-revision`

## Implemented

- Deterministic constrained sampler keyed by `spawn_seed + scenario_id + route`, including severity buckets and correlated parameters.
- Extended episode evidence: resolved parameters, actor manifest, conflict evidence, route completion, functional success, and RuleMaker profile.
- S5 middle-lane spawn, two real adjacent actors, random ahead/behind relations, hard-brake ranges, and three RuleMaker profiles.
- S6 deterministic two-gap selection and conflict-time reverse spawn solve.
- S7 atomic 4--6 actor recipe with critical/next gap roles and delayed topology-gated merge decision.
- S8 atomic exit-gap actors and right-lane/connector KEEP gating.
- S9 lane-1 blocker, lane-0 constraint actor, topology-gated LEFT, and sequential agent0/1/2 bypass to avoid joint sweep conflicts.
- Static v1 contract snapshot preserving SHA256 `e70b07d7e73d6969f90417441f983b49ecd6bfe48003be310fde5860e24e8e34`.
- Candidate-v2 contract generator; it becomes frozen only when the artifact runner passes every hard gate.
- Sequential 35-episode artifact runner with MP4/NPZ/JSON existence, non-empty, decode, and SHA256 validation.

## Verification performed

- Focused regression suite: `152 passed`.
- Real MetaDrive smoke tests with seed 17, zero background density and three ego vehicles:
  - S5: both adjacent actors are present on lane 0/lane 2 when ego starts on lane 1; 40/40 planning was previously reached before the functional gate fix.
  - S6: gap intruder is atomically spawned from the reverse timing solve; 5/5 planning.
  - S7: all six high-severity actors are atomically spawned; delayed merge preparation gives 20/20 planning.
  - S8: front/rear exit-gap actors are atomically spawned; 5/5 planning.
  - S9: blocker and target-lane actor are atomically spawned; sequential LEFT preparation gives 15/15 planning and avoids the earlier agent0/agent1 joint-trajectory conflict.

## Gate state

The 35 x 200-step video evaluation has **not** been declared successful and v2 remains **unfrozen**. Short smoke tests establish actor and planning-chain viability but do not prove route completion. Run:

```bash
python -m evaluation.run_s5_s9_candidate_revision \
  --output-root outputs/s5_s9_candidate_revision_eval_20260811
```

The runner writes `artifact_manifest.json` and exits non-zero on any functional, planning, collision, route, profile-diversity, or artifact-integrity failure. Project memory/state must only be advanced to “five-seed validation complete / v2 frozen” after that manifest reports `hard_gates_passed: true`.

The final S9 seed-17 exact-path pilot produced decodable artifacts but failed
after 24 successful planning steps with
`committed_trajectory_tracking_deviation`.  It is retained under the isolated
`pilot_final/` directory as failure-analysis evidence and is not counted as an
accepted evaluation episode.  Earlier trigger/horizon/joint/rolling-horizon
experiments also failed existing hard feasibility gates and their unvalidated
strategy changes were reverted.  The NormalPlanner tracking envelope was
deliberately not relaxed.

Failure-evidence artifact SHA256 values:

- top-down MP4: `2852dc51dd4fa473768e91a05309e0af8d8392c790a77caa013312a9fb385b11`
- semantic-BEV MP4: `e5c12776a45e0498716cbc5eb8ef6fe1b565729e3a14751f103aae7bd4f3cdf5`
- exact trajectory NPZ: `2d039c477b3aa6e8d11e0ac48cbd5c0529906475c3d3f765d6f4f8085befb74b`
