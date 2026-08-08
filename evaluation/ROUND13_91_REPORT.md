# Round 13.91 — S5–S9 reward calibration gate

## Result

Round 13.91 is **accepted**.  The frozen S5–S9 preflight, both development
calibrations, and both independent holdout calibrations pass without changing
the 5 m/7 m safety distances, the 0.1 rad tracking envelope, or calibration
metadata.

The accepted execution boundary is:

```text
raw Stage 1 / GRPO trajectory
  -> deterministic kinematic trajectory optimizer
  -> actuator-lag/jerk-aware temporal profile when brake release is abrupt
  -> proxy reward and independent simulator branch
```

Raw GRPO rollouts remain immutable.  The optimizer configuration SHA256 used
by all four calibration reports is
`08dac4861f34443ece55b0f8468ab93aef93103e24e10bbdfd56b76d42aa69bd`.

## Preflight

- scenarios: complete ordered S5–S9 contract;
- seeds: `[17, 23]`;
- three history-ready states per episode;
- all ten episodes passed;
- S5 seeds 17/23 both report the actual hard-brake trigger, realization, and
  `2/2` completed recipes before sampling;
- maximum prefix replay position and heading errors were both zero.

Artifact: `/tmp/bev-round13-91-final-preflight.json`.

## Development calibration

Each variant used 30 joint groups (`5 scenarios x 2 seeds x 3 states`) and
four trajectories per group.

| Metric | Variant A | Variant B | Gate |
|---|---:|---:|---:|
| informative groups | 21 | 18 | >= 15 |
| mean group Spearman | 0.7352 | 0.7950 | >= 0.50 |
| pairwise agreement | 91.67% | 95.95% | >= 70% |
| false-safe count | 0 | 0 | **0** |
| lateral tracking P95 | 0.1102 m | 0.1187 m | <= 0.5 m |
| heading tracking P95 | 0.0248 rad | 0.0300 rad | <= 0.1 rad |
| maximum state longitudinal P95 | 0.2254 m | 0.2365 m | <= 1.0 m |
| maximum state longitudinal P99 | 0.4935 m | 0.4993 m | <= 1.5 m |
| maximum continuous saturation | 0 s | 0 s | <= 1.0 s |
| optimized trajectory validity | 360/360 | 360/360 | 100% |

Artifacts:

- `/tmp/bev-round13-91-final-A-development.json`
- `/tmp/bev-round13-91-final-B-development.json`

## Independent holdout calibration

Holdout uses seeds `[31, 47]`, which were not used for temporal-profile
development.  Each variant uses only its own frozen development envelope.

| Metric | Variant A | Variant B | Gate |
|---|---:|---:|---:|
| `passed` | true | true | true |
| informative groups | 19 | 15 | >= 15 |
| mean group Spearman | 0.7131 | 0.8434 | >= 0.50 |
| pairwise agreement | 90.14% | 96.77% | >= 70% |
| false-safe count | 0 | 0 | **0** |
| lateral tracking P95 | 0.1088 m | 0.1225 m | <= 0.5 m |
| heading tracking P95 | 0.0264 rad | 0.0332 rad | <= 0.1 rad |
| maximum state longitudinal P95 | 0.2254 m | 0.2365 m | <= 1.0 m |
| maximum state longitudinal P99 | 0.4935 m | 0.4993 m | <= 1.5 m |
| maximum STOP terminal speed | 0.2152 m/s | 0.1484 m/s | <= 0.3 m/s |
| maximum continuous saturation | 0 s | 0 s | <= 1.0 s |

Artifacts:

- `/tmp/bev-round13-91-final-A-holdout.json`
- `/tmp/bev-round13-91-final-B-holdout.json`

## Scope and next dependency

These are diagnostic checkpoints, so this acceptance validates infrastructure
and calibration semantics rather than paper-level performance.  Some states
still contain no closed-loop-safe candidate (A holdout: 20/30, B holdout:
19/30); these groups are correctly classified unsafe and create no false-safe,
but expose limited candidate coverage in the diagnostic Stage 1 policies.

Round 13.92 may now start A/B online GRPO using the accepted holdout reports.
Round 13.93 and Round 14 remain pending.
