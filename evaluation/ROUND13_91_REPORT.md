# Round 13.91 — S5–S9 reward calibration gate

## Result

Round 13.91 is **blocked at the development calibration gate**.  Rounds
13.92 (A/B online GRPO), 13.93 (four-model evaluation), and Round 14 cleanup
were intentionally not started.

The reward contract now treats the already-frozen safety distances as hard
constraints:

- background bumper gap below 5 m is unsafe;
- platoon bumper gap below 7 m is unsafe;
- collision, out-of-drivable, and clearance violations remain separately
  observable in `JointRewardResult`.

This change reduced development false-safe counts, but did not eliminate them.

## Development calibration

Inputs:

- scenarios: the complete ordered S5–S9 contract;
- seeds: 17 and 23;
- three history-ready states per episode;
- four joint candidates per state;
- deterministic kinematic trajectory optimizer before proxy/simulator scoring.

| Metric | Variant A | Variant B | Gate |
|---|---:|---:|---:|
| informative groups | 22 | 22 | >= 15 |
| mean group Spearman | 0.6642 | 0.6642 | >= 0.50 |
| pairwise agreement | 90.67% | 90.67% | >= 70% |
| false-safe count | 6 | 7 | **0** |
| lateral tracking P95 | 0.2772 m | 0.2772 m | <= 0.5 m |
| heading tracking P95 | 0.0745 rad | 0.0746 rad | <= 0.1 rad |

The ordinary P95 tracking gate passes, but the conservative-envelope
construction is not admissible:

| Tracking P99 | Variant A | Variant B | permitted envelope cap |
|---|---:|---:|---:|
| longitudinal | 3.0436 m | 3.0436 m | 1.5 m |
| lateral | 0.4571 m | 0.4571 m | 1.0 m |
| heading | 0.1794 rad | 0.1794 rad | 0.15 rad |

Therefore the holdout CLI correctly refuses to derive an envelope from this
development report.

## Remaining false-safe distribution

- Variant A: four controller-tracking cases and two termination-semantics
  cases.
- Variant B: four controller-tracking cases and three termination-semantics
  cases.
- Common controller cases are concentrated in S6 and S8.
- Common termination cases have real branch road clearance below zero or a
  real background gap below 5 m, while the open-loop proxy remains safe.

The full evidence is stored in:

- `/tmp/bev-round13-91-calibration-A-development.json`
- `/tmp/bev-round13-91-calibration-B-development.json`

## Stop decision

The next admissible task is a separate evidence-driven correction of the S6
and S8 proxy-versus-tracking discrepancy and the remaining road-termination
semantics.  It must then rerun development calibration from scratch.  It is
not valid to cap the measured envelope silently, edit report metadata, proceed
to holdout, or launch GRPO with these reports.
