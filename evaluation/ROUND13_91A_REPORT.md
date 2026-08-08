# Round 13.91a — S6/S8 proxy/closed-loop semantics

## Outcome

Round 13.91a fixed the observed S6/S8 proxy/closed-loop semantic errors, but
the full Round 13.91 gate remains blocked.  Variant B development contains one
dynamics-valid S5 brake/recovery trajectory whose closed-loop longitudinal P95
is above the unchanged 1.0 m limit.  Holdout and GRPO were therefore not
started.

No report metadata was edited to manufacture a pass, no tracking envelope was
clipped to bypass a gate, and no road or 5 m/7 m safety boundary was relaxed.

## Implemented fixes

- S6/S8 calibration sampling now waits until every hazard-defining background
  recipe has physically spawned.  This prevents an actor absent from the proxy
  snapshot from appearing later inside the four-second simulator branch.
- Scenario summaries expose real recipe counts and completion state.
- Background prediction follows the actor's complete navigation checkpoints
  through junctions and holds at a true finite route terminal instead of
  falling back to unconstrained world-frame motion.
- GroundTruthIDM/IDM prediction includes candidate-reactive braking and
  route-required adjacent-lane occupancy when the navigation road loses lanes.
- The fixed-world branch reference now includes the vehicle's actual initial
  speed.  Its first acceleration is `current speed -> first 0.5 s segment`, not
  the unrelated `first segment -> second segment` gradient.
- Proxy reports include minimum background/platoon gaps so false-safe causes
  are attributable without changing reward values.
- Controller-quality diagnostics distinguish independent trajectory tracking
  from formation-locked follower gap tracking and only use closed-loop-safe
  groups to establish a tracking envelope.

## Evidence

Preflight:

```text
/tmp/bev-round13-91a-preflight-final.json
passed: true
S5--S9, seeds [17,23], 3/3 states per episode
```

Variant A development after the initial-speed fix:

```text
/tmp/bev-round13-91a-final-A-development.json
groups:                    30
informative groups:        24
mean group Spearman:       0.76194
pairwise agreement:        93.55%
false-safe:                0
tracking lateral P95:      0.11045 m
tracking heading P95:      0.02634 rad
max state longitudinal P95 0.31741 m
longitudinal blockers:     none
```

Variant B development after the same fix:

```text
/tmp/bev-round13-91a-final-B-development.json
groups:                    30
informative groups:        19
mean group Spearman:       0.79762
pairwise agreement:        93.98%
false-safe:                0
tracking lateral P95:      0.11488 m
tracking heading P95:      0.03110 rad
max state longitudinal P95 1.40545 m
blocker:                   longitudinal_p95
```

The blocker is `S5_hard_brake_lead`, seed 17, state 2, rear role.  Only group 1
is closed-loop safe.  The reference commands a strong brake followed by the
configured +1 m/s2 recovery; the actuator follows without saturation, but the
vehicle accumulates about 1.48 m of positive arc error during the transition.

One evidence-driven position-feedback experiment (`position_kp 0.15 -> 0.30`,
still bounded at +/-0.5 m/s2) reduced the maximum state P95 to 1.26184 m but did
not pass:

```text
/tmp/bev-round13-91a-final2-B-development.json
false-safe:                0
max state longitudinal P95 1.26184 m
blocker:                   longitudinal_p95
```

That unsuccessful gain change was reverted.  No second controller tuning was
attempted.

An earlier A holdout run is retained only as diagnostic evidence because it
preceded the corrected development rerun:

```text
/tmp/bev-round13-91a-A-holdout.json
Spearman:                  0.74169
pairwise agreement:        91.95%
false-safe:                0
blocker:                   S6 seed31 longitudinal_p95 (1.03606 m)
```

The fixed initial-speed reference removes that S6 development error (A maximum
state P95 becomes 0.31741 m), but a new A holdout was intentionally not run
after Variant B failed development.

## Stop condition

Round 13.91 is not accepted.  Variant B did not pass development, so the
following were deliberately not run:

- corrected A/B holdout calibration on seeds `[31,47]`;
- online GRPO updates;
- four-model evaluation.

The next change should address executable reference shaping for abrupt
brake-to-recovery transitions (for example a jerk/actuator-lag-aware temporal
profile), then restart 13.91 from development.  It must not weaken the 1.0 m
tracking gate or convert the measured error into an envelope exemption.
