# Round 13.96: Round 13 Infrastructure Closeout

## Verdict

Round 13 is accepted as an **infrastructure and training-chain closeout**.

This acceptance does not convert any diagnostic checkpoint into a formal one and
does not claim policy superiority.  The user-approved same-seed exception is
recorded explicitly: closed-loop MetaDrive/Bullet trajectories are no longer
required to be bit-exact across repeated runs.  Initial scene equality,
identical-input model determinism and formal multi-seed statistics remain hard
requirements.

## Accepted scope

- frozen S5--S9 expert/data contract;
- joint-first schema-v2 bit-packed semantic BEV chain;
- ResNet18.a1 BEV-only Stage 1 A/B training chain;
- deterministic execution-time trajectory optimizer;
- proxy--simulator development and holdout calibration chain;
- online A/B joint GRPO training and checkpoint chain;
- four-model evaluator, S9 final-mode execution mask and semantic-BEV latency;
- strict refusal to treat diagnostic checkpoints as formal artifacts.

## Explicit non-claims

- Current diagnostic A/B/GRPO checkpoints are not eligible for formal training
  or paper conclusions.
- Diagnostic collision, gap and formation metrics do not establish model
  superiority.
- The complete planning tick remains above 100 ms P95 (`125.14 ms` maximum in
  the accepted diagnostic evidence).  Model inference (`20.84 ms` maximum P95)
  and semantic-BEV construction (`89.97 ms` maximum P95) satisfy their frozen
  component limits.  Full-tick optimization is non-blocking for offline formal
  data collection/training but remains visible for final deployment evaluation.

## Machine-verifiable freeze

The manifest `evaluation/ROUND13_96_FREEZE.json` is generated and verified by:

```bash
python -m evaluation.round13_closeout --write
python -m evaluation.round13_closeout --verify
```

Frozen hashes:

```text
S5--S9 scenario contract  e70b07d7e73d6969f90417441f983b49ecd6bfe48003be310fde5860e24e8e34
semantic BEV codec       a762933f0baac58ba02a37bc6a4e4ae06687fea446f43b0889f4798720f0a8f6
hard mode mask           9346cddbf5021c2100c2127cdf6ce30c7bc51bf50cbaf4e6d146a03da7243fbc
trajectory optimizer     08dac4861f34443ece55b0f8468ab93aef93103e24e10bbdfd56b76d42aa69bd
joint reward             bd6e726135082d9e3ff983b2ce0fbf55021479c678d731250395341ca93cd7c1
```

The manifest also freezes storage schema v2, Stage 1 checkpoint schema v2,
GRPO checkpoint schema v1, all contract configs and SHA256 values for the
Round 13.89/13.91/13.92/13.94/13.95 evidence reports.  Any code or evidence
drift causes `--verify` to fail.

## Formal eligibility boundary

The next hard gate is the 15,000-step formal joint-BEV pilot.  The required
sequence is frozen as:

```text
formal 15k pilot + full verifier
  -> formal full-scale joint-BEV dataset
  -> Stage 1 A/B run_mode=formal
  -> fresh A/B reward calibration
  -> A/B GRPO run_mode=formal
  -> semantic-preserving Round 14 cleanup
  -> formal multi-seed S5--S9 four-model evaluation
```

Old diagnostic data/checkpoints must not be mixed into the formal chain, and
metadata editing cannot confer eligibility.

## Luna worker configuration

The requested file exists at:

```text
/home/kong/.codex/agents/luna-worker.toml
```

Validated fields:

```text
name                    luna-worker
model                   gpt-5.6-luna
model_reasoning_effort  max
description             non-empty
developer_instructions  non-empty (857 characters)
```

Validation consisted of:

1. Python `tomllib` strict parse and required-field assertions;
2. Codex `doctor` config load (`config.load: ok`) and stable multi-agent feature
   detection;
3. actual `luna-worker` read-only delegation, which loaded the frozen manifest
   and correctly returned `infrastructure_accepted`, the scenario SHA256 and
   `current_checkpoints_eligible=false`.

`codex doctor` also reported unrelated environment problems (restricted-network
reachability and an existing memories database access failure).  They do not
invalidate the TOML parse or successful worker delegation and were not modified.

The file already contained the requested exact configuration at the start of
Round 13.96, so it was preserved rather than rewritten solely to change its
timestamp.  A creation-style diff against `/dev/null` is included in the user
handoff.

## Verification

```text
round13 closeout tests: 2 passed
freeze write/read/verify: passed
git diff --check: passed
```

## Next action

Do not start broad Round 14 deletion/refactoring yet.  Start the formal
15,000-step dataset pilot against this frozen manifest.  Round 14 remains after
formal Stage 1 and GRPO checkpoints exist and before the final formal evaluator.
