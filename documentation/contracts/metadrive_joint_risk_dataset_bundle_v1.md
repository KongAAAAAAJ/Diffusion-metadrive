# MetaDrive Joint Planning + Risk Dataset Bundle Protocol v1

Status: frozen for the Round 13.97 pilot implementation  
Bundle format: `metadrive-joint-planning-risk-bundle`  
Bundle version: `1.0.0`

## 1. Scope

One simulator rollout produces two linked datasets with different retention
policies:

1. the Diffusion-MetaDrive base dataset owns joint semantic BEV, planner inputs,
   hard-mode masks, and expert planning labels;
2. the RiskEntry actor sidecar owns the complete raw actor/lane/event timeline
   needed to derive risk labels offline.

The two components are a bundle, not a merged episode directory. Sidecar files
must never be added to a base episode because the base schema validates an exact
file set.

## 2. Component contracts

| Component | Format | Version | Owner |
|---|---|---:|---|
| Base | `joint-first-packed-semantic-bev-npy-episodes` | 2 | Diffusion-MetaDrive |
| Actor sidecar | `riskentry-metadrive-actor-sidecar` | 1.0.0 | RiskEntry |

The normative RiskEntry sidecar specification is
`RiskModeCodes/docs/contracts/metadrive_actor_sidecar_v1.md`, together with the
three `RiskModeCodes/schemas/metadrive_actor_sidecar_*_v1.schema.json` schemas.

The base tensor schema remains unchanged:

```text
bev                         uint8   [3,8,256,256]
ego_state                   float32 [3,8]
ego_pose_global             float32 [3,3]
formation_relation_state    float32 [3,12]
relation_valid_mask         bool    [3,2]
agent_role                  int64   [3]
coarse_trajectories         float32 [3,10,8,3]
mode_valid_mask             bool    [3,10]
gt_mode                     int64   [3]
expert_trajectory           float32 [3,8,3]
```

## 3. Bundle layout

```text
bundle_root/
├── bundle_contract.json
├── bundle_episode_index.jsonl
├── platoon_joint_bev/                 # base schema v2 root
└── riskentry_actor_sidecar/            # sidecar schema v1 root
```

The component roots retain their own `dataset_contract.json`, split manifests,
and atomic episode directories. Absolute filesystem paths are not part of an
immutable fingerprint; component directory names are relative to the bundle
root.

`bundle_contract.json` freezes:

```text
bundle format and version
base format, schema version, and dataset fingerprint
sidecar format and schema version
split assignment policy and seed
decision_dt_s
scenario_contract_sha256
actor identity policy
timeline policy
retention policy
```

## 4. Shared episode identity and split

The cross-component join key is:

```text
(base_dataset_fingerprint, episode_index, split)
```

The following provenance must also agree for the same attempted episode:

```text
scenario_id
local_route
spawn_seed
decision_dt_s
scenario_contract_sha256
```

The split is assigned once by the base episode split assigner. The sidecar must
mirror it, including for an episode rejected by the base. A sidecar must never
perform a second random split.

## 5. Actor identity

New shared datasets use exactly:

```text
agent0 -> P0 -> platoon_leader
agent1 -> P1 -> platoon_middle
agent2 -> P2 -> platoon_rear
```

External actors use stable episode-local IDs `V000`, `V001`, ... ordered by
`(first_seen_step, source_object_id)`. Their original MetaDrive registry key is
stored as `source_object_id`.

Legacy RiskEntry data or runners that use `P1/P2/P3` are not compliant with this
bundle protocol. A consumer may perform a one-time explicit legacy migration,
but a dataset must not contain both identity conventions and must not persist a
silent alias mapping.

## 6. Time and sample alignment

For decision interval `decision_dt_s`, raw sidecar state `k` is the simulator
state immediately before action `k`:

```text
step_index[k] = k
timestamp_s[k] = k * decision_dt_s
```

If an episode executes `K` actions, the sidecar stores `K+1` states, including
the terminal post-step state. An event caused by action `k` belongs to result
state `k+1`.

The base stores only history-ready joint planning samples. The explicit mapping
is:

```text
sidecar/base_sample_step_index[s] -> raw sidecar step k
```

Base sample row number must never be used to infer raw time. Diagnostic
subsampling and rejected joint labels make that inference invalid.

## 7. Data ownership and leakage boundary

Base-only fields:

```text
semantic BEV
ego and formation tensors
dynamic coarse trajectories
hard mode-valid mask
GT mode and expert trajectory
```

Sidecar-only fields:

```text
complete actor identity table and raw timeline
world pose, velocity, acceleration, yaw rate, and validity masks
lane table and per-frame lane state
scenario key actor IDs
raw trigger, collision, road, terminal, and despawn events
outcome and retention metadata
```

The sidecar does not persist TTC, THW, risk-entry edges, 1/3/5-second labels,
MLLM labels, or duplicated future trajectories. RiskEntry derives these offline
from the raw timeline.

`key_actor_ids`, future states, and outcome events are label/audit data. They
must not enter an online risk-model input at a prediction time.

## 8. Independent retention with mandatory linkage

The base and sidecar intentionally have different acceptance rules:

- The base commits only complete, native-expert episodes whose stored samples
  satisfy planner, hard-mask, GT-mode, and trajectory contracts.
- The sidecar commits collision, out-of-road, out-of-route, near-collision,
  expert-failure, terminal, and truncated episodes because they are valid risk
  outcomes.
- A base expert failure therefore normally produces a sidecar-only episode with
  an empty `base_sample_step_index`.
- A sidecar data-integrity failure (missing frame, unstable actor identity,
  missing required active state, NaN/Inf, or identity/fingerprint mismatch) is a
  bundle-level rejection. A new formal base episode must not be committed
  without its valid sidecar counterpart.

`bundle_episode_index.jsonl` contains one immutable result row per attempted
episode:

```text
episode_index
split
scenario_id
local_route
spawn_seed
base_status                 committed | rejected
base_rejection_reason       string | null
sidecar_status              committed | rejected
sidecar_rejection_reason    string | null
raw_steps
base_samples
outcome
```

The sidecar episode set is therefore expected to be a superset of the base
episode set. Every committed base episode must have exactly one committed
sidecar episode with the same join key.

## 9. Collection transaction

The simulator runs once. Collection proceeds as follows:

1. create the episode identity and mirrored split;
2. begin an in-memory/staged base episode and sidecar episode;
3. append sidecar raw state 0;
4. before each action, optionally build a history-ready base planning sample;
5. execute the expert/control action;
6. append sidecar result state and raw events at step `k+1`;
7. decide base eligibility and build `base_sample_step_index` from the exact
   accepted base sample steps;
8. validate the complete sidecar first;
9. if base is eligible, atomically commit base, then sidecar with the exact
   mapping; otherwise commit the sidecar with an empty mapping;
10. append the bundle episode status only after component commit results are
    known.

Crash or road departure must stop further actions but must not prevent capture
of the terminal result state.

## 10. Compatibility and acceptance gates

The Round 13.97 pilot must demonstrate:

1. base Dataset output remains the exact ten-field `[B,3,...]` contract;
2. base episode directories contain no sidecar files;
3. a `K`-action episode has `K+1` raw sidecar states;
4. base sample-to-raw-step mapping is exact after history warm-up,
   subsampling, and rejected joint labels;
5. P0/P1/P2 and Vxxx identities remain stable across disappearance and
   reappearance;
6. collision and road-departure episodes are retained in the sidecar while the
   base may reject them;
7. component episode index, split, base fingerprint, scenario contract, route,
   seed, and decision interval match;
8. all arrays are non-pickle `.npy`, mmap-readable, finite, and have exact
   dtype/shape and explicit validity masks;
9. missing terminal state, timestamp gaps, identity collision, missing active
   pose/velocity, invalid key actor, NaN/Inf, or fingerprint mismatch is
   rejected;
10. both component verifiers and a bundle cross-link verifier pass before a
    pilot is eligible for Stage 1 or RiskEntry preprocessing.

## 11. Implementation boundary

The protocol does not change planner model inputs, Stage 1 loss, GRPO inputs,
BEV packing, or the base data schema. Implementation work is limited to:

- parallel raw actor/lane/event capture;
- a concrete atomic RiskEntry sidecar writer;
- shared episode identity, split, fingerprint, and scenario-contract metadata;
- terminal post-step capture;
- bundle episode status and cross-link verification;
- RiskEntry consumption of P0/P1/P2 for new shared datasets.

