# TruckSim execution handoff v1

This handoff is the CF-2 boundary between the user-owned Windows simulation and
the Linux chassis-surrogate project. The Linux repository does not implement or
invoke the Windows controller or TruckSim solver.

## Generate the executable example

From the `chassis-fusion` worktree:

```bash
python -m chassis_execution.trucksim_fixture --output /tmp/trucksim-export-fixture
python -m chassis_execution.verify_trucksim_export --root /tmp/trucksim-export-fixture
```

The generated `signal_mapping.json` is the authoritative executable example.
Copy it to the Windows collection root, replace source field names and unit
conversions with the real TruckSim/controller channels, then recalculate its
canonical SHA256 and place that value in `export_contract.json` and every
`run.json`.

## Windows responsibilities

- Execute exactly the `tau_cmd.npy` trajectory; never substitute raw `tau_d`.
- Run leader, middle and rear synchronously in one run and preserve role order.
- Log state boundaries from `t=0` through at least `t=4.0s` without gaps.
- Log the final controller outputs applied to the simulation, not requested
  acceleration or unprocessed optimizer values.
- Export actual TruckSim world pose, chassis signals, road-wheel angle and
  native rollover/LTR. Missing signals must fail collection and must not be
  zero-filled.
- Record actual project parameters and hashes for `.sim`, solver DLL,
  controller config and vehicle config.
- Write all payload files before setting `run.json.status` to `complete`, then
  generate `files.sha256.json` last.

## Time semantics

`raw_time_s.npy[k]` is a state boundary shared by all three vehicles.
`raw_controller_log[:, k]` is the control beginning at that boundary. The Linux
converter linearly interpolates state signals to `0.1..4.0s`, unwraps heading
before interpolation, and stores for each target state the left-limit control
that produced that state. Extrapolation and interpolation across missing raw
samples are forbidden.

## Native rollover requirement

`chassis_state.rollover_index` must bind a native TruckSim rollover/LTR export,
set `native=true`, and normalize it to dimensionless `[-1,1]`. Values inferred
from roll angle or lateral acceleration are not accepted.

## Real smoke gate

Set collection `purpose` to `real_smoke` and provide all eight maneuver labels
listed in `schemas/trucksim_execution_export_v1.json`. The verifier then checks
longitudinal actuation, lateral/chassis excitation, both controller modes,
component identity and direct conversion to `ChassisExecutionTarget`.

CF-3 may begin only after a real Windows smoke root passes this verifier.
