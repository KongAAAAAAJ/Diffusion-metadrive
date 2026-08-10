# CF-3 diagnostic virtual dataset

CF-3 uses the original chassis-fusion round sequence. It does not create a
parallel implementation path. The user authorized one temporary diagnostic
override of the CF-2 real Windows TruckSim smoke so the storage, surrogate
training, integration and evaluation software can be exercised before the
Windows export is ready.

## Eligibility boundary

Every virtual-data artifact must contain all of the following values:

```text
data_origin = synthetic_virtual
diagnostic_only = true
eligible_for_formal_training = false
cf2_real_windows_smoke_passed = false
diagnostic_override_id = cf2_real_smoke_diagnostic_override_v1
```

These values are enforced by the storage constructor and verifier. Editing a
dataset or checkpoint manifest cannot make it formally eligible. Formal
surrogate training remains blocked until CF-2 passes with real Windows
TruckSim runs.

## Diagnostic data generation

The generator produces synchronized three-role samples for eight maneuvers.
It applies deterministic virtual response lag, mass/friction conditioning,
drive/brake delay, yaw response and roll response. This is protocol-shaped
test data, not a claim that the virtual dynamics approximate TruckSim.

```bash
python -m chassis_execution.synthetic_data \
  --output <new-empty-dataset-root> \
  --samples 2048 \
  --seed 17 \
  --report <report.json>
```

The physical dataset is immutable and split-first. Every array is an ordinary
`.npy` file that can be memory-mapped; a `run_group_id` is assigned atomically
to exactly one of train, validation or test.

## Verification

```bash
python -m chassis_execution.verify_dataset --root <dataset-root>
pytest tests/test_chassis_execution_storage.py \
       tests/test_chassis_execution_contract.py \
       tests/test_chassis_execution_trucksim_export.py -q
```

CF-4 may consume this dataset only in diagnostic mode. CF-2 remains pending.
