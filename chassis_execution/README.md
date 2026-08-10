# Chassis execution surrogate boundary

This package owns the independent `tau_cmd -> tau_a` machine contract. It is
deliberately separate from the BEV diffusion planner and from the online GRPO
trainer.

CF-0/CF-1 provide strict tensors, dataset identity, deterministic run-level
splits, ensemble aggregation, and synthetic fixtures. CF-2 adds the strict
cross-platform exchange boundary for raw Windows TruckSim exports; it does not
run TruckSim or own the Windows controller.

The surrogate receives only trajectories produced by
`KinematicTrajectoryOptimizer`. Raw diffusion `tau_d` remains owned by the
policy probability model and is rejected at this boundary.

Generate and verify the executable CF-2 handoff example with:

```bash
python -m chassis_execution.trucksim_fixture --output /tmp/trucksim-export-fixture
python -m chassis_execution.verify_trucksim_export --root /tmp/trucksim-export-fixture
```

See `docs/contracts/trucksim_execution_handoff_v1.md` before producing a real
Windows smoke export. Neural models, formal mmap writing, loss functions,
training, and GRPO integration remain later rounds.
