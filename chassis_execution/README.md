# Chassis execution surrogate boundary

This package owns the independent `tau_cmd -> tau_a` machine contract. It is
deliberately separate from the BEV diffusion planner and from the online GRPO
trainer.

CF-0/CF-1 provide only strict tensors, dataset identity, deterministic run-level
splits, ensemble aggregation, and synthetic fixtures. Neural models, TruckSim
export, loss functions, training, and GRPO integration belong to later rounds.

The surrogate receives only trajectories produced by
`KinematicTrajectoryOptimizer`. Raw diffusion `tau_d` remains owned by the
policy probability model and is rejected at this boundary.
