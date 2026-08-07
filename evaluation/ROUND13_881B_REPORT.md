# Round 13.881b — S8 G-block junction drivable geometry

## Scope

This round fixes only the missing drivable pavement between the right-most
mainline lane and the first exit bend of MetaDrive's `FreeOutRampOnStraight`
(`G`) block.  It does not widen either lane, change the route graph, relax the
XL footprint audit, or modify decision, control, reward, or diffusion code.

## Root cause

For the S8 `g0` block, the source lane and exit bend share a graph node but
their centers are one lane-width (3.5 m) apart.  Both individual lane surfaces
terminate at the seam.  The smooth route-chain centerline from Round 13.881a
is valid, but an XL vehicle straddling the seam has corners outside the union
of those two truncated surfaces.

A read-only width scan showed that changing a single lane width does not fix
the junction.  The minimum correct fix is a 3.5 m-wide pavement surface along
the existing Hermite transition; no road-width expansion is required.

## Implementation

- Added a shared lane-seam transition helper used by both route geometry and
  junction pavement generation.
- `FreeOutRampOnStraight` now publishes one non-routing junction surface.  It
  is not inserted into the road graph, so navigation and lane localization are
  unchanged.
- The hybrid map exposes that surface as a drivable geometry overlay.
- The strict Normal planner / committed-executor XL footprint audit expands a
  lane's declared junction surfaces automatically.
- The environment's route-aware out-of-road query expands the same declared
  surface without inserting it into lane localization.
- Simulator GT semantic BEV includes the overlay in `drivable` only; it does
  not create a fake lane centerline, boundary, or navigation route.

## Verification

### Geometry and regression tests

```text
tests/test_platoon_normal_planner.py
tests/test_semantic_bev_rasterizer.py
tests/test_s8_g_block_drivable_geometry.py
=> 54 passed
```

Broader expert/collector regression:

```text
tests/test_rule_maker.py
tests/test_platoon_normal_planner.py
tests/test_bev_longitudinal_reference.py
tests/test_semantic_bev_rasterizer.py
tests/test_s8_g_block_drivable_geometry.py
tests/test_scenario_definitions.py
tests/test_joint_bev_collection.py
=> 172 passed
```

Real S8 map assertions:

- overlay count: `1`
- overlay width: `3.5 m`
- overlay is absent from `road_network.get_all_lanes()`
- continuous S8 route XL footprint audit: `passed`
- overlay polygon present in semantic BEV drivable geometry: `true`

### Closed-loop smoke

S8, seeds `[17, 23]`, `2 x 80`:

- collision rate: `0`
- out-of-road rate: `0`
- junction footprint rejection: `0`
- both episodes stop at `0.8 s` with
  `committed_trajectory_tracking_deviation`
- exact failure: agent1 longitudinal error `-1.023554 m`, compared with the
  existing `1.0 m` committed tracking envelope

This stop occurs before reaching the G-block junction and is unchanged from
the pre-13.881b evidence.  It is therefore a separate tracking-contract issue,
not a remaining junction geometry defect.

Evidence directory:

```text
/tmp/bev-round13-881b-s8-smoke
```

S6 seed `17`, `1 x 80` non-regression:

- `80/80` native planning steps
- no failure, collision, or out-of-road termination

Evidence directory:

```text
/tmp/bev-round13-881b-s6-regression
```

## Conclusion

Round 13.881b's geometry boundary is complete: S8's G-block route path,
strict XL footprint road audit, and semantic BEV drivable surface now agree.
Full S8 episode persistence remains blocked by the already isolated
longitudinal committed-tracking envelope failure and must not be addressed by
widening this junction or weakening road safety checks.
