# Adaptive-grid paper-reproduction mode (reference only)

This note preserves an earlier line of work that was never merged into the
code in this repository. It is kept as **reference material**. None of the code
described here is part of the current solver; the accompanying patch is a
record of the approach, not something to apply.

## Provenance

The work is a single commit, "Reproduce paper conditions with adaptive grid"
(2026-08-06), made on a development branch that has since been deleted. It
forked from an earlier version of the solver (2026-08-05) and never shared
history with the current code beyond that point.

The full diff is stored beside this note as `adaptive_grid.patch` (722 lines;
regenerated figures were dropped, code and documentation kept). It will not
apply to the current code — the solver has moved on since the fork.

## What it did

It added a `paper_exact` switch to both cases, on by default, with
`--stabilized` restoring the previous behaviour. Six things changed under it.

**1. Node redistribution.** `PigFlowSolver` gained `adaptive_nodes`,
`N_total` and `min_side_cells`. The paper holds the total cell count fixed and
splits it between the two sides of the pig in proportion to their lengths;
`_node_counts` reproduces that, and `_rebalance_domains` applies it on
`initialise` and at the top of every `step`. `Domain.remesh` carries the
solution across a cell-count change, interpolating `p` and `h` linearly on cell
centres and `V` on faces. The paper does not say what interpolation it used, so
linear was a choice, not a reading.

`_save`/`_load` were extended to carry `up.N` and `dn.N`, because a rejected
time step must restore the mesh it was taken on, not just the fields.

**2. Central differencing for enthalpy.** `scheme_h` moved from `upwind` to
`central`. Pressure and velocity were already central, so this was the only
scheme change needed to match the paper's stated discretisation.

**3. Discontinuous area change in Case 2.** Diameter and wall thickness became
step functions at 10 km and 20 km (`np.where`) instead of the 20 m `tanh`
blend. The contact-force boundary was corrected at the same time: `in_end`
tested `s > 2*L_SEC`, which excluded the transition node itself, and became
`s >= 2*L_SEC`.

**4. Case 2 outlet valve equation.** The paper writes
`mdot = rho (Cd A)_0 chi sqrt((p - p_res)/rho)`, giving `V = C sqrt(dp/rho)`.
The implementation had the conventional orifice form with a factor of two. A
`flow_factor` field on the valve boundary spec selects between them — 1.0 in
paper mode, 2.0 otherwise — and the Jacobian entry `dvdp` was rewritten to stay
consistent with whichever is chosen.

**5. Initial pig position.** The continuum problem starts at the entrance, but
the split-domain discretisation needs cells on both sides of the pig. Paper
mode starts at `1e-6 * L` as the numerical limit of `s = 0`; the earlier code
started at 10 m (Case 1) and 5 m (Case 2).

**6. Convergence made loud.** `picard_iter` went 6 to 12 and `picard_relax`
1.0 to 0.8. `fail_on_picard` raises `FloatingPointError` when Picard iteration
exhausts its count instead of returning quietly, and `run` now raises if a step
is still rejected at `dt_min` rather than accepting it and continuing. Both
turn silent inaccuracy into a stop.

`requirements.txt` was added in the same commit, pinning numpy 2.5.1,
scipy 1.18.0 and matplotlib 3.11.1.

A test, `test_paper_adaptive_mesh`, checked the 10/30 and 30/10 splits at
`s/L = 0.25` and `0.75`, that a uniform pressure field survives a remesh, and
that the paper valve form carries no factor of two.

## Results it reported

Total-cell convergence, paper mode:

| | cells | exit time |
|---|---|---|
| Case 1 | 200 / 400 / 800 | 1216.7 / 1212.5 / 1213.8 s |
| Case 2 | 70 / 140 / 280 | 1544.6 / 1544.5 / 1545.4 s |

The pig trajectory converges. The temperature field does not behave as well:
central differencing leaves Gibbs-type oscillation behind the sharp temperature
front at the pig. Snapshot minima between 400 and 1200 s sit near -2 to -12 C,
and a very short downstream stretch just before exit undershoots to about
-37 C. That is what the paper's stated scheme produces here; the upwind
enthalpy of `--stabilized` existed to suppress it.

## Why this is reference only

The current code answered the same accuracy question from the other end. It left the
numerics alone and corrected the digitised comparison data instead, re-reading
the paper figures at 600 dpi — see `one_pig_sensitivity.md`, which states
explicitly that its lower errors are a correction of the comparison data and
not an improvement to the solver.

So the two lines are not competing versions of one fix. This one moves the
discretisation toward what the paper says it did; the current code moves the reference
values toward what the paper actually printed. **They have never been run
against each other**, and the numbers above were produced on the pre-fork
solver, so they are not comparable to anything produced by the current code.

## If it is ever revived

The patch cannot be replayed. Anything taken from it has to be rewritten
against the current solver, and three things would need checking first:

- whether the Gibbs oscillation is acceptable, or whether the paper's scheme
  needs a limiter that the paper does not mention;
- whether node redistribution still pays for itself now that the reference
  values have been corrected — the error it was chasing may no longer be there;
- the valve `flow_factor`, which is a reading of the paper's equation and worth
  confirming against the source before it is relied on again.
