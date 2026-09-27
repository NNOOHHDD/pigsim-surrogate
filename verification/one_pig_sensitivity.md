# Paper-figure re-reading and numerical sensitivity check

This note records the numerical-sensitivity checks of the paper reproduction
against Nieckele et al. (2000). The comparison is a reproduction of printed
paper figures, not a comparison with the authors' raw numerical output or
source code.

## Reference-data correction

Figures 3, 7, 8, 10 and 11 were rendered at 600 dpi and re-read. The corrected
coordinates are stored in `plots/paper_reference.py`, separately from the
solver and case definitions.

The main corrections are:

- The rising non-isothermal branch of Figure 7 had previously been read too
  low. At 1200 m, for example, the corrected reading is about 10.43 m/s rather
  than 7.5 m/s.
- The old 45 m/s point at 12 km in Figure 10 was not on the non-isothermal
  curve. The corrected reading is about 35 m/s. Additional points around the
  10 km expansion were included to represent the sharp peak and decay.
- The approximately 600 s arrival at the 20 km contraction, explicitly
  described in the paper text, was restored to the Figure 11 milestones.
- The Figure 3 inlet value at 600 s was re-read as about 61 atm rather than
  60 atm.

These are digitised samples of printed lines and statements, not exact values
published by the authors. No uncertainty values are assigned because the
paper provides none, and the source does not contain raw tabulated results.

This change does not modify the flow equations, solver controls, initial or
boundary conditions, fluid properties, contact forces, bypass coefficient,
hole sizes, valve coefficient, mesh, or time step. Consequently, the lower
errors below are a correction of the comparison data, not an improvement to
the numerical solver.

## Procedure

- Case 1: vary the cells in each moving domain, maximum time step, and the
  unavoidable non-zero initial pig position.
- Case 2: vary the cells in each moving domain, maximum time step, and the
  smoothing length used for the two diameter steps.
- Evaluate the same corrected paper points for every run. Arrival times are
  linearly interpolated between the two time levels that bracket each target.
- Keep all published physical inputs fixed; no physical coefficient is fitted
  to the digitised curves.

The metric is the RMS of pointwise relative errors. `Combined` is the RMS of
the panel-level RMS values. It is useful for comparing numerical settings
within a case, but it is not an experimental uncertainty or validation limit.

## Results

### Case 1 - dewatering riser

| Cells/domain | dt max (s) | Initial position (m) | Fig. 7 velocity | Fig. 3 inlet pressure | Fig. 8 arrival | Combined |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 2.0 | 10 | 11.70% | 9.53% | 3.89% | 9.00% |
| 300 | 2.0 | 10 | 13.78% | 9.56% | 4.04% | 9.96% |
| 200 | 0.5 | 10 | 13.07% | 9.66% | 3.71% | 9.62% |
| 200 | 2.0 | 1 | 12.03% | 9.88% | 3.39% | 9.20% |

The largest absolute Figure 7 difference in the baseline is about 0.64 m/s at
1280 m. The larger relative errors occur at the low-speed portion near 850 m,
where a small absolute difference is divided by a value below 1 m/s.
Increasing the grid or reducing the maximum time step shifts these small
oscillations but does not consistently improve the full comparison.

### Case 2 - severe area changes

| Cells/domain | dt max (s) | Transition (m) | Fig. 10 velocity | Fig. 11 arrival | Combined |
|---:|---:|---:|---:|---:|---:|
| 70 | 2.0 | 20 | 7.75% | 9.17% | 8.49% |
| 140 | 2.0 | 20 | 7.76% | 9.19% | 8.51% |
| 70 | 0.5 | 20 | 7.94% | 9.23% | 8.61% |
| 70 | 2.0 | 5 | 7.75% | 9.16% | 8.48% |

The earlier 12 km mismatch was mainly a digitisation error. Refining the grid,
time step, or area-transition length does not materially improve the corrected
comparison. The main remaining discrepancy is timing: the paper places the
20 km contraction arrival at approximately 600 s, while this implementation
reaches it at about 697.9 s.

## Decision

- Use the corrected, centrally stored figure coordinates for future plots and
  sensitivity calculations.
- Treat the revised error values as corrected reporting, not solver accuracy
  gained by tuning.
- Keep the existing solver and physical parameters unchanged.
- Freeze these reference coordinates before future solver comparisons so that
  a code change and a reference-data change are never reported as one result.

## Reproduction

```powershell
python plots/one_pig_sensitivity.py --case case1 --N 200 --dt-max 2 --s0 10
python plots/one_pig_sensitivity.py --case case2 --N 70 --dt-max 2 --transition 20 --s0 5
```

## Related

An unmerged attempt at the same accuracy question from the numerical side —
paper-exact central differencing and proportional node redistribution — is
recorded in `adaptive_grid_reference.md`. It is reference material; none of it
is implemented here.
