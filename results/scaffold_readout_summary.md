# Scaffold-matrix readout table (Section 4.1)

Both readouts on identical frozen spike counts; 3 rules x 4 scaffolds
x 5 seeds = 60 cells. Per-cell values in `scaffold_readout_table.csv`.

## Per-scaffold means (%)

| rule | scaffold | post-hoc | linear | gap |
|---|---|--:|--:|--:|
| pair_exp | S_full | 78.99 | 85.92 ± 0.33 | +6.9 |
| pair_exp | S_inh_only | 9.96 | 28.13 ± 2.86 | +18.2 |
| pair_exp | S_thr_only | 9.80 | 16.92 ± 0.32 | +7.1 |
| pair_exp | S_none | 9.74 | 19.42 ± 2.03 | +9.7 |
| triplet | S_full | 80.87 | 87.73 ± 0.21 | +6.9 |
| triplet | S_inh_only | 10.15 | 25.32 ± 1.94 | +15.2 |
| triplet | S_thr_only | 9.80 | 17.69 ± 0.30 | +7.9 |
| triplet | S_none | 9.74 | 19.59 ± 2.33 | +9.8 |
| voltage | S_full | 13.85 | 19.36 ± 0.20 | +5.5 |
| voltage | S_inh_only | 13.82 | 19.34 ± 0.28 | +5.5 |
| voltage | S_thr_only | 9.74 | 27.74 ± 2.40 | +18.0 |
| voltage | S_none | 14.41 | 19.27 ± 0.11 | +4.9 |

## Claims quoted in Section 4.1

| claim | recomputed |
|---|---|
| bound holds, cells clearing majority floor | 60/60 |
| bound holds, pair+triplet cells | 40/40 |
| median gap (pair+triplet) | +7.7 pp |
| max gap (pair+triplet) | +23.3 pp |
| lowest-scoring pair/triplet cell under linear readout | 16.4% |
| cells at/below majority floor under linear readout | 0 (all voltage) |
| worst shortfall below floor | -5.08 pp |
| majority-class floor (MNIST) | 11.35% |

## Readout effect vs rule effect at S_full

- switching the readout: +6.9 pp (pair), +6.9 pp (triplet)
- switching the rule (pair->triplet): +1.9 pp (post-hoc), +1.8 pp (linear)
