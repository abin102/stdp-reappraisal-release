# Scaffold-matrix readout table (Section 4.1)

Both readouts on identical frozen spike counts; 3 rules x 4 scaffolds
x 5 seeds = 60 cells. Per-cell values in `scaffold_readout_table.csv`.

## Per-scaffold means (%)

| rule | scaffold | post-hoc | linear | gap |
|---|---|--:|--:|--:|
| pair_exp | S_full | 79.36 | 85.93 ± 0.33 | +6.6 |
| pair_exp | S_inh_only | 18.97 | 28.14 ± 2.86 | +9.2 |
| pair_exp | S_thr_only | 9.80 | 16.92 ± 0.33 | +7.1 |
| pair_exp | S_none | 9.78 | 19.42 ± 2.03 | +9.6 |
| triplet | S_full | 80.92 | 87.73 ± 0.21 | +6.8 |
| triplet | S_inh_only | 15.13 | 25.32 ± 1.94 | +10.2 |
| triplet | S_thr_only | 9.80 | 17.69 ± 0.30 | +7.9 |
| triplet | S_none | 9.84 | 19.59 ± 2.33 | +9.8 |
| voltage | S_full | 16.82 | 19.36 ± 0.20 | +2.5 |
| voltage | S_inh_only | 16.79 | 19.34 ± 0.28 | +2.6 |
| voltage | S_thr_only | 10.24 | 27.70 ± 2.39 | +17.5 |
| voltage | S_none | 14.68 | 19.27 ± 0.11 | +4.6 |

## Claims quoted in Section 4.1

| claim | recomputed |
|---|---|
| bound holds, cells clearing majority floor | 60/60 |
| bound holds, pair+triplet cells | 40/40 |
| median gap (pair+triplet) | +7.3 pp |
| max gap (pair+triplet) | +13.2 pp |
| lowest-scoring pair/triplet cell under linear readout | 16.4% |
| cells at/below majority floor under linear readout | 0 (all voltage) |
| worst shortfall below floor | -5.08 pp |
| majority-class floor (MNIST) | 11.35% |

## Readout effect vs rule effect at S_full

- switching the readout: +6.6 pp (pair), +6.8 pp (triplet)
- switching the rule (pair->triplet): +1.6 pp (post-hoc), +1.8 pp (linear)
