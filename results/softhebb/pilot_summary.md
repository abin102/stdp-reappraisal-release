# SoftHebb pilot: trained against untrained weights and against the pixels

Test accuracy (%) at the last readout epoch, full official test split. Mean ± SD over seeds.

## CIFAR-10 (whole training split): reproduction check

- Stage-1 bar, seed 0 trained: 79.72% (bar 78.3%): met. Paper: 80.31 ± 0.14 (4 runs).

| | Seeds 0-2 | Mean ± SD |
|---|---|---|
| trained | 79.72, 80.48, 80.46 | 80.22 ± 0.43 |
| random | 73.86, 73.72, 73.70 | 73.76 ± 0.09 |
| pixels | 41.01, 41.45, 41.35 | 41.27 ± 0.23 |

Gain per seed (trained minus random): +5.86, +6.76, +6.76

## Medical sets, 10,000 training images (the STDP paper's matrix size): readings

| Dataset | Trained | Random | Pixels | Gain per seed (trained minus random) | Trained > random | Training, 5 seeds | Training, stage 1 (seeds 0-2) | Trained > pixels | Input, 5 seeds | Random > pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| organamnist | 89.98 ± 0.31 | 89.49 ± 0.27 | 63.22 ± 0.43 | -0.08, +0.91, +0.56, +1.12, -0.10 | 3 of 5 | mixed | mixed | 5 of 5 | beats its input | 5 of 5 |
| bloodmnist | 91.53 ± 0.60 | 92.62 ± 0.29 | 79.92 ± 0.26 | -0.91, -0.67, -0.26, -1.81, -1.75 | 0 of 5 | does not add | does not add | 5 of 5 | beats its input | 5 of 5 |
| pathmnist | 78.94 ± 1.28 | 77.47 ± 0.80 | 56.35 ± 0.86 | +1.09, +2.48, +0.64, +2.60, +0.52 | 5 of 5 | adds | adds | 5 of 5 | beats its input | 5 of 5 |
| tissuemnist | 54.20 ± 0.27 | 56.51 ± 0.21 | 47.26 ± 0.14 | -2.22, -2.27, -2.31, -2.82, -1.87 | 0 of 5 | does not add | n/a (not in stage 1) | 5 of 5 | beats its input | 5 of 5 |

## Medical sets, whole training split (reported, no reading)

| Dataset | Trained | Random | Pixels | Gain per seed (trained minus random) | Trained > random | Trained > pixels |
|---|---|---|---|---|---|---|
| organamnist | 91.09 ± 0.12 | 90.45 ± 0.17 | 64.09 ± 0.12 | +0.94, +0.60, +0.61, +0.66, +0.39 | 5 of 5 | 5 of 5 |
| bloodmnist | 91.83 ± 0.59 | 92.90 ± 0.23 | 80.22 ± 0.17 | -1.55, -0.85, -0.06, -1.32, -1.61 | 0 of 5 | 5 of 5 |
| pathmnist | 77.33 ± 0.23 | 81.67 ± 0.40 | 58.19 ± 0.40 | -5.10, -3.76, -3.94, -4.39, -4.50 | 0 of 5 | 5 of 5 |

