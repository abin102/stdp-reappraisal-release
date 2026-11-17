# SoftHebb pilot: trained against untrained weights

## Reproduction check (CIFAR-10, whole training split, seed 0)

- trained: 79.72% (paper 80.3%; bar 78.3%): met
- random: 73.86%; gain +5.86 points

## Medical sets, 10,000 training images (the STDP paper's matrix size)

| Dataset | Trained, seeds 0-2 | Random, seeds 0-2 | Gain per seed, points | Mean gain | Reading |
|---|---|---|---|---|---|
| organamnist | 89.79, 90.17, 90.00 | 89.88, 89.26, 89.44 | -0.08, +0.91, +0.56 | +0.46 | mixed |
| bloodmnist | 91.82, 91.58, 92.31 | 92.72, 92.25, 92.58 | -0.91, -0.67, -0.26 | -0.61 | does not add |
| pathmnist | 77.60, 80.86, 78.75 | 76.52, 78.38, 78.11 | +1.09, +2.48, +0.64 | +1.40 | adds |

## Medical sets, the whole training split

| Dataset | Trained, seeds 0-2 | Random, seeds 0-2 | Gain per seed, points | Mean gain | Reading |
|---|---|---|---|---|---|
| organamnist | 91.27, 91.10, 91.04 | 90.33, 90.49, 90.43 | +0.94, +0.60, +0.61 | +0.72 |  |
| bloodmnist | 91.46, 91.67, 92.84 | 93.01, 92.52, 92.90 | -1.55, -0.85, -0.06 | -0.82 |  |
| pathmnist | 77.17, 77.41, 77.69 | 82.27, 81.17, 81.63 | -5.10, -3.76, -3.94 | -4.27 |  |

