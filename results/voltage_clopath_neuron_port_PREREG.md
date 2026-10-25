# Pre-registration: Clopath's voltage-STDP rule on Clopath's own neuron, MNIST

Written 2026-10-13 14:00 IST, **before** the calibration run, before any training run of this
experiment and before any test accuracy of it was seen. Section 8 is appended after the
calibration (STDP off, training images, no labels) and before the grid is launched. The
SHA-256 of this file at launch is stored in every run's `meta.json` (`prereg_sha256`) and
in `logs_clopath_port/master.log`. Nothing below may be changed after launch; corrections
go into a dated addendum at the end.

## 1. Question

Our fully connected network applies Clopath et al. (2010)'s voltage-STDP rule to an AdEx
neuron with Diehl and Cook's constants, and it falls below the same network with STDP
disabled (MNIST: 25.7 +/- 0.4 % vs 71.0 +/- 0.6 %, 3 seeds). The likely reviewer objection
is "you never ran Clopath's rule on Clopath's own neuron". This experiment does that:
does the rule, on Clopath's neuron, learn MNIST features that beat the same network with
STDP disabled?

## 2. Taken unchanged from Clopath's code (ModelDB 144566, aEIF.m and RFdevelop.m)

Implemented in `models/clopath_port.py`; validated in `results/clopath_port_validation/`
(the NumPy reference matches the original .m files run in GNU Octave 10.3 bit for bit over
RFdevelop's 100-s run; the torch version used here matches the reference to rounding).

- Neuron (aEIF.m): C 281 pF, g_L 30 nS, E_L -70.6 mV, V_T rest -50.4 mV, Delta_T 2 mV,
  a 4 nS, tau_w 144 ms, b = 0.0805 added to the pA-valued adaptation variable (so 0.08 pA,
  as in the code), afterdepolarisation 400 pA (tau 40 ms), V_T jump 20 mV (tau 50 ms);
  spike registered at u > 20 mV, clamped at 29.4 and 32.862 mV, then reset to
  E_L + 15 + 6.0984 = -49.5016 mV; 1-ms Euler steps.
- Rule (RFdevelop.m), every 1-ms step:
  w += A3p * r * (u_sig_mp * uy)' - A2d * inp * (u_sig_md * theta / theta_tar)',
  with r the presynaptic trace (tau 15 ms), u_mp and u_md the voltage averages (tau 7 and
  10 ms) read one step late (u1ss; the code comment says 2 ms, the code gives 1 step),
  theta+ = E_L + 25.3 = -45.3 mV, theta- = E_L = -70.6 mV, homeostasis theta = 1.2-s
  average of (u - E_L)^2, theta_tar = 60 mV^2; weights clipped to [0, 3] every step;
  initial weights U(0.5, 2).
- Amplitudes, two settings: RFdevelop's 10x values (A3p = 8e-4, A2d = 1.4e-3) and the
  published 1x values (A3p = 8e-5, A2d = 1.4e-4).

## 3. Taken unchanged from our fully connected network (`experiments/scripts/run_single.py`)

400 neurons; MNIST via the same loader (pixel values in [0, 1]); Poisson encoding at
pixel x 63.75 Hz in 1-ms steps for 350 ms, then 150 ms without input; training order
`randperm(60000, seed + 1000)`, first 10,000 images; test = the first test images in order.

## 4. Choices beyond Clopath's code (all fixed here)

1. Input current I = g * C * W' * spikes (Clopath: g = 4, i.e. a 4w-mV step per input
   spike). g is free (Section 5): MNIST delivers about 6.5 input spikes per ms, so g = 4
   saturates every neuron.
2. Competition: lateral inhibition as a current. A spike registered in step t-1 gives
   every other neuron of the network a current of -k_inh * C pA during step t, i.e. a
   k_inh-mV step (the same form as the excitatory input). k_inh is free (Section 5).
3. Voltage floor at -100 mV (the inhibitory reversal potential of Diehl & Cook 2015),
   applied after each neuron step. Reason, found on 2026-10-13 in a 60-image STDP-off
   benchmark before this pre-registration: all neurons receive nearly the same input, so
   100-300 of them cross threshold in the same step; an unbounded current pulse per spike
   then drove u to between -1,000 and -51,000 mV and Clopath's subthreshold adaptation
   (a = 4 nS) to a depolarising current of up to 165,000 pA, which produced rebound
   bursts (firing increased with inhibition). Without inhibition the neuron never comes
   near -100 mV, so the floor does not change RFdevelop.m (checked).
4. State carries over between images (one continuous simulation, as RFdevelop.m, whose
   patterns change every 100 ms without a reset); the 150-ms silent gap separates images.
   This includes the homeostatic average theta (1.2-s window, longer than one image).
5. Features: spikes registered during each image's 500-ms slot, per neuron.
6. Test: STDP off (weights frozen), the simulation continues from the state reached at
   the end of training; the first 10,000 test images are presented.
7. All networks of one seed share the image order, the Poisson spike trains and the
   initial weights; trained and STDP-off networks of a cell differ only in plasticity.
   One STDP-off network per (cell, seed) serves both amplitude settings (STDP off does
   not depend on the amplitudes).
8. No Diehl & Cook adaptive threshold and no L1 weight normalisation (Clopath's model has
   neither; its homeostasis is the theta term). No refractory period beyond Clopath's
   (spike clamp, V_T jump).
9. Numerics: float64; each 500-step image is replayed as a CUDA graph (bitwise identical
   to the eager code); the weight update of trained networks is one fused kernel, which
   can differ from the unfused update in the last bit (fused multiply-add). Both are
   checked against the NumPy reference in `scripts/clopath_port_equivalence.py`.

## 5. Free parameters: calibration and the pre-registered choice

Calibration (`scripts/clopath_port_mnist.py calibrate`): seed 0, the first 1,000 images of
seed 0's training order, STDP off, no labels, 120 networks:
g in {0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8},
k_inh in {0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300} mV. (These ranges were set from an
exploratory 50-image STDP-off check on training images, no labels: below g = 0.15 the
network is silent, at g = 0.8 every neuron fires.)

Definitions: rate = mean over images and the 400 neurons of the spikes in the 350-ms
presentation / 0.35 s; active fraction = mean over images of the fraction of neurons with
at least one spike in the presentation.

Choice (`scripts/clopath_port_select_grid.py`, deterministic): among all product grids
{g1 < g2 < g3} x {k1 < k2 < k3} of calibration values with k1 > 0 (every cell has
inhibition), take the one minimising the sum, over the six marginal targets (rate 2, 6.32,
20 Hz; active fraction 5, 15.8, 50 %), of the distance in log10 units from the target to
the nearest of its 9 cells. Ties: lexicographically smallest (g1, g2, g3, k1, k2, k3).
Rationale: in the exploratory check rate and active fraction move together (20 Hz comes
with about 70 % of neurons active, 5 % active with about 0.3 Hz), so the grid is asked to
span both ranges, not to place every cell inside both.

## 6. Main grid

3 x 3 cells x {trained 10x, trained 1x, STDP off}, seeds 0, 1, 2, 10,000 training images,
test on the first 10,000 test images. 81 networks, run as 3 processes (one per seed).

Recorded for every network: spikes and active fraction per training image; per trained
network the total potentiation and depression per image; weight mean, SD, fraction at 0
and at 3, mean pairwise cosine between neurons' weight vectors; weights at 1,000, 5,000
and 10,000 images; homeostatic theta; test spike counts.

## 7. Readout, decision rule, confirmation

Primary probe (`scripts/clopath_port_score.py`, the voltage-STDP probe of
`results/voltage_matched_random_60k.md`): LogisticRegression(max_iter=2000, C=1.0) on the
spike counts of the first 5,000 test images, stratified 70/30 split (random_state 0).
Secondary, reported only: the same classifier on a 70/30 RandomState(0) permutation of all
10,000 test images (perm70).

For each of the 18 trained cells: gap_s = trained_s - STDP-off_s (same seed, same g and
k_inh), mean m and sample SD sd over the 3 seeds. The cell **beats random** if m > 0 and
m > 2 sd. Every cell is reported; nothing is selected on test accuracy; no cell is added
or removed after results are seen (follow-ups, if any, are labelled exploratory and cannot
change the verdict).

Confirmation: every cell that beats random is rerun with 60,000 training images, seeds
0-2, trained and STDP off, all 10,000 test images, scored with perm70. Confirmed if
m > 0 and m > 2 sd on perm70 (strat5k also reported). With 18 cells and n = 3, false
positives at 10,000 images are expected occasionally; the confirmation guards against them.

Verdict, stated in advance:
- No cell beats random at 10,000 images: on Clopath's own neuron, the rule does not beat
  random initialisation in this MNIST setting.
- Some cell beats random but none is confirmed at 60,000 images: same verdict, with the
  unconfirmed cells reported.
- At least one cell confirmed: the rule beats random on Clopath's neuron in that setting,
  and the paper's voltage-STDP result becomes a rule-neuron compatibility finding.

## 8. Calibration outcome and the chosen grid

(appended after the calibration, before launch)
Appended 2026-10-13 14:14 IST, after the calibration and before any training run of the grid.

Calibration: `results/clopath_port_calibration/calibration.json` (seed 0, first 1,000 training images, STDP off, no labels, 13 min). Rate (Hz) / active fraction (%) of the STDP-off network:

| g \ k_inh (mV) | 0 | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 | 300 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.15 | 0.34 / 4 | 0.25 / 4 | 0.16 / 3 | 0.09 / 2 | 0.05 / 1 | 0.04 / 1 | 0.03 / 1 | 0.03 / 1 | 0.02 / 1 | 0.02 / 1 |
| 0.175 | 1.23 / 14 | 0.91 / 12 | 0.58 / 10 | 0.34 / 7 | 0.19 / 5 | 0.13 / 4 | 0.11 / 3 | 0.09 / 3 | 0.07 / 2 | 0.06 / 2 |
| 0.2 | 2.94 / 25 | 2.19 / 23 | 1.43 / 21 | 0.84 / 16 | 0.49 / 11 | 0.36 / 9 | 0.26 / 7 | 0.22 / 6 | 0.16 / 5 | 0.14 / 4 |
| 0.225 | 5.57 / 37 | 4.18 / 35 | 2.69 / 31 | 1.60 / 26 | 0.93 / 20 | 0.69 / 16 | 0.53 / 13 | 0.40 / 11 | 0.33 / 9 | 0.26 / 7 |
| 0.25 | 9.03 / 50 | 6.84 / 47 | 4.38 / 43 | 2.62 / 37 | 1.55 / 29 | 1.13 / 24 | 0.87 / 20 | 0.69 / 17 | 0.50 / 13 | 0.42 / 11 |
| 0.275 | 13.19 / 62 | 10.10 / 59 | 6.48 / 55 | 3.89 / 49 | 2.29 / 39 | 1.68 / 33 | 1.24 / 27 | 0.93 / 23 | 0.68 / 18 | 0.54 / 14 |
| 0.3 | 17.83 / 71 | 13.82 / 69 | 8.86 / 65 | 5.39 / 59 | 3.18 / 49 | 2.29 / 42 | 1.70 / 36 | 1.27 / 30 | 0.90 / 23 | 0.73 / 19 |
| 0.35 | 28.02 / 84 | 22.19 / 82 | 14.38 / 80 | 8.78 / 75 | 5.20 / 66 | 3.68 / 58 | 2.70 / 50 | 1.97 / 42 | 1.25 / 31 | 1.03 / 26 |
| 0.4 | 38.57 / 90 | 31.19 / 89 | 20.50 / 87 | 12.69 / 84 | 7.40 / 77 | 5.18 / 69 | 3.69 / 61 | 2.71 / 52 | 1.56 / 38 | 1.34 / 32 |
| 0.5 | 58.71 / 96 | 49.30 / 96 | 33.58 / 95 | 21.12 / 93 | 12.13 / 89 | 8.23 / 84 | 5.68 / 76 | 4.18 / 67 | 2.15 / 48 | 1.82 / 40 |
| 0.6 | 76.69 / 99 | 66.34 / 99 | 47.09 / 98 | 29.73 / 97 | 16.68 / 95 | 11.07 / 91 | 7.67 / 85 | 5.71 / 78 | 2.72 / 57 | 2.34 / 49 |
| 0.8 | 106.89 / 100 | 95.91 / 100 | 74.02 / 100 | 46.65 / 99 | 25.53 / 99 | 16.43 / 97 | 11.32 / 94 | 8.20 / 89 | 4.19 / 72 | 3.56 / 64 |

Chosen by `scripts/clopath_port_select_grid.py` (score 0.129; file `results/clopath_port_calibration/grid.json`): **g = 0.2, 0.275, 0.5; k_inh = 0.1, 0.3, 100 mV.**

| cell | g | k_inh (mV) | calibration rate (Hz) | calibration active fraction (%) |
|---|--:|--:|--:|--:|
| c11 | 0.2 | 0.1 | 1.43 | 20.6 |
| c12 | 0.2 | 0.3 | 0.84 | 16.3 |
| c13 | 0.2 | 100 | 0.16 | 4.6 |
| c21 | 0.275 | 0.1 | 6.48 | 55.1 |
| c22 | 0.275 | 0.3 | 3.89 | 48.6 |
| c23 | 0.275 | 100 | 0.68 | 17.8 |
| c31 | 0.5 | 0.1 | 33.58 | 95.0 |
| c32 | 0.5 | 0.3 | 21.12 | 93.3 |
| c33 | 0.5 | 100 | 2.15 | 48.4 |

Nearest cell to each target: rate 2 Hz: c33; rate 6.32 Hz: c21; rate 20 Hz: c32; active 5 %: c13; active 15.8 %: c12; active 50 %: c22.
The k_inh values are unevenly spaced (0.1, 0.3, 100 mV); that is the rule's output and is kept.

Cost estimate (before launch): benchmark on GPU 1 alone, one seed's 27 networks (18 trained + 9 STDP off) in one process: 0.38 s per training image, 0.15 s per test image, i.e. about 63 + 25 = 88 min per seed. Three seeds run in parallel on GPU 1, which a guest job (`train_amp.py`, user guest) also uses since 13:52; expected wall-clock 4-6 h, about 4-5 GPU-hours. A 60,000-image confirmation, if needed, costs about 6 x the training part per seed for the confirmed cells only.

## 9. Addendum 1 (2026-10-13 19:00 IST, after the grid results): exploratory, cannot change the verdict

The pre-registered grid gave no cell above random (17 of 18 trained networks silent on the
test images). Because every grid cell has lateral inhibition (k_inh >= 0.1 mV, which during
100-spike volleys still sums to about 10 mV), an exploratory run checks whether the collapse
needs our inhibition: k_inh = 0 (no inhibition; the only addition to Clopath's code is the
input gain g), g = 0.2, 0.275, 0.5 (the grid's values; STDP-off calibration rates 2.94, 13.19,
58.71 Hz, active fractions 25, 62, 96 %), amplitudes 10x and 1x, STDP off, seeds 0-2, 10,000
training images, first 5,000 test images, same probe. File:
`results/clopath_port_calibration/grid_explore_k0.json`; runs
`experiments/runs/clopath_port/explore_k0/`. Reported separately from the grid and labelled
exploratory.

Outcome of addendum 1 (2026-10-13 21:05): no cell beats random; 5 of 6 trained settings silent on
every test image, the sixth (g = 0.5, 1x) silent on 95 % of them (13.0 +/- 2.7 % vs 80.0 +/- 1.1 %).
Reported in `results/voltage_clopath_neuron_port.md`, Section 11.
