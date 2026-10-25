# Pre-registration: homeostatic target sweep for the exact Clopath port, MNIST

Written 2026-10-13 22:10 IST, before any run of this experiment (including the reproduction
check of Section 6) and before any test accuracy of it was seen. Only timing benchmarks were
run before this file (33 training images per layout, STDP on, theta_tar = 60, outputs
discarded, no labels). The SHA-256 of this file at launch is stored in every run's
`meta.json` (`prereg_sha256`, `prereg_file`) and in `logs_clopath_port/master.log`; a frozen
copy goes to `logs_clopath_port/`. Nothing below may be changed after launch; corrections go
into a dated addendum at the end.

Parent experiment: `results/voltage_clopath_neuron_port_PREREG.md` (the first port, SHA-256
at launch `225e36b3...`) and `results/voltage_clopath_neuron_port.md`. Everything not stated
here is as in that pre-registration (its Sections 2-4 and 6-7).

## 1. Question

The exact port of Clopath et al. (2010)'s neuron and rule (ModelDB 144566, `aEIF.m` +
`RFdevelop.m`) fell below the same network with STDP disabled in all 18 pre-registered MNIST
settings; 17 of 18 trained networks fell silent. The port kept Clopath's homeostatic target
theta_tar = 60 mV^2, which was set for his input (10 Gaussian patterns, at most about 15 Hz
per input) and not for MNIST (784 inputs, up to 63.75 Hz). In `RFdevelop.m` line 90 the
target enters only the depression term:

    w = w + A3p*r*(u_sig_mp.*uy)' - A2d*inp*(u_sig_md.*(theta./theta_tar))'

where theta is the 1.2-s running average of (u - E_L)^2. A larger target weakens depression
by the same factor in every state of the network. Question: **does any homeostatic target
let the rule beat the same network with STDP disabled, on Clopath's own neuron, on MNIST?**

Facts already known (training diagnostics of the first grid, seed 0, no test data):
the STDP-off networks run at a homeostatic average of 74-1,219 mV^2 (so at theta_tar = 60
depression acts at 1.2-20 times its nominal strength), the trained networks that fell silent
at 0.3-1.5 mV^2, and the one network that kept firing (c33, 1x) at about 200-290 mV^2. In
RFdevelop's own setup the average is 240-520 mV^2 (first port, Section 2).

## 2. Values

theta_tar = 60, 240, 960, 3840, 15360 mV^2 (60 x 4^k, k = 0-4). 60 is Clopath's value;
15360 makes depression 256 times weaker. 240-960 cover RFdevelop's own operating point and
the lower STDP-off averages on MNIST, 3840 lies above every STDP-off average seen, 15360 far
above.

**theta_tar = 60 is not rerun:** its trained networks are the first grid's
(`experiments/runs/clopath_port/grid10k/seed*/`), computed by the same code path (checked
bit for bit, Section 6). New runs: theta_tar = 240, 960, 3840, 15360.

## 3. Cells, amplitudes, seeds, length

- Cells: **all 9** (g, k_inh) cells of the first grid (g = 0.2, 0.275, 0.5; k_inh = 0.1,
  0.3, 100 mV; `results/clopath_port_calibration/grid.json`, unchanged). Decision from the
  cost estimate (Section 7): all 9 cells cost about 6.6 GPU-hours, about 1.4 times the first
  grid's GPU time and finished overnight on two GPUs; the 5-cell subset (c21, c22, c23, c32,
  c33) would save about 3 GPU-hours. All 9 are run, so no cell is chosen by its STDP-off
  accuracy.
- Amplitudes: 10x (RFdevelop: A3p = 8e-4, A2d = 1.4e-3) and 1x (published: 8e-5, 1.4e-4).
- Seeds 0, 1, 2; 10,000 training images (order `randperm(60000, seed + 1000)`); test on the
  first 10,000 test images with STDP off, the simulation continuing from the end-of-training
  state, as in the first port.
- 4 new targets x 2 amplitudes x 9 cells = 72 new trained networks per seed, 216 in all.

## 4. STDP-off networks: reused, not rerun

With STDP off the weight update, and with it theta_tar, is never evaluated, so the STDP-off
networks do not depend on theta_tar. The STDP-off spike counts are taken from
`experiments/runs/clopath_port/grid10k/seed{0,1,2}/test_spike_counts.npz` (kind "random").
Pairing holds exactly: for a given seed every network sees the same initial weights
(`init_weights(seed)`), the same image order and the same Poisson spike trains, training and
test (the encoder draws from one generator per process, once per image, independently of the
networks). The scorer checks the initial weights, test labels, n_train, n_test and seed
against the reused run before pairing.

## 5. Probe and decision rule (as the first port)

- Primary probe (decisions): `LogisticRegression(max_iter=2000, C=1.0)` on the raw spike
  counts of the first 5,000 test images, stratified 70/30 split
  (`train_test_split(..., test_size=0.3, random_state=0, stratify=labels)`).
- Secondary, reported only: the same classifier on a 70/30 `RandomState(0)` permutation of
  all 10,000 test images (perm70).
- For each of the 72 new cells (g, k_inh, amplitude, theta_tar): gap_s = trained_s -
  STDP-off_s (same seed, same g and k_inh), mean m and sample SD sd (ddof = 1) over the
  3 seeds. The cell **beats random** if m > 0 and m > 2 sd.
- Every cell is reported, silent ones included (11.4 % = the majority class of the test
  split, the score of a network with no spikes). The 18 cells at theta_tar = 60 are reported
  again from the first grid (none beats random there). Nothing is selected on test accuracy;
  no cell is added or removed after results are seen.

**Confirmation.** Every cell that beats random is rerun with 60,000 training images, seeds
0-2, trained and STDP off in the same process (new STDP-off networks at 60,000 images, since
the test state and test spike trains differ), all 10,000 test images, scored with perm70.
Confirmed if m > 0 and m > 2 sd on perm70 (strat5k also reported). If the confirmation would
cost more than 12 GPU-hours, the author is asked before it is launched; no cell is dropped
from it on the basis of its accuracy.

**Verdict, stated in advance.**
- No cell beats random at any target: no homeostatic target between 60 and 15,360 mV^2 lets
  the rule beat random initialisation on Clopath's own neuron in this MNIST setting; the
  untuned-target objection does not explain the failure.
- Some cell beats random but none is confirmed: same verdict, the unconfirmed cells reported.
- At least one cell confirmed: with that target the rule beats random on Clopath's neuron;
  the failure at 60 mV^2 is then a parameter-transfer result, and the Table 3 row and its
  note must say so.

## 6. Code change and checks before the sweep

- `models/clopath_port.py`: `ClopathNet(..., theta_tar=...)`, a scalar or one value per
  network (default 60, the module constant). If all networks of an object share one value,
  the update uses `theta / theta_tar` with a Python float, the expression used so far (so
  the default is the old code path, operation for operation). With several values per
  object, theta_tar is a per-network column (true division; on CUDA torch divides by a
  Python scalar by multiplying with its reciprocal, so the two forms can differ in the last
  bit). `network_ref(..., theta_tar=...)` gets the same argument (NumPy, true division).
- `scripts/clopath_port_mnist.py train`: `--theta_tar` (comma-separated, default 60; specs =
  theta_tar x amplitude x cell), `--no_random` (no STDP-off networks; used by the sweep),
  `--prereg` (the file whose SHA-256 goes into `meta.json`).
- Check 1 (per-network column path): the batched torch network with theta_tar = 60, 240,
  960, 3840, 15360 in one object against `network_ref` run separately for each target
  (5 networks x 12 neurons x 784 inputs, 1.2 s of Poisson input, inhibition on, float64,
  fused and unfused update, CPU and GPU). Pass: identical spike trains; voltage and weight
  differences at rounding level (first port: 3.4e-8 mV, 3.6e-10).
- Check 2 (default = old code path, before the sweep is launched): rerun seed 0 of the first
  grid with the modified script, default theta_tar, STDP-off networks left out (they do not
  share any arithmetic with the trained ones). The 18 trained networks keep their batch of
  18 (cuBLAS picks its kernel for the input current by batch size; 17-30 give bit-identical
  results to 27, checked on 2026-10-13, so the test-phase batch of 18 matches the first
  grid's 27). Pass: per-image training spike counts, active fractions, potentiation and
  depression totals (all 10,000 images), weights at 1,000, 5,000 and 10,000 images, and the
  per-neuron test spike counts of all 10,000 test images are identical to
  `grid10k/seed0` for all 18 networks. If not identical, the sweep is not launched.
- The sweep runs 72 (or 36) trained networks per object with the per-network column; its
  networks therefore differ from a hypothetical 18-network run only by rounding (as the fused
  update does); none of them is compared bit for bit with anything.

## 7. Cost estimate and schedule

Benchmarks on GPU 1 alone (2026-10-13 21:45-22:05, float64, CUDA graph, fused update):

| layout | training s/image | test s/image |
|---|--:|--:|
| 18 trained networks (first grid's trained batch) | 0.194 | 0.106 |
| 36 trained networks, one object | 0.309 | 0.145 |
| 72 trained networks, one object | 0.524 | 0.205 |
| 72 trained networks as 4 objects of 18 | 0.785 | 0.421 |
| two 72-network processes sharing one GPU | 1.13 each | - |

Two processes on one GPU are slower than one after the other, so each GPU runs one process at
a time. Jobs: seed 0 and seed 2, all four targets in one 72-network process each (about
87 min training + 34 min test = 2.0 h each); seed 1 split into two 36-network processes
(theta_tar 240 + 960, and 3840 + 15360; about 52 + 24 = 1.3 h each). Queue: GPU 1 runs the
check-2 reproduction (about 50 min), then seed 0, then seed 1 part a; GPU 0 runs seed 2, then
seed 1 part b. Total about 6.6 GPU-hours, about 3.3 h wall-clock after the check.

GPU 0 belongs to a guest: each GPU-0 job starts only if `nvidia-smi --query-compute-apps`
lists nothing on GPU 0 except `gnome-remote-desktop-daemon` at that moment; otherwise the job
moves to the end of GPU 1's queue (about 2-3 h more). Guest processes are never touched.
Runs: `experiments/runs/clopath_port/thetatar10k/seed{0_all,1_a,1_b,2_all}/`; check 2:
`experiments/runs/clopath_port/thetatar_repro/seed0/`. Logs: `logs_clopath_port/`.

## 8. Recorded per network

- Test: spikes per test image (400 neurons, 500-ms slot), fraction of silent test images,
  neurons active on at least one test image.
- Training: spikes and active fraction per training image; **fell silent during training** =
  no spike in any training image after some image before the last one; reported as the last
  training image with at least one spike (1-based; 10,000 means it did not fall silent), and
  the number of training images without a spike.
- Weights (end of training; also at 1,000 and 5,000 images): mean, SD, fraction at 0, at 3,
  mean pairwise cosine between neurons' weight vectors.
- Potentiation and depression: totals over training of the per-step sums A3p * sum(r) *
  sum(u_sig_mp * uy) and A2d * sum(inp) * sum(u_sig_md * theta / theta_tar) (before
  clipping, summed over synapses, indicative as in the first port), and per image over the
  last 1,000 images.
- Homeostatic average theta (mean over neurons) every 500 images, and theta / theta_tar at
  the end of training (the depression factor).

## 9. Outputs

`results/voltage_clopath_port_thetatar.md` (narrative, verdict, what it means for the
Table 3 row "Clopath's neuron, exact port" and its note), `results/voltage_clopath_port_thetatar.json`
(every network and cell), `results/voltage_clopath_port_thetatar_tables.md`, one figure of
the gap against theta_tar (`results/clopath_port_thetatar_figures/`). The paper drafts are not
edited.
