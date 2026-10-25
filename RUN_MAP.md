# Run map

This file says which networks stand behind each group of numbers in the paper.
For each group it gives the run directories and seeds, the configuration in
`run_configs/`, the commands that train and score them, and the results file.
`run_configs/INDEX.md` lists every configuration file with its run directories;
its group labels (P1, R6, ...) are given in brackets below.

Conventions:

- Run directories are under `experiments/runs/`. `{S}` is the seed.
- A configuration `run_configs/<family>.yaml` trains the run
  `experiments/runs/<family with _seed{S} after _N400>`. For example,
  `run_configs/pair_exp_S_full_N400_fmnist_probefix_trained.yaml` with seed 3 gives
  `pair_exp_S_full_N400_seed3_fmnist_probefix_trained`.
- Train one family over seeds with
  `SEEDS="0 1 2 3 4" GPUS="0" bash scripts/train/run_seeds.sh <family>`.
- "Re-extracted" runs are scored from spike counts recorded again from the trained
  network (`*_pf.npz`): run `bash scripts/extract/reextract_all.sh` on the lists in
  `results/run_lists/` (README, step 2).
- Commands are run from the repository root. Python scripts are written without
  their `PYTHONPATH=. python` prefix. Scorers run with one BLAS thread (README,
  step 3). "Linear probe" always means the probe described in the README.

## 1. Eight-dataset matrix: Table 2, Figure 7, Sections 4.2 and 4.7, Table A.2, Appendix I, Table A.8

Also the reference row of Table 3 (29.0 against 75.6) and the Fashion-MNIST
comparison of Section 4.3 (64.2 against 64.5).

| | |
|---|---|
| networks | `{rule}_S_full_N400_seed{S}{_dataset}_probefix_{trained,frozen}`; rule = `pair_exp`, `triplet`, `voltage`; `_dataset` is empty for MNIST, else `_nmnist`, `_emnist_letters`, `_fmnist`, `_organamnist`, `_bloodmnist`, `_tissuemnist`, `_pathmnist`; `frozen` = STDP disabled; 240 networks |
| seeds | 0-4 |
| configuration | `run_configs/{rule}_S_full_N400{_dataset}_probefix_{trained,frozen}.yaml` [P1]; pair and triplet use the default full-scaffold configuration, voltage-STDP the tuned one |
| train | `bash scripts/train/probefix_matrix_n5.sh "0" 4` (the runs save their own training- and test-image counts) |
| score | `scripts/score/probefix_aggregate.py` |
| results | `results/probefix_matrix_n5.{json,md}`: Δ-over-random (its Table 1), post-hoc against linear (its Table 3 = paper Table A.2) |

Built on the same networks:

| paper element | command | inputs | results |
|---|---|---|---|
| Table 2, encoded-input and raw-pixel columns | `scripts/score/input_baselines_probefix.py`, then `input_baselines_probefix_table.py` | the datasets only (first 10,000 training images of each seed's order) | `results/input_baselines_probefix.{json,md}` |
| Table 2 | `scripts/score/table2_probefix.py` | `results/input_baselines_probefix.json`, `results/probefix_matrix_n5.json` | `results/table2_probefix.{md,json}` |
| Figure 7 | `scripts/figures/make_paper_figs.py` | `results/probefix_matrix_n5.json` | `figures/fig3_delta_over_random.pdf` |
| Appendix I (Fisher ratio, filter similarity) | `scripts/score/appendix_i_probefix.py` | the trained networks above on six datasets (test counts, `metrics.json`), `results/probefix_matrix_n5.json` | `results/appendix_i_probefix.{md,json}` |
| Table A.8, unsup-STDP column | `scripts/score/supervised_ceiling_5seed.py` (GPU) | pair-STDP rows of `results/probefix_matrix_n5.json`; the CNN is trained with torch seeds 0-4, the raw-pixel regression once | `results/supervised_ceiling_5seed.{md,json}` |

## 2. Voltage-STDP thresholds and spike plateau: Table 3, Table A.7(a), Figure 4(c), Section 4.4

| | |
|---|---|
| networks | `voltage_S_full_N400_seed{S}_probefix_{label}`; label = `thltd_m68`, `thltd_m65`, `thltd_m62`, `thltd_m60`, `thltd_m58`, `thltd_m55`, `thltd_m50` (θ₋ sweep), `thltp_m52` (θ₊ = -52 mV), `thltd_m65_thltp_m52` (Table 3 voltage-thresholds row), `plateau` (Table 3 spike-plateau row); reference = `voltage_S_full_N400_seed{S}_probefix_trained`, random = `voltage_S_full_N400_seed{S}_probefix_frozen` (section 1) |
| seeds | 0-4 |
| configuration | `run_configs/voltage_S_full_N400_probefix_{label}.yaml` [P2] |
| train | `bash scripts/train/probefix_voltage_thresholds_n5.sh "0" 4` |
| score | `scripts/score/probefix_voltage_thresholds_aggregate.py` |
| results | `results/probefix_voltage_thresholds_n5.{md,json}` |

## 3. Table 3, training-length row (Figure A.2, second row)

| | |
|---|---|
| networks | trained: `voltage_S_full_N400_seed{S}_vb3_trained60k` for S = 0, 3, 4 and `voltage_S_full_N400_seed{S}_mnist_vhz60000` for S = 1, 2; random: `voltage_S_full_N400_seed{S}_vb3_frozen60k` |
| seeds | 0-4 |
| configuration | `run_configs/voltage_S_full_N400_vb3_trained60k.yaml`, `voltage_S_full_N400_mnist_vhz60000.yaml`, `voltage_S_full_N400_vb3_frozen60k.yaml` [R8, R10]; tuned configuration, 60,000 training images |
| train, extract | `run_seeds.sh`, then `reextract_all.sh` |
| score | `scripts/score/reextract_probefix_score.py` (one value per network); the per-seed pairing is made by `scripts/figures/make_voltage_controls_figure.py` |
| results | `results/reextract_probefix_scores.{md,json}` (column "new"); `figures/fig_voltage_controls_data.json` |

## 4. Table 3, weight-management row; Table A.5

| | |
|---|---|
| networks, Table 3 | trained `voltage_S_full_N400_seed{S}_fixref_B_clopath_ref2400_ap0.002_h0.9`, random `voltage_S_full_N400_seed{S}_fixref_frozen_h0.9`, S = 0-4 |
| networks, Table A.5 (seed 0) | settings A: u_ref² = 15, 240, 600, 2,400, 6,000 → `fixref_C_ref{15,240,600,2400,6000}_ap0.002_h0.9`; 60 → `fixref_A_ref60_ap0.002_h0.9`. Settings B: 60 → `fixref_B_clopath_ap0.002_h0.9`; 600, 2,400, 6,000 → `fixref_B_clopath_ref{600,2400,6000}_ap0.002_h0.9`. Random: `fixref_frozen_h0.9`. Repeated on seeds 0-4: `fixref_C_ref240_ap0.002_h0.9` and `fixref_B_clopath_ref2400_ap0.002_h0.9`. (All names follow `voltage_S_full_N400_seed{S}_`.) |
| configuration | `run_configs/voltage_S_full_N400_fixref_*.yaml` [R4]; the other `fixref_*` files there are single-seed screens that the tables do not print |
| train, extract | `run_seeds.sh`, then `reextract_all.sh` |
| score | `scripts/score/reextract_probefix_score.py`; Table 3 pairing in `scripts/figures/make_voltage_controls_figure.py` |
| results | `results/reextract_probefix_scores.md` (section "Table 3 weight-management row; Appendix E (Table A.5)"), `.json` |

## 5. Spike plateau: Table A.6 and its note (Appendix E)

| | |
|---|---|
| networks (seed 0) | T: `plateau_T_ap{0.0003,0.001,0.003}`; R: `plateau_R_ap{0.0003,0.001,0.003}`; W: `plateau_W_x{0.03,0.1,0.3,1,3}`; random for T and R: `plateau_frozen_tuned`; random for W: `fixref_frozen_h0.9` (all after `voltage_S_full_N400_seed{S}_`) |
| repeated on seeds 0-4 | `plateau_T_ap0.001` (against `probefix_frozen`, section 1) and `plateau_W_x0.03` (against `fixref_frozen_h0.9`) |
| configuration | `run_configs/voltage_S_full_N400_plateau_*.yaml` [R3] |
| score | `scripts/score/reextract_probefix_score.py` → `results/reextract_probefix_scores.md` (section "Appendix E: spike plateau") |
| potentiation : depression (note of Table A.6) | `scripts/score/voltage_plateau_gate_probe.py --config-file run_configs/voltage_S_full_N400_plateau_T_ap0.001.yaml --plateau off --device cuda:0 --out results/voltage_plateau_gate_probe/tuned_off.json`; the same without `--plateau off` gives `tuned_on.json`; `..._plateau_R_ap0.001.yaml` gives `rest_on.json` and `..._plateau_W_x0.03.yaml` gives `W003_on.json`. Seed 0, first 1,000 training images, no trained run needed. |

## 6. Homeostasis on the tuned network (Appendix E)

| | |
|---|---|
| networks | `voltage_S_full_N400_seed{S}__tier0_{OFF,ty1,ty100,ty1000,ty10000}` (spike-count homeostasis; OFF = none), `voltage_S_full_N400_seed{S}_r1volt_ty100` (Clopath's voltage-averaged term); 60,000 training images |
| seeds | 0-4 |
| configuration | `run_configs/voltage_S_full_N400__tier0_*.yaml`, `voltage_S_full_N400_r1volt_ty100.yaml` [R2] |
| score | `scripts/score/reextract_probefix_score.py` → `results/reextract_probefix_scores.md` (section "Appendix E: homeostasis") |

## 7. Voltage-STDP tuning: Table A.4, Appendix C, Appendix I (hyperparameter transfer)

| | |
|---|---|
| networks | `voltage_S_full_N400_seed{S}__c_conv_parity` (the six copied values, seeds 0-4); `__d_phaseB1_B1_{1-5}` (L₁ scale with A_LTP, seed 0); `__e_phaseB2_B2_{1-5}` (one-value checks, seed 0); tuned: `__f_phaseB3_winner` (seeds 1-4) with `voltage_S_full_N400_seed0_vb3_trained60k` (seed 0); `__i4_smooth_int{16,32,64}` (input intensity, seed 0); `__mnist_adex_{baseline,a1,dT05,a1_dT05}` (neuron checks, seed 0). Default configuration: the voltage-STDP S_full networks of section 8. |
| configuration | `run_configs/voltage_S_full_N400__*.yaml` [R5]; each file holds the values of that run |
| score | `scripts/score/reextract_probefix_score.py` → `results/reextract_probefix_scores.md` (section "Appendix C"); the n = 5 means pool the per-network rows listed above |

## 8. Scaffold matrix at 60,000 images: Section 4.1, Table A.3, Figures 2 and 6, Section 4.6

| | |
|---|---|
| networks | pair and triplet: `{pair_exp,triplet}_S_{full,inh_only,thr_only,none}_N400_seed{S}` [S]; voltage-STDP (default configuration, adaptive-threshold decay 10⁴ ms): `voltage_S_{full,inh_only,thr_only,none}_N400_seed{S}__b_fix_tc1e4`, except S_full seed 0 = `voltage_S_full_N400_seed0__b_fix_tc1e4_60k` [R6]; 60 networks |
| seeds | 0-4 |
| configuration | `run_configs/{pair_exp,triplet}_S_{scaffold}_N400.yaml`, `run_configs/voltage_S_{scaffold}_N400__b_fix_tc1e4.yaml` and `voltage_S_full_N400__b_fix_tc1e4_60k.yaml` |
| extract | training-image counts: `scripts/extract/extract_train_features.py <run dirs> --n_subset 10000` (pair, triplet) and `bash scripts/train/voltage_scaffold_tc1e4_trainfeats.sh` (voltage) |
| score | `scripts/score/posthoc_both_conventions.py`, then `scaffold_readout_table.py`, then `identical_neuron_check.py` (the 20 pair and triplet networks without lateral inhibition; reads the datasets); `scripts/checks/verify_readout_bound.py` prints the Remark 1 check |
| results | `results/posthoc_both_conventions.csv`, `results/scaffold_readout_table.csv`, `results/scaffold_readout_summary.md`, `results/identical_neuron_check.csv` |
| figures | Figure 2 (`fig2_readout.pdf`) and Figure 6 (`fig5_scaffold_fragility.pdf`) from `scripts/figures/make_paper_figs.py` |

## 9. Matched random baseline at 60,000 images: Section 4.2, Figure 6 dashed line

| | |
|---|---|
| networks | trained: `pair_exp_S_full_N400_seed{S}` (section 8); random: `pair_exp_S_full_N400_seed{S}_frozen60k` [M] |
| seeds | 0-4 |
| configuration | `run_configs/pair_exp_S_full_N400.yaml`, `run_configs/pair_exp_S_full_N400_frozen60k.yaml` |
| extract | `scripts/extract/extract_train_features.py <run dirs> --n_subset 10000` |
| score | `scripts/score/matched_random_60k_table.py` |
| results | `results/matched_random_60k.json` |

## 10. Figure 3, receptive fields

`scripts/figures/make_paper_figs.py` reads `weights_final.npz` of
`pair_exp_S_full_N400_seed0`, `triplet_S_full_N400_seed0` and
`voltage_S_full_N400_seed0__b_fix_tc1e4_60k` (section 8). The random row is a
fresh draw from the initial distribution. Output `figures/fig7_receptive_fields.pdf`.

## 11. Training length: Section 4.5, Figure 5, Appendix G

| paper element | networks (seeds 0-4) | configuration | score | results |
|---|---|---|---|---|
| Figure 5, default configurations | `{pair_exp,triplet,voltage}_S_full_N400_seed{S}_flen{5k,10k,20k}` [R9]; 60,000 images = the S_full networks of section 8 | `run_configs/{rule}_S_full_N400_flen*.yaml` | `scripts/score/section45_probefix_mnist.py` | `results/section45_probefix_mnist.{md,json}` |
| Section 4.5, tuned voltage-STDP on MNIST | `voltage_S_full_N400_seed{S}_{vhz2000,vhz5000,mnist_vhz60000}` [R10] | `run_configs/voltage_S_full_N400_{vhz2000,vhz5000,mnist_vhz60000}.yaml` | same | same |
| check of the re-extraction (in the same table) | `voltage_S_full_N400_seed{S}_thltd_ref` and `_plateau_T_ap0.001`, seeds 0-2 [R1, R3], against `_probefix_trained` and `_probefix_plateau` | as listed | same | same |
| receptive-field cosine and clustering NMI | `voltage_S_full_N400_seed{S}_flen{5k,10k,20k}` and the voltage S_full networks of section 8 | as above | `scripts/score/section45_cosine_nmi.py` (reads `metrics.json`) | `results/section45_cosine_nmi.{md,json}` |
| Appendix G, scaffolds | `voltage_S_{full,inh_only,none,thr_only}_N400_seed{S}_peak5k`, `voltage_S_{inh_only,none,thr_only}_N400_seed{S}_mid20k` [R7]; S_full at 20,000 = `flen20k`; 60,000 = section 8 | `run_configs/voltage_S_*_N400_{peak5k,mid20k}.yaml` | `scripts/score/appendix_g_probefix.py` | `results/appendix_g_probefix.{md,json}` |
| Appendix G, datasets (tuned) | 2,000 and 5,000 images: `voltage_S_full_N400_seed{S}{_dataset}_vhz{2000,5000}` [R10]; 10,000: the voltage networks of section 1; 60,000: `_mnist_vhz60000`, `_emnist_letters_emnist_letters_vhz60000` [R10], `_nmnist__nmnist_n5`, `_fmnist__f3_fmnist_voltage_tuned`, `_organamnist__n5_organa_tuned` (seed 0: `__organa_smoke_tuned`), `_bloodmnist__n5_blood_tuned` (seed 0: `__blood_smoke_tuned`) [R11] | as listed in `run_configs/INDEX.md` | same | same; the `runs` fields of the JSON name every network of every cell |

All rows except the cosine and NMI use re-extracted counts (`*_pf.npz`):
`section45_probefix_mnist.py` fits the probe itself, and `appendix_g_probefix.py`
takes the values from `results/reextract_probefix_scores.json` (written by
`scripts/score/reextract_probefix_score.py`) and its 10,000-image points from
`results/probefix_matrix_n5.json`. The pair- and triplet-STDP 60,000-image networks
were not re-extracted; they use the training-image counts of section 8 with their
original test pass. Figure 5 (`figures/fig4_forgetting.pdf`) is drawn by
`scripts/figures/make_paper_figs.py` from `results/section45_probefix_mnist.json`.

## 12. Convolutional networks: Table 3 architecture row, Section 4.3, Appendix I

Voltage-STDP convolutional network (seeds 1-5):

| | |
|---|---|
| trained filters + surrogate readout | `checkpoints/conv2l_voltage_clopath_faithful_mnist_paper_seed{S}`, S = 1-5: `SEEDS="1 2 3 4 5" bash scripts/train/run_seeds.sh conv_stdp` (= `conv/conv2layer_voltage_stdp.py --config conv/configs/conv_voltage_config.yaml --seed S --name-suffix _paper_seedS`) |
| fresh surrogate readout on those filters | `checkpoints/..._TRAINEDREADOUT_seed{S}`, S = 1-4 (seed 5 uses the readout of `conv_stdp`): `run_seeds.sh conv_trainedreadout` |
| surrogate readout on random filters | `checkpoints/..._RANDOMFILTERS_seed{S}`, S = 1-5: `run_seeds.sh conv_randomfilters` (configs `conv/configs/conv_seed5/randomfilters_seed{1-5}.yaml`, `run_configs/conv/randomfilters_seed5.yaml`) |
| linear readout (Table 3 architecture row) | `run_seeds.sh conv_svm` (runs `scripts/extract/conv_svm_readout.py` and `scripts/score/classify_feats_torch.py`; 12,000 training and 10,000 test images) → `results/conv_svm_{trained,random}_s{1-5}.json`; then `scripts/score/conv_arch_logreg_n5.py` → `results/conv_arch_logreg_n5.{md,json}` |
| surrogate readout, five seeds (96.9 against 97.3) | `scripts/score/conv_surrogate_readout_n5.py --log_dir DIR` → `results/conv_surrogate_readout_n5.{md,json}` |

`conv_surrogate_readout_n5.py` reads `trainedreadout_seed{1-4}.log`,
`paper_seed5.log` and `randomfilters_seed{1-5}.log` from `DIR`. With
`run_seeds.sh` (default `LOG_DIR=logs_run_seeds`) these logs are written under
other names; copy them first:

```bash
mkdir -p results/logs_conv5
for s in 1 2 3 4; do cp logs_run_seeds/conv_trainedreadout_seed$s.log results/logs_conv5/trainedreadout_seed$s.log; done
cp logs_run_seeds/conv_stdp_seed5.log results/logs_conv5/paper_seed5.log
for s in 1 2 3 4 5; do cp logs_run_seeds/conv_randomfilters_seed$s.log results/logs_conv5/randomfilters_seed$s.log; done
```

Pair-STDP convolutional control (Section 4.3, "0.4 ± 0.4 points", seeds 0-4):
the weights `conv2l_stdp_ckpt_pairgrade_seed{0-4}.pt` at the repository root were
trained with `conv/conv2layer_stdp_surrogate.py`; the command of that training was
not recorded and no launcher for it is shipped. Given the weights,
`bash scripts/train/conv_pair_trained_vs_random.sh cuda:0` writes
`results/conv_feats_pairN5_{trained,random}_s{0-4}.npz`, and
`scripts/score/conv_pair_n5_table.py` writes `results/conv_pair_n5.json`.

## 13. Gate measurements: Table A.7(b), Figure 4(b), Section 4.4

These run the plasticity rule for the first 1,000 training images of seed 0 and
need no trained run.

| paper element | command | results |
|---|---|---|
| Table A.7(b), Figure 4(b) | `scripts/score/voltage_gate_saturation_probe.py --config-file results/gate_probes_2026-10/<p>.yaml --seed 0 --n-train 1000 --device cuda:0 --out results/gate_probes_2026-10/<p>.json` for `<p>` = `p0_tuned_repro`, `p1_no_inhibition`, `p2_thltd_m65`, `p3_thltd_m60`, `p4_thltd_m58`, `p5_thltd_m50`, `p6_thltp_m52`; then `scripts/score/make_section44_table.py` | `results/gate_probes_2026-10/*.json`, `results/section44_table.md` |
| Section 4.4, "267 to 319 times" (default configuration) | the same probe with `--config-file run_configs/voltage_S_full_N400.yaml --seed S --out results/gate_scaffold_seedS.json`, S = 0-4 [V0] | `results/gate_scaffold_seed{0-4}.json` |
| Figure 4 | `CLOPATH_MODELDB_DIR=... scripts/figures/make_section44_mechanism_figure.py`; reads `run_configs/voltage_S_full_N400_vb3_trained.yaml` [V1], the gate-probe JSONs, `results/section44_table.md` and `results/probefix_voltage_thresholds_n5.json` | `figures/fig_section44_mechanism.pdf` |

## 14. Exact reimplementation of Clopath's neuron: Table 3, Tables A.9 and A.10, Appendix J

Run directories are `experiments/runs/clopath_port/<family>/seed{S}`; each holds a
`meta.json` with every setting. The seed-0 `meta.json` of each family is in
`run_configs/clopath_port/`.

| step | command | results |
|---|---|---|
| calibration with STDP disabled (seed 0, 1,000 images, no labels) | `scripts/train/clopath_port_mnist.py calibrate --seed 0 --n_images 1000 --g 0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.35,0.4,0.5,0.6,0.8 --k 0,0.03,0.1,0.3,1,3,10,30,100,300 --device cuda:0 --out results/clopath_port_calibration` | `results/clopath_port_calibration/calibration.json` |
| pre-registered choice of the 3 x 3 grid | `scripts/train/clopath_port_select_grid.py results/clopath_port_calibration/calibration.json --out results/clopath_port_calibration/grid.json` | `results/clopath_port_calibration/grid.json` |
| grid10k: 9 settings x 2 amplitude scales, trained, and the 9 STDP-disabled networks (Tables 3 and A.9) | `SEEDS="0 1 2 3 4" bash scripts/train/run_seeds.sh port_grid10k` | `experiments/runs/clopath_port/grid10k/seed{0-4}` |
| explore_k0: no lateral inhibition (Appendix J) | `SEEDS="0 1 2 3 4" bash scripts/train/run_seeds.sh port_explore_k0` (grid `run_configs/clopath_port/grid_explore_k0.json`, 5,000 test images) | `experiments/runs/clopath_port/explore_k0/seed{0-4}` |
| check that the sweep code reproduces grid10k (required by the next step) | `scripts/train/clopath_port_mnist.py train --seed 0 --grid results/clopath_port_calibration/grid.json --theta_tar 60 --no_random --prereg results/voltage_clopath_port_thetatar_PREREG.md --n_train 10000 --n_test 10000 --device cuda:0 --out experiments/runs/clopath_port/thetatar_repro/seed0` | `experiments/runs/clopath_port/thetatar_repro/seed0`; `results/clopath_port_thetatar/check_*.json` |
| thetatar10k: u_ref² = 240, 960, 3,840, 15,360 mV², trained only (Table A.10) | seeds 0-2: `bash scripts/train/clopath_port_thetatar_sweep.sh` (set `GPU0_UUID` for your machine, see its header); seeds 3 and 4: `scripts/train/clopath_port_mnist.py train --seed S --grid results/clopath_port_calibration/grid.json --theta_tar 240,960 --no_random --prereg results/voltage_clopath_port_thetatar_PREREG.md --n_train 10000 --n_test 10000 --device cuda:0 --out experiments/runs/clopath_port/thetatar10k/seedS_a`, and the same with `--theta_tar 3840,15360` and `seedS_b` | `experiments/runs/clopath_port/thetatar10k/seed{0_all,1_a,1_b,2_all,3_a,3_b,4_a,4_b}` |
| re-extraction | `bash scripts/extract/reextract_all.sh -f results/run_lists/jobs_port.txt` (it lists `seed0_all` as two halves, `seed0_all_a` and `seed0_all_b`; their `meta.json` points back to `seed0_all`) | `*_pf.npz` in each run directory |
| linear probe, trained against STDP disabled (same seed, g and k_inh) | `scripts/score/clopath_port_probefix_score.py` | `results/clopath_port_probefix_scores.{md,json}` |
| counts of Appendix J (silent settings, weights at the bound, best settings, probe ranges) | `scripts/score/appendix_j_counts.py` (uses the per-network statistics of `scripts/score/clopath_port_thetatar_score.py`) | `results/appendix_j_counts.{md,json}` |
| equivalence with Clopath's code (831 spikes, 2.3 × 10⁻⁷ mV, 14 protocols) | `CLOPATH_MODELDB_DIR=... scripts/checks/clopath_port_equivalence.py` (needs GNU Octave) | `results/clopath_port_validation/equivalence.json` |
| validation in Clopath's own setting (10 seeds) and pairing protocols | `scripts/checks/clopath_port_validate.py` | `results/clopath_port_validation/validation.json` |
| read-out delay of `RFdevelop.m` (+28% at 0.1 Hz) | `scripts/checks/clopath_port_delay_check.py` | `results/clopath_port_validation/delay_check.json` |

The pre-registrations are `results/voltage_clopath_neuron_port_PREREG.md` (grid10k)
and `results/voltage_clopath_port_thetatar_PREREG.md` (thetatar10k). The launchers
check the SHA-256 of these files, so they must not be edited.

## 15. Positive control: Appendix D, Figure A.1

| | |
|---|---|
| production plasticity step driven by pairing protocols | `scripts/checks/clopath_positive_control.py` → `results/clopath_positive_control.json` (described in `results/clopath_positive_control.md`) |
| independent implementation of the equations, and the figure | `scripts/figures/clopath_curve_reproduction.py` → `results/clopath_curve_reproduction.json`, `figures/fig6_clopath_curves.pdf` |

No trained network is needed.

## 16. Tables and figures without run data

- Table 1: dataset properties. The majority-class column is the most frequent class
  of each official test split.
- Table A.1: the hyperparameters are in `run_configs/` (for example
  `run_configs/{pair_exp,triplet,voltage}_S_full_N400_probefix_trained.yaml`),
  `experiments/configs/` and `conv/configs/conv_voltage_config.yaml`.
- Figures 1 and 8 are schematics (`scripts/figures/make_paper_figs.py`,
  `scripts/figures/make_protocol_figure.py`).
- Figure A.2 is drawn by `scripts/figures/make_voltage_controls_figure.py` from
  `results/probefix_voltage_thresholds_n5.json` and
  `results/reextract_probefix_scores.json` (sections 2 to 4).
