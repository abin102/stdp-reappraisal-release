# Run configurations

Every FC network behind a number in the current paper was trained by `experiments/scripts/run_single.py`'s
`train_and_eval`. Each family below is one configuration run over several seeds. Its file here is the
`config.yaml` the run saved for the family's first seed, with only the machine-specific
`experiment.device` line removed. For every family, the configs of all other seeds differ from this file
only in `experiment.seed`, `experiment.name` and `experiment.device` (checked for all 673 run
directories listed below on 2026-10-24: `scripts/train/run_config.py` reproduces each saved config
exactly, given the original device).

## How to run

```bash
# one seed of one family (writes experiments/runs/<name with _seed<S>>)
PYTHONPATH=. python scripts/train/run_config.py run_configs/<file>.yaml --seed S [--device cuda:0]
# several families x seeds, one worker per GPU
SEEDS="0 1 2 3 4" GPUS="0 1" bash scripts/train/run_seeds.sh <file> [<file> ...]
# then, for the families scored from re-extracted features (groups R below and the Clopath port)
GPUS="0 1" bash scripts/extract/reextract_all.sh -f results/run_lists/jobs_runsingle.txt \
    -f results/run_lists/jobs_homeo.txt -f results/run_lists/jobs_seeds34.txt -f results/run_lists/jobs_port.txt
```

`<file>` is the name in the first column (without `.yaml`). A run with `completed.flag` is skipped.
Run names are those the scorers in `scripts/score/` read.

Notes:

1. The shipped `run_single.py` presents the test set in a fixed shuffled order (seed + 3000) and, when
   `evaluation.save_train_features` is true, saves the training-image counts itself. Runs made before that
   change were re-extracted (`*_pf.npz`, groups R and the Clopath port) or had training features
   extracted afterwards (groups S, M and R6, `scripts/extract/extract_train_features.py`; launcher for
   R6: `scripts/train/voltage_scaffold_tc1e4_trainfeats.sh`). A retrained run's `test_spike_counts.npz` is
   therefore in shuffled order, with its labels saved beside it.
2. Four families have their seed 0 under a different name (column 'note'). The configurations are identical
   apart from seed, name and device (checked); the scorers read the names as listed. To recreate such a
   seed 0, run the aliased family's file with `--seed 0`.
3. Fig 3 (`scripts/figures/make_paper_figs.py`) draws the voltage receptive fields from
   `voltage_S_full_N400_seed0__b_fix_tc1e4_60k`, the tau_theta = 1e4 network that Table A.3, Sec 4.6 and
   Fig 6 also use (group R6). The tau_theta = 1e6 network (group V0) is used only by the gate probe of
   Sec 4.4.
4. Single-seed families (Appendix C search cells, weight-management and plateau screens) are screens
   reported at one seed, as in the paper.

## P1. Probe-fix matrix: 3 rules x 8 datasets, trained and STDP-disabled, 10k images

Paper: Table 2, Fig 7, Sec 4.2, 4.3 (FMNIST 64.2 vs 64.5), 4.7, Table A.2, App H (STDP column), App I; abstract gains.

How: launcher `scripts/train/probefix_matrix_n5.sh` (all 240 runs), or run_seeds.sh per family; scorer `scripts/score/probefix_aggregate.py`. The voltage MNIST pair is also the reference of P2..

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `pair_exp_S_full_N400_bloodmnist_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_bloodmnist_probefix_frozen` | 0-4 | bloodmnist | 10000 |  |
| `pair_exp_S_full_N400_bloodmnist_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_bloodmnist_probefix_trained` | 0-4 | bloodmnist | 10000 |  |
| `pair_exp_S_full_N400_emnist_letters_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_emnist_letters_probefix_frozen` | 0-4 | emnist_letters | 10000 |  |
| `pair_exp_S_full_N400_emnist_letters_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_emnist_letters_probefix_trained` | 0-4 | emnist_letters | 10000 |  |
| `pair_exp_S_full_N400_fmnist_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_fmnist_probefix_frozen` | 0-4 | fmnist | 10000 |  |
| `pair_exp_S_full_N400_fmnist_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_fmnist_probefix_trained` | 0-4 | fmnist | 10000 |  |
| `pair_exp_S_full_N400_nmnist_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_nmnist_probefix_frozen` | 0-4 | nmnist | 10000 |  |
| `pair_exp_S_full_N400_nmnist_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_nmnist_probefix_trained` | 0-4 | nmnist | 10000 |  |
| `pair_exp_S_full_N400_organamnist_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_organamnist_probefix_frozen` | 0-4 | organamnist | 10000 |  |
| `pair_exp_S_full_N400_organamnist_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_organamnist_probefix_trained` | 0-4 | organamnist | 10000 |  |
| `pair_exp_S_full_N400_pathmnist_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_pathmnist_probefix_frozen` | 0-4 | pathmnist | 10000 |  |
| `pair_exp_S_full_N400_pathmnist_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_pathmnist_probefix_trained` | 0-4 | pathmnist | 10000 |  |
| `pair_exp_S_full_N400_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_probefix_frozen` | 0-4 | mnist | 10000 |  |
| `pair_exp_S_full_N400_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_probefix_trained` | 0-4 | mnist | 10000 |  |
| `pair_exp_S_full_N400_tissuemnist_probefix_frozen` | `experiments/runs/pair_exp_S_full_N400_seed{S}_tissuemnist_probefix_frozen` | 0-4 | tissuemnist | 10000 |  |
| `pair_exp_S_full_N400_tissuemnist_probefix_trained` | `experiments/runs/pair_exp_S_full_N400_seed{S}_tissuemnist_probefix_trained` | 0-4 | tissuemnist | 10000 |  |
| `triplet_S_full_N400_bloodmnist_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_bloodmnist_probefix_frozen` | 0-4 | bloodmnist | 10000 |  |
| `triplet_S_full_N400_bloodmnist_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_bloodmnist_probefix_trained` | 0-4 | bloodmnist | 10000 |  |
| `triplet_S_full_N400_emnist_letters_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_emnist_letters_probefix_frozen` | 0-4 | emnist_letters | 10000 |  |
| `triplet_S_full_N400_emnist_letters_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_emnist_letters_probefix_trained` | 0-4 | emnist_letters | 10000 |  |
| `triplet_S_full_N400_fmnist_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_fmnist_probefix_frozen` | 0-4 | fmnist | 10000 |  |
| `triplet_S_full_N400_fmnist_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_fmnist_probefix_trained` | 0-4 | fmnist | 10000 |  |
| `triplet_S_full_N400_nmnist_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_nmnist_probefix_frozen` | 0-4 | nmnist | 10000 |  |
| `triplet_S_full_N400_nmnist_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_nmnist_probefix_trained` | 0-4 | nmnist | 10000 |  |
| `triplet_S_full_N400_organamnist_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_organamnist_probefix_frozen` | 0-4 | organamnist | 10000 |  |
| `triplet_S_full_N400_organamnist_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_organamnist_probefix_trained` | 0-4 | organamnist | 10000 |  |
| `triplet_S_full_N400_pathmnist_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_pathmnist_probefix_frozen` | 0-4 | pathmnist | 10000 |  |
| `triplet_S_full_N400_pathmnist_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_pathmnist_probefix_trained` | 0-4 | pathmnist | 10000 |  |
| `triplet_S_full_N400_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_probefix_frozen` | 0-4 | mnist | 10000 |  |
| `triplet_S_full_N400_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_probefix_trained` | 0-4 | mnist | 10000 |  |
| `triplet_S_full_N400_tissuemnist_probefix_frozen` | `experiments/runs/triplet_S_full_N400_seed{S}_tissuemnist_probefix_frozen` | 0-4 | tissuemnist | 10000 |  |
| `triplet_S_full_N400_tissuemnist_probefix_trained` | `experiments/runs/triplet_S_full_N400_seed{S}_tissuemnist_probefix_trained` | 0-4 | tissuemnist | 10000 |  |
| `voltage_S_full_N400_bloodmnist_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_bloodmnist_probefix_frozen` | 0-4 | bloodmnist | 10000 |  |
| `voltage_S_full_N400_bloodmnist_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_bloodmnist_probefix_trained` | 0-4 | bloodmnist | 10000 |  |
| `voltage_S_full_N400_emnist_letters_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_emnist_letters_probefix_frozen` | 0-4 | emnist_letters | 10000 |  |
| `voltage_S_full_N400_emnist_letters_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_emnist_letters_probefix_trained` | 0-4 | emnist_letters | 10000 |  |
| `voltage_S_full_N400_fmnist_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_fmnist_probefix_frozen` | 0-4 | fmnist | 10000 |  |
| `voltage_S_full_N400_fmnist_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_fmnist_probefix_trained` | 0-4 | fmnist | 10000 |  |
| `voltage_S_full_N400_nmnist_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_nmnist_probefix_frozen` | 0-4 | nmnist | 10000 |  |
| `voltage_S_full_N400_nmnist_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_nmnist_probefix_trained` | 0-4 | nmnist | 10000 |  |
| `voltage_S_full_N400_organamnist_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_organamnist_probefix_frozen` | 0-4 | organamnist | 10000 |  |
| `voltage_S_full_N400_organamnist_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_organamnist_probefix_trained` | 0-4 | organamnist | 10000 |  |
| `voltage_S_full_N400_pathmnist_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_pathmnist_probefix_frozen` | 0-4 | pathmnist | 10000 |  |
| `voltage_S_full_N400_pathmnist_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_pathmnist_probefix_trained` | 0-4 | pathmnist | 10000 |  |
| `voltage_S_full_N400_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_frozen` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_trained` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_tissuemnist_probefix_frozen` | `experiments/runs/voltage_S_full_N400_seed{S}_tissuemnist_probefix_frozen` | 0-4 | tissuemnist | 10000 |  |
| `voltage_S_full_N400_tissuemnist_probefix_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_tissuemnist_probefix_trained` | 0-4 | tissuemnist | 10000 |  |

## P2. Voltage-STDP threshold and spike-plateau settings, 10k images

Paper: Table 3 voltage-thresholds and spike-plateau rows, Sec 4.4, Fig 4c, Table A.7a, Fig A.2 rows 3-4.

How: launcher `scripts/train/probefix_voltage_thresholds_n5.sh`, or run_seeds.sh per family; scorer `scripts/score/probefix_voltage_thresholds_aggregate.py`.

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_probefix_plateau` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_plateau` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m50` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m50` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m55` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m55` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m58` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m58` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m60` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m60` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m62` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m62` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m65_thltp_m52` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m65_thltp_m52` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m65` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m65` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltd_m68` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltd_m68` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_probefix_thltp_m52` | `experiments/runs/voltage_S_full_N400_seed{S}_probefix_thltp_m52` | 0-4 | mnist | 10000 |  |

## S. Scaffold matrix, pair and triplet, 60k images

Paper: Table A.3, Fig 2, Fig 6, Sec 4.1, 4.6, Discussion; the pair S_full family is also the trained side of Sec 4.2's matched-random comparison.

How: run_seeds.sh, then `scripts/extract/extract_train_features.py DIR... --n_subset 10000` (configs predate `evaluation.save_train_features`); scorers `scaffold_readout_table.py`, `posthoc_both_conventions.py`, `identical_neuron_check.py`, `matched_random_60k_table.py`.

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `pair_exp_S_full_N400` | `experiments/runs/pair_exp_S_full_N400_seed{S}` | 0-4 | mnist | 60000 |  |
| `pair_exp_S_inh_only_N400` | `experiments/runs/pair_exp_S_inh_only_N400_seed{S}` | 0-4 | mnist | 60000 |  |
| `pair_exp_S_none_N400` | `experiments/runs/pair_exp_S_none_N400_seed{S}` | 0-4 | mnist | 60000 |  |
| `pair_exp_S_thr_only_N400` | `experiments/runs/pair_exp_S_thr_only_N400_seed{S}` | 0-4 | mnist | 60000 |  |
| `triplet_S_full_N400` | `experiments/runs/triplet_S_full_N400_seed{S}` | 0-4 | mnist | 60000 |  |
| `triplet_S_inh_only_N400` | `experiments/runs/triplet_S_inh_only_N400_seed{S}` | 0-4 | mnist | 60000 |  |
| `triplet_S_none_N400` | `experiments/runs/triplet_S_none_N400_seed{S}` | 0-4 | mnist | 60000 |  |
| `triplet_S_thr_only_N400` | `experiments/runs/triplet_S_thr_only_N400_seed{S}` | 0-4 | mnist | 60000 |  |

## M. Matched random baseline at 60k (pair, STDP disabled)

Paper: Sec 4.2 (69.7 +- 0.9, +16.2), Fig 6 dashed line, Sec 4.7, introduction.

How: run_seeds.sh, then `scripts/extract/extract_train_features.py DIR... --n_subset 10000`; scorer `scripts/score/matched_random_60k_table.py`.

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `pair_exp_S_full_N400_frozen60k` | `experiments/runs/pair_exp_S_full_N400_seed{S}_frozen60k` | 0-4 | mnist | 60000 |  |

## V0. Voltage-STDP default configuration, tau_theta = 1e6 (superseded scaffold cell)

Paper: `--config-file` of `scripts/score/voltage_gate_saturation_probe.py` (Sec 4.4 '267 to 319 times', seeds 0-4; `results/gate_scaffold_seed*.json`).

How: the gate probe needs only the config file (no trained run).

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400` | `experiments/runs/voltage_S_full_N400_seed{S}` | 0-4 | mnist | 60000 |  |

## V1. Tuned voltage-STDP configuration (config source only)

Paper: Fig 4 (scripts/figures/make_section44_mechanism_figure.py reads this file directly).

How: needs no training: the figure reads `run_configs/voltage_S_full_N400_vb3_trained.yaml` itself.

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_vb3_trained` | `experiments/runs/voltage_S_full_N400_seed{S}_vb3_trained` | 0 | mnist | 10000 |  |

## R. Families scored from re-extracted features

Listed in `results/run_lists/jobs_{runsingle,homeo,seeds34}.txt`; scorer `scripts/score/reextract_probefix_score.py`
(single probe on `*_pf.npz`), which feeds `appendix_g_probefix.py` and `section45_probefix_mnist.py`.
How: run_seeds.sh, then `scripts/extract/reextract_all.sh` on the run directories.

### R1. Validation: re-extraction vs fresh probe-fix runs

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_thltd_ref` | `experiments/runs/voltage_S_full_N400_seed{S}_thltd_ref` | 0-2 | mnist | 10000 |  |

### R2. Appendix E: homeostasis on the tuned network (60k)

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400__tier0_OFF` | `experiments/runs/voltage_S_full_N400_seed{S}__tier0_OFF` | 0-4 | mnist | 60000 |  |
| `voltage_S_full_N400__tier0_ty10000` | `experiments/runs/voltage_S_full_N400_seed{S}__tier0_ty10000` | 0-4 | mnist | 60000 |  |
| `voltage_S_full_N400__tier0_ty1000` | `experiments/runs/voltage_S_full_N400_seed{S}__tier0_ty1000` | 0-4 | mnist | 60000 |  |
| `voltage_S_full_N400__tier0_ty100` | `experiments/runs/voltage_S_full_N400_seed{S}__tier0_ty100` | 0-4 | mnist | 60000 |  |
| `voltage_S_full_N400__tier0_ty1` | `experiments/runs/voltage_S_full_N400_seed{S}__tier0_ty1` | 0-4 | mnist | 60000 |  |
| `voltage_S_full_N400_r1volt_ty100` | `experiments/runs/voltage_S_full_N400_seed{S}_r1volt_ty100` | 0-4 | mnist | 60000 |  |

### R3. Appendix E: spike plateau (Table A.6); Table 3 spike-plateau row

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_plateau_R_ap0.0003` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_R_ap0.0003` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_R_ap0.001` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_R_ap0.001` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_R_ap0.003` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_R_ap0.003` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_T_ap0.0003` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_T_ap0.0003` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_T_ap0.001` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_T_ap0.001` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_T_ap0.003` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_T_ap0.003` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_W_x0.03` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_W_x0.03` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_W_x0.1` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_W_x0.1` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_W_x0.3` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_W_x0.3` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_W_x1` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_W_x1` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_W_x3` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_W_x3` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_plateau_frozen_tuned` | `experiments/runs/voltage_S_full_N400_seed{S}_plateau_frozen_tuned` | 0 | mnist | 10000 |  |

### R4. Table 3 weight-management row; Appendix E (Table A.5); Sec 4.3 weight management

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_fixref_A_ref60_ap0.002_h0.5` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_A_ref60_ap0.002_h0.5` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_A_ref60_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_A_ref60_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_A_ref60_ap0.005_h0.5` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_A_ref60_ap0.005_h0.5` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_A_ref60_ap0.005_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_A_ref60_ap0.005_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_A_ref60_ap0.02_h0.5` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_A_ref60_ap0.02_h0.5` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_A_ref60_ap0.02_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_A_ref60_ap0.02_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ap0.002_h0.5` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ap0.002_h0.5` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ap0.005_h0.5` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ap0.005_h0.5` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ap0.005_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ap0.005_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ap0.02_h0.5` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ap0.02_h0.5` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ap0.02_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ap0.02_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ref2400_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ref2400_ap0.002_h0.9` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ref6000_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ref6000_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_B_clopath_ref600_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_B_clopath_ref600_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_C_ref15_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_C_ref15_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_C_ref2400_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_C_ref2400_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_C_ref240_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_C_ref240_ap0.002_h0.9` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_C_ref6000_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_C_ref6000_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_C_ref600_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_C_ref600_ap0.002_h0.9` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_frozen_h0.5` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_frozen_h0.5` | 0 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_frozen_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_frozen_h0.9` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_fixref_repro_auto_ap0.002_h0.9` | `experiments/runs/voltage_S_full_N400_seed{S}_fixref_repro_auto_ap0.002_h0.9` | 0 | mnist | 10000 |  |

### R5. Appendix C: hyperparameter search (Table A.4)

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400__c_conv_parity` | `experiments/runs/voltage_S_full_N400_seed{S}__c_conv_parity` | 0-4 | mnist | 60000 |  |
| `voltage_S_full_N400__d_phaseB1_B1_1` | `experiments/runs/voltage_S_full_N400_seed{S}__d_phaseB1_B1_1` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__d_phaseB1_B1_2` | `experiments/runs/voltage_S_full_N400_seed{S}__d_phaseB1_B1_2` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__d_phaseB1_B1_3` | `experiments/runs/voltage_S_full_N400_seed{S}__d_phaseB1_B1_3` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__d_phaseB1_B1_4` | `experiments/runs/voltage_S_full_N400_seed{S}__d_phaseB1_B1_4` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__d_phaseB1_B1_5` | `experiments/runs/voltage_S_full_N400_seed{S}__d_phaseB1_B1_5` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__e_phaseB2_B2_1` | `experiments/runs/voltage_S_full_N400_seed{S}__e_phaseB2_B2_1` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__e_phaseB2_B2_2` | `experiments/runs/voltage_S_full_N400_seed{S}__e_phaseB2_B2_2` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__e_phaseB2_B2_3` | `experiments/runs/voltage_S_full_N400_seed{S}__e_phaseB2_B2_3` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__e_phaseB2_B2_4` | `experiments/runs/voltage_S_full_N400_seed{S}__e_phaseB2_B2_4` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__e_phaseB2_B2_5` | `experiments/runs/voltage_S_full_N400_seed{S}__e_phaseB2_B2_5` | 0 | mnist | 30000 |  |
| `voltage_S_full_N400__f_phaseB3_winner` | `experiments/runs/voltage_S_full_N400_seed{S}__f_phaseB3_winner` | 1-4 | mnist | 60000 | seed 0 = `voltage_S_full_N400_seed0_vb3_trained60k` |
| `voltage_S_full_N400__i4_smooth_int16` | `experiments/runs/voltage_S_full_N400_seed{S}__i4_smooth_int16` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__i4_smooth_int32` | `experiments/runs/voltage_S_full_N400_seed{S}__i4_smooth_int32` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__i4_smooth_int64` | `experiments/runs/voltage_S_full_N400_seed{S}__i4_smooth_int64` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__mnist_adex_a1_dT05` | `experiments/runs/voltage_S_full_N400_seed{S}__mnist_adex_a1_dT05` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__mnist_adex_a1` | `experiments/runs/voltage_S_full_N400_seed{S}__mnist_adex_a1` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__mnist_adex_baseline` | `experiments/runs/voltage_S_full_N400_seed{S}__mnist_adex_baseline` | 0 | mnist | 60000 |  |
| `voltage_S_full_N400__mnist_adex_dT05` | `experiments/runs/voltage_S_full_N400_seed{S}__mnist_adex_dT05` | 0 | mnist | 60000 |  |

### R6. Section 4.6 / Figure 6 / Table A.3: voltage-STDP scaffolds at 60k

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400__b_fix_tc1e4_60k` | `experiments/runs/voltage_S_full_N400_seed{S}__b_fix_tc1e4_60k` | 0 | mnist | 60000 | seed 0 of `voltage_S_full_N400_seed{S}__b_fix_tc1e4` |
| `voltage_S_full_N400__b_fix_tc1e4` | `experiments/runs/voltage_S_full_N400_seed{S}__b_fix_tc1e4` | 1-4 | mnist | 60000 | seed 0 = `voltage_S_full_N400_seed0__b_fix_tc1e4_60k` |
| `voltage_S_inh_only_N400__b_fix_tc1e4` | `experiments/runs/voltage_S_inh_only_N400_seed{S}__b_fix_tc1e4` | 0-4 | mnist | 60000 |  |
| `voltage_S_none_N400__b_fix_tc1e4` | `experiments/runs/voltage_S_none_N400_seed{S}__b_fix_tc1e4` | 0-4 | mnist | 60000 |  |
| `voltage_S_thr_only_N400__b_fix_tc1e4` | `experiments/runs/voltage_S_thr_only_N400_seed{S}__b_fix_tc1e4` | 0-4 | mnist | 60000 |  |

### R7. Section 4.6 / Appendix G: voltage-STDP scaffolds at 5k and 20k

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_peak5k` | `experiments/runs/voltage_S_full_N400_seed{S}_peak5k` | 0-4 | mnist | 5000 |  |
| `voltage_S_inh_only_N400_mid20k` | `experiments/runs/voltage_S_inh_only_N400_seed{S}_mid20k` | 0-4 | mnist | 20000 |  |
| `voltage_S_inh_only_N400_peak5k` | `experiments/runs/voltage_S_inh_only_N400_seed{S}_peak5k` | 0-4 | mnist | 5000 |  |
| `voltage_S_none_N400_mid20k` | `experiments/runs/voltage_S_none_N400_seed{S}_mid20k` | 0-4 | mnist | 20000 |  |
| `voltage_S_none_N400_peak5k` | `experiments/runs/voltage_S_none_N400_seed{S}_peak5k` | 0-4 | mnist | 5000 |  |
| `voltage_S_thr_only_N400_mid20k` | `experiments/runs/voltage_S_thr_only_N400_seed{S}_mid20k` | 0-4 | mnist | 20000 |  |
| `voltage_S_thr_only_N400_peak5k` | `experiments/runs/voltage_S_thr_only_N400_seed{S}_peak5k` | 0-4 | mnist | 5000 |  |

### R8. Table 3 training-length row (60k, trained vs STDP disabled); Fig A.2 row 2

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_fmnist_vb3_frozen60k` | `experiments/runs/voltage_S_full_N400_seed{S}_fmnist_vb3_frozen60k` | 0-4 | fmnist | 60000 | the paper uses seeds 0-4 |
| `voltage_S_full_N400_fmnist_vb3_trained60k` | `experiments/runs/voltage_S_full_N400_seed{S}_fmnist_vb3_trained60k` | 0-4 | fmnist | 60000 | ; seeds 1-2 = `__f_phaseB3_winner` |
| `voltage_S_full_N400_vb3_frozen60k` | `experiments/runs/voltage_S_full_N400_seed{S}_vb3_frozen60k` | 0-9 | mnist | 60000 | the paper uses seeds 0-4 |
| `voltage_S_full_N400_vb3_trained60k` | `experiments/runs/voltage_S_full_N400_seed{S}_vb3_trained60k` | 0,3,4,5,6,7,8,9 | mnist | 60000 | seed 0 of `voltage_S_full_N400_seed{S}__f_phaseB3_winner`; seeds 1-2 = `__f_phaseB3_winner` |

### R9. Section 4.5 / Figure 5: training length, default configurations

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `pair_exp_S_full_N400_flen10k` | `experiments/runs/pair_exp_S_full_N400_seed{S}_flen10k` | 0-4 | mnist | 10000 |  |
| `pair_exp_S_full_N400_flen20k` | `experiments/runs/pair_exp_S_full_N400_seed{S}_flen20k` | 0-4 | mnist | 20000 |  |
| `pair_exp_S_full_N400_flen5k` | `experiments/runs/pair_exp_S_full_N400_seed{S}_flen5k` | 0-4 | mnist | 5000 |  |
| `triplet_S_full_N400_flen10k` | `experiments/runs/triplet_S_full_N400_seed{S}_flen10k` | 0-4 | mnist | 10000 |  |
| `triplet_S_full_N400_flen20k` | `experiments/runs/triplet_S_full_N400_seed{S}_flen20k` | 0-4 | mnist | 20000 |  |
| `triplet_S_full_N400_flen5k` | `experiments/runs/triplet_S_full_N400_seed{S}_flen5k` | 0-4 | mnist | 5000 |  |
| `voltage_S_full_N400_flen10k` | `experiments/runs/voltage_S_full_N400_seed{S}_flen10k` | 0-4 | mnist | 10000 |  |
| `voltage_S_full_N400_flen20k` | `experiments/runs/voltage_S_full_N400_seed{S}_flen20k` | 0-4 | mnist | 20000 |  |
| `voltage_S_full_N400_flen5k` | `experiments/runs/voltage_S_full_N400_seed{S}_flen5k` | 0-4 | mnist | 5000 |  |

### R10. Section 4.5 / Appendix G: tuned voltage-STDP, 2k / 5k / 60k

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_bloodmnist_vhz2000` | `experiments/runs/voltage_S_full_N400_seed{S}_bloodmnist_vhz2000` | 0-4 | bloodmnist | 2000 |  |
| `voltage_S_full_N400_bloodmnist_vhz5000` | `experiments/runs/voltage_S_full_N400_seed{S}_bloodmnist_vhz5000` | 0-4 | bloodmnist | 5000 |  |
| `voltage_S_full_N400_emnist_letters_emnist_letters_vhz60000` | `experiments/runs/voltage_S_full_N400_seed{S}_emnist_letters_emnist_letters_vhz60000` | 0-4 | emnist_letters | 60000 |  |
| `voltage_S_full_N400_emnist_letters_vhz2000` | `experiments/runs/voltage_S_full_N400_seed{S}_emnist_letters_vhz2000` | 0-4 | emnist_letters | 2000 |  |
| `voltage_S_full_N400_emnist_letters_vhz5000` | `experiments/runs/voltage_S_full_N400_seed{S}_emnist_letters_vhz5000` | 0-4 | emnist_letters | 5000 |  |
| `voltage_S_full_N400_fmnist_vhz2000` | `experiments/runs/voltage_S_full_N400_seed{S}_fmnist_vhz2000` | 0-4 | fmnist | 2000 |  |
| `voltage_S_full_N400_fmnist_vhz5000` | `experiments/runs/voltage_S_full_N400_seed{S}_fmnist_vhz5000` | 0-4 | fmnist | 5000 |  |
| `voltage_S_full_N400_mnist_vhz60000` | `experiments/runs/voltage_S_full_N400_seed{S}_mnist_vhz60000` | 0-4 | mnist | 60000 |  |
| `voltage_S_full_N400_nmnist_vhz2000` | `experiments/runs/voltage_S_full_N400_seed{S}_nmnist_vhz2000` | 0-4 | nmnist | 2000 |  |
| `voltage_S_full_N400_nmnist_vhz5000` | `experiments/runs/voltage_S_full_N400_seed{S}_nmnist_vhz5000` | 0-4 | nmnist | 5000 |  |
| `voltage_S_full_N400_organamnist_vhz2000` | `experiments/runs/voltage_S_full_N400_seed{S}_organamnist_vhz2000` | 0-4 | organamnist | 2000 |  |
| `voltage_S_full_N400_organamnist_vhz5000` | `experiments/runs/voltage_S_full_N400_seed{S}_organamnist_vhz5000` | 0-4 | organamnist | 5000 |  |
| `voltage_S_full_N400_vhz2000` | `experiments/runs/voltage_S_full_N400_seed{S}_vhz2000` | 0-4 | mnist | 2000 |  |
| `voltage_S_full_N400_vhz5000` | `experiments/runs/voltage_S_full_N400_seed{S}_vhz5000` | 0-4 | mnist | 5000 |  |

### R11. Appendix G: tuned voltage-STDP at 60k, other datasets

| file | run directories | seeds | dataset | training images | note |
|---|---|---|---|--:|---|
| `voltage_S_full_N400_bloodmnist__blood_smoke_tuned` | `experiments/runs/voltage_S_full_N400_seed{S}_bloodmnist__blood_smoke_tuned` | 0 | bloodmnist | 60000 | seed 0 of `voltage_S_full_N400_seed{S}_bloodmnist__n5_blood_tuned` |
| `voltage_S_full_N400_bloodmnist__n5_blood_tuned` | `experiments/runs/voltage_S_full_N400_seed{S}_bloodmnist__n5_blood_tuned` | 1-4 | bloodmnist | 60000 | seed 0 = `voltage_S_full_N400_seed0_bloodmnist__blood_smoke_tuned` |
| `voltage_S_full_N400_fmnist__f3_fmnist_voltage_tuned` | `experiments/runs/voltage_S_full_N400_seed{S}_fmnist__f3_fmnist_voltage_tuned` | 0-4 | fmnist | 60000 |  |
| `voltage_S_full_N400_nmnist__nmnist_n5` | `experiments/runs/voltage_S_full_N400_seed{S}_nmnist__nmnist_n5` | 0-4 | nmnist | 60000 |  |
| `voltage_S_full_N400_organamnist__n5_organa_tuned` | `experiments/runs/voltage_S_full_N400_seed{S}_organamnist__n5_organa_tuned` | 1-4 | organamnist | 60000 | seed 0 = `voltage_S_full_N400_seed0_organamnist__organa_smoke_tuned` |
| `voltage_S_full_N400_organamnist__organa_smoke_tuned` | `experiments/runs/voltage_S_full_N400_seed{S}_organamnist__organa_smoke_tuned` | 0 | organamnist | 60000 | seed 0 of `voltage_S_full_N400_seed{S}_organamnist__n5_organa_tuned` |
## C. Clopath-neuron port (scripts/train/clopath_port_mnist.py)

Paper: Table 3 row 7, Sec 4.3 Clopath paragraphs, Tables A.9 and A.10, App J. Scorers
`scripts/score/clopath_port_probefix_score.py` (single probe on `*_pf.npz`) and
`scripts/score/appendix_j_counts.py` (per-network counts, using the functions of
`scripts/score/clopath_port_thetatar_score.py`). Each run directory holds `meta.json` (seed, image counts,
grid file and its SHA-256, all settings in `specs`); the seed-0 copies are in `run_configs/clopath_port/`.

| family | run directories | seeds | grid | command |
|---|---|---|---|---|
| grid10k (pre-registered grid, 9 cells x {10x, 1x} trained + 9 STDP-off) | `experiments/runs/clopath_port/grid10k/seed{S}` | 0-4 | `results/clopath_port_calibration/grid.json` | `bash scripts/train/run_seeds.sh port_grid10k`, or `SEEDS="0 1 2 3 4" bash scripts/train/clopath_port_grid.sh` |
| explore_k0 (no inhibition, App J) | `experiments/runs/clopath_port/explore_k0/seed{S}` | 0-4 | `run_configs/clopath_port/grid_explore_k0.json` (SHA-256 7a24bfd5..., as in meta.json) | `bash scripts/train/run_seeds.sh port_explore_k0` (`--n_test 5000`) |
| thetatar10k (homeostatic-target sweep, trained only) | `experiments/runs/clopath_port/thetatar10k/seed{0_all,1_a,1_b,2_all,3_a,3_b,4_a,4_b}` | 0-4 | `grid.json` | `bash scripts/train/clopath_port_thetatar_sweep.sh` (seeds 0-2); seeds 3-4 used the same command per half: `--theta_tar 240,960` (`_a`) and `3840,15360` (`_b`) |
| thetatar_repro (check that the sweep code reproduces grid10k at theta_tar 60) | `experiments/runs/clopath_port/thetatar_repro/seed0` | 0 | `grid.json` | `clopath_port_mnist.py train --seed 0 --grid results/clopath_port_calibration/grid.json --theta_tar 60 --no_random --prereg results/voltage_clopath_port_thetatar_PREREG.md --n_train 10000 --n_test 10000 --out experiments/runs/clopath_port/thetatar_repro/seed0`; the sweep launcher requires it and `results/clopath_port_thetatar/check_default_reproduction.json` |

Seeds 0-2 of grid10k and explore_k0 were trained before `clopath_port_mnist.py` recorded `theta_tar`; its
default (60) is the value they used. The grid10k seeds 3-4 record a later SHA-256 of the pre-registration file
(amended after seeds 0-2). `thetatar10k/seed0_all` (72 networks) was split into `seed0_all_a` and
`seed0_all_b` (36 each, `split_from` in their meta.json) only to re-extract it in two GPU processes;
`results/run_lists/jobs_port.txt` lists the split halves.

## K. Convolutional networks (conv/)

Configurations: `conv/configs/conv_voltage_config.yaml` (the paper's conv voltage-STDP network) and
`conv/configs/conv_seed5/*.yaml` (seed 5 and the random-filter readouts); `run_configs/conv/randomfilters_seed5.yaml`
is the one random-filter config missing there. `run_seeds.sh` sets `WANDB_MODE=disabled`; the base config
has `use_wandb: true`, which is the only difference from `conv_seed5/paper_seed5.yaml` besides seed, name and device.

| family | output | seeds | paper | command |
|---|---|---|---|---|
| conv_stdp: STDP + surrogate readout | `checkpoints/conv2l_voltage_clopath_faithful_mnist_paper_seed{S}` | 1-5 | trained filters for the three rows below | `SEEDS="1 2 3 4 5" bash scripts/train/run_seeds.sh conv_stdp` (= `conv/conv2layer_voltage_stdp.py --config conv/configs/conv_voltage_config.yaml --seed S --name-suffix _paper_seedS`) |
| conv_trainedreadout: fresh surrogate readout on seed-S filters | `checkpoints/..._TRAINEDREADOUT_seed{S}` | 1-4 (seed 5 is conv_stdp's own readout) | Sec 4.3 / App I surrogate readout 96.4 +- 0.4 (trained) | `run_seeds.sh conv_trainedreadout` (copies `..._paper_seedS/ckpt_L2_ep1.pt` to the new directory as `conv_voltage_stdp_ckpt.pt`, then runs the readout) |
| conv_randomfilters: surrogate readout on random filters (`epochs_stdp: [0, 0]`) | `checkpoints/..._RANDOMFILTERS_seed{S}` | 1-5 | Sec 4.3 / App I 97.1 +- 0.2 (random) | `run_seeds.sh conv_randomfilters` |
| conv_svm: linear readout, trained and random filters, 12k/10k images | `results/conv_svm_{trained,random}_s{S}.json` | 1-5 | Table 3 architecture row (94.3 +- 0.2 vs 95.2 +- 0.3) | `run_seeds.sh conv_svm` (needs conv_stdp seed S) |
| pair conv trained vs random | `results/conv_feats_pair*_s{S}`, `results/conv_pair_n5.json` | 0-4 | Sec 4.3 '+0.4 +- 0.4' | `bash scripts/train/conv_pair_trained_vs_random.sh`; needs the pair-STDP weights `conv2l_stdp_ckpt_pairgrade_seed{0..4}.pt` at the repo root (see gaps) |

## Not covered (no launcher survives)

- Pair-STDP conv weights `conv2l_stdp_ckpt_pairgrade_seed{0..4}.pt`: trained with `conv/conv2layer_stdp_surrogate.py`;
  the seed/command used is not recorded.
- Gate probes p0-p6 and the plateau gate probes: their configs ship beside the results
  (`results/gate_probes_2026-10/*.yaml`); run `scripts/score/voltage_gate_saturation_probe.py` /
  `voltage_plateau_gate_probe.py` with `--config-file` on them.
