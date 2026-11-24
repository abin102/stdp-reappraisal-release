# How to evaluate unsupervised STDP

Code, configurations and run-selection maps for the paper *How to evaluate unsupervised
STDP: separating the rule from the readout, the scaffold and the input* (Abin S and
Raji Susan Mathew).

The paper proposes an evaluation protocol for unsupervised plasticity rules. Each trained
network is compared with the same network with plasticity switched off (the untrained
network); both are read by neuron labelling, the readout of Diehl and Cook, and by a
linear probe; and both are compared with a linear probe on the input spikes. The protocol
is applied to three STDP rules, pair-STDP (Diehl and Cook), triplet-STDP (Pfister and
Gerstner) and voltage-STDP (Clopath et al.), each training a competitive, fully connected
spiking layer of 400 neurons, on eight datasets: MNIST, N-MNIST, EMNIST-Letters,
Fashion-MNIST and four MedMNIST sets. It is also applied to an ablation of the competitive
scaffold, to a two-layer convolutional network and to SoftHebb (Journé et al., 2023), a
deep Hebbian method, on CIFAR-10 and four medical datasets (Table 5). Controls for
voltage-STDP include an exact reimplementation of Clopath's neuron and update rule.

**Labels.** This README and `RUN_MAP.md` use the paper's numbering: Sections 2.x
(Methods), 3.x (Results) and 4.x (Discussion), Tables 1 to 5 and Figures 1 to 6. Labels
with an S (for example Section S5 or Table S5) are in the paper's Supplementary material.
Script names and comments, the results sheets in `results/`, the section names quoted from
them and `run_configs/INDEX.md` keep the labels of earlier drafts: there, Appendices A to J
are Sections S1 to S10, Tables A.1 to A.10 are Tables S1 to S10, Figures A.1 and A.2 are
Figures S1 and S2, and main-text section, table and figure numbers refer to earlier drafts.
In the code, neuron labelling is called the post-hoc readout, and the untrained network the
random, frozen or STDP-disabled network.

## What is in this repository

| folder | contents |
|---|---|
| `models/` | the fully connected STDP layer (`fc_stdp_layer.py`, all three rules) and the reimplementation of Clopath's neuron (`clopath_port.py`) |
| `experiments/` | `scripts/run_single.py` (trains and evaluates one network), `config.py`, `metrics.py`, `run_matrix.py`, and the base configurations in `configs/` |
| `conv/` | the convolutional voltage-STDP network (`conv2layer_voltage_stdp.py`) and its configurations |
| `data/` | dataset loaders; the datasets themselves are downloaded here at run time |
| `run_configs/` | the saved configuration of every network family behind the paper, and `INDEX.md`, which lists each family with its run directories and seeds |
| `scripts/train/` | launchers that train the networks |
| `scripts/extract/` | scripts that record spike counts from trained networks |
| `scripts/score/` | one scorer per results table; they fit the linear probe and write `results/` |
| `scripts/measure/` | the protocol's readout analyses (Section 3.3) and Figure 3; they re-analyse the runs scored above |
| `scripts/figures/` | the figure scripts |
| `scripts/checks/` | checks: regenerate and compare every table, plus the checks of the Clopath implementations |
| `results/` | the result tables the paper cites, the run lists the scorers read, and the pre-registrations; `results/softhebb/` holds the SoftHebb runs of Table 5 and their notes |
| `RUN_MAP.md` | for each group of reported numbers: the networks behind it, their configuration, the scorer and the results file |

## What is not included

- **Trained weights and run outputs.** The run directories (`experiments/runs/`),
  checkpoints, spike counts and logs are not shipped. They are regenerated with the
  launchers below.
- **Datasets.** They are downloaded on first use into `data/`:
  MNIST, Fashion-MNIST and EMNIST through `torchvision`; OrganAMNIST, BloodMNIST,
  TissueMNIST and PathMNIST through `medmnist`; N-MNIST through `tonic`. If a
  download fails, place the files where the library expects them under `data/`
  (for N-MNIST: `data/NMNIST/`, from the N-MNIST page on Mendeley Data).
- **The paper source.**
- **Clopath's MATLAB code.** The reimplementation was checked against the code
  released with Clopath et al. (2010), ModelDB accession 144566
  (https://modeldb.science/144566). It is not redistributed. Download it and set
  `CLOPATH_MODELDB_DIR` to the folder that holds `aEIF.m`, `RFdevelop.m` and
  `VoTri.m`. Only two scripts need it:
  `scripts/figures/make_section44_mechanism_figure.py` (Figure S4) and the Octave
  comparisons of `scripts/checks/clopath_port_equivalence.py`. Both stop with
  these instructions when the variable is not set.
- **SoftHebb's code.** The SoftHebb runs (Table 5) import the demo released by
  Journé et al. at https://github.com/NeuromorphicComputing/SoftHebb, commit
  `f4252307633ba38a26ab52504f5fd40352f5a6fd`. That repository has no licence, so it
  is not redistributed. Clone it into `third_party/SoftHebb` (or set `SOFTHEBB_DIR`);
  `results/softhebb/README.md` gives the commands. It is needed only to rerun the
  SoftHebb networks, not to rebuild Table 5 from the shipped runs.

## Setup

Python 3.10 (the reported runs used 3.10.12), with a CUDA GPU for training.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` gives the minimum versions and, in comments, the exact versions
used. The `medmnist` and `tonic` packages download their datasets on first use.
GNU Octave is needed only for `scripts/checks/clopath_port_equivalence.py`.

Run every command from the repository root with `PYTHONPATH=.`. Several scripts
and the base configurations default to `cuda:1`; pass `--device cuda:0` (or set
`experiment.device` in a configuration) on a machine with one GPU. The launchers
in `scripts/train/` and `scripts/extract/` take a GPU list (`GPUS="0 1"`) and set
the device themselves.

## Pipeline

The work runs in five steps. `RUN_MAP.md` and `run_configs/INDEX.md` say which
networks each step needs for each table.

### 1. Train

```bash
# one seed of one network family (families: run_configs/INDEX.md)
PYTHONPATH=. python scripts/train/run_config.py run_configs/pair_exp_S_full_N400_probefix_trained.yaml --seed 0 --device cuda:0

# several families and seeds, one worker per GPU
SEEDS="0 1 2 3 4" GPUS="0" bash scripts/train/run_seeds.sh pair_exp_S_full_N400 pair_exp_S_full_N400_frozen60k

# the 240 networks of Table 3 and Figure 3 (3 rules x 8 datasets x 5 seeds x trained / plasticity off)
bash scripts/train/probefix_matrix_n5.sh "0" 4

# the 50 voltage-STDP threshold and spike-plateau networks (Table 4, Table S7)
bash scripts/train/probefix_voltage_thresholds_n5.sh "0" 4

# the convolutional voltage-STDP network, seeds 1-5 (Table 4, Section 3.6)
SEEDS="1 2 3 4 5" bash scripts/train/run_seeds.sh conv_stdp
SEEDS="1 2 3 4" bash scripts/train/run_seeds.sh conv_trainedreadout
SEEDS="1 2 3 4 5" bash scripts/train/run_seeds.sh conv_randomfilters conv_svm

# the reimplementation of Clopath's neuron (Table 4, Tables S9 and S10)
SEEDS="0 1 2 3 4" bash scripts/train/run_seeds.sh port_grid10k port_explore_k0
bash scripts/train/clopath_port_thetatar_sweep.sh   # seeds 0-2; needs the check run of RUN_MAP.md, section 14

# SoftHebb with the matched random baseline, 114 runs (Table 5); needs SoftHebb's code, see results/softhebb/README.md
DEVICE=cuda:0 bash scripts/train/softhebb_runs.sh
```

A finished run has a `completed.flag` and is skipped, so a launcher can be
restarted. `RUN_MAP.md` gives the remaining commands (the homeostatic-reference
sweep for seeds 3 and 4, the calibration of the Clopath grid, and the gate
probes).

### 2. Extract spike counts

Most networks save their training- and test-image spike counts while they run.
The others are scored from counts recorded again from the trained network with the
weights frozen (`*_pf.npz`), or need their training-image counts recorded. The lists
in `results/run_lists/` and `RUN_MAP.md` name them.

```bash
GPUS="0" bash scripts/extract/reextract_all.sh \
    -f results/run_lists/jobs_runsingle.txt -f results/run_lists/jobs_homeo.txt \
    -f results/run_lists/jobs_seeds34.txt -f results/run_lists/jobs_port.txt

# training-image counts of the 60,000-image scaffold and matched-random networks
PYTHONPATH=. python scripts/extract/extract_train_features.py \
    experiments/runs/pair_exp_S_full_N400_seed0 [more run directories] --n_subset 10000 --device cuda:0
bash scripts/train/voltage_scaffold_tc1e4_trainfeats.sh

```

### 3. Score

Each scorer writes one table to `results/`. Run them in this order (later ones read
earlier tables), with one BLAS thread:

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/score/posthoc_both_conventions.py
python scripts/score/scaffold_readout_table.py
python scripts/score/identical_neuron_check.py
python scripts/score/matched_random_60k_table.py
python scripts/score/probefix_aggregate.py
python scripts/score/probefix_voltage_thresholds_aggregate.py
python scripts/score/input_baselines_probefix.py
python scripts/score/input_baselines_probefix_table.py
python scripts/score/table2_probefix.py
python scripts/score/reextract_probefix_score.py
python scripts/score/appendix_g_probefix.py
python scripts/score/appendix_i_probefix.py
python scripts/score/section45_probefix_mnist.py
python scripts/score/section45_cosine_nmi.py
python scripts/score/clopath_port_probefix_score.py
python scripts/score/appendix_j_counts.py
python scripts/score/conv_arch_logreg_n5.py
python scripts/score/conv_surrogate_readout_n5.py --log_dir DIR   # readout logs, see RUN_MAP.md
python scripts/score/make_section44_table.py
python scripts/score/supervised_ceiling_5seed.py                  # GPU: trains the supervised CNNs
python scripts/score/softhebb_table5.py                           # Table 5, from results/softhebb/runs/
python scripts/score/softhebb_summary.py
python scripts/measure/gain_by_readout.py                         # Section 3.3: the protocol's readout analyses
python scripts/measure/readout_vs_rule_paired.py
python scripts/measure/posthoc_degeneracy_check.py
python scripts/measure/gain_by_readout_60k.py
python scripts/measure/labelling_vs_training_length.py
```

Some tables come from a GPU computation instead of saved runs: the gate probes
(`voltage_gate_saturation_probe.py`, `voltage_plateau_gate_probe.py`), the
convolutional linear readout (`scripts/extract/conv_svm_readout.py`, run by
`run_seeds.sh conv_svm`) and the supervised ceiling. `RUN_MAP.md` gives their
commands.

**The linear probe.** Every accuracy reported as "linear probe" is a multinomial
logistic regression with L2 regularisation and C = 1, on raw spike counts without
scaling. It is fitted to convergence (`max_iter` 100000) on the frozen spike counts
of 10,000 training images and scored on the full test set. Every fit runs with one
BLAS thread: fits on raw counts are badly conditioned, so where L-BFGS stops depends
on the order of floating-point sums, which changes with the number of threads
(`scripts/score/_blas.py`). All shipped tables were recomputed this way on
2026-10-26. The probes on the encoded input (`results/input_baselines_probefix.json`,
the encoded-input column of Table 3) are the most sensitive; they were recomputed one
fit at a time, in a single process, on 2026-11-24, which changed Table 3 by at most
0.4 points and no conclusion. The SoftHebb comparison (Table 5) does not use this probe: as in
SoftHebb's own code, its readout and its pixel baseline are linear layers trained with
dropout for 50 epochs (Section 3.7, Section S18).

### 4. Figures

```bash
export PYTHONPATH=. FIG_OUT_DIR=figures            # FIG_OUT_DIR defaults to figures/
python scripts/figures/make_paper_figs.py           # Figures 2, 4, 5, 6 and S3
python scripts/measure/fig_measurement.py          # Figure 3
python scripts/figures/clopath_curve_reproduction.py --plot-only   # Figure S1 (without --plot-only it reruns the simulation)
python scripts/figures/make_protocol_figure.py      # Figure 1
python scripts/figures/make_voltage_controls_figure.py             # Figure S2
CLOPATH_MODELDB_DIR=/path/to/modeldb python scripts/figures/make_section44_mechanism_figure.py  # Figure S4
```

The figures are drawn from the tables in `results/`, except Figure 5 (receptive
fields), which needs the final weights of three trained networks
(`RUN_MAP.md`), and the two schematics (Figures 1 and 2), which need no data.

### 5. Check

`scripts/checks/check_results.py` regenerates every table in a scratch folder and
compares it with the shipped file (see "Checking a regenerated result" below).
The other checks test the Clopath implementations:

```bash
PYTHONPATH=. python scripts/checks/clopath_positive_control.py   # Section S4
PYTHONPATH=. python scripts/checks/clopath_port_validate.py      # Section S10, validation
PYTHONPATH=. python scripts/checks/clopath_port_delay_check.py   # Section S10, read-out delay
CLOPATH_MODELDB_DIR=/path/to/modeldb PYTHONPATH=. python scripts/checks/clopath_port_equivalence.py  # Section S10, needs Octave
PYTHONPATH=. python scripts/checks/verify_readout_bound.py       # Remark 1 on the 60 scaffold networks
```

## Paper element to command and output

Commands are run from the repository root with `PYTHONPATH=.`. `RUN_MAP.md` lists
the networks each scorer reads.

| paper element | command(s) | output file(s) |
|---|---|---|
| Table 1, datasets | none (dataset properties; majority class = most frequent class of the official test split) | — |
| Table 3, input decomposition | `scripts/score/input_baselines_probefix.py`, `input_baselines_probefix_table.py`, `probefix_aggregate.py`, `table2_probefix.py` | `results/table2_probefix.{md,json}`, `results/input_baselines_probefix.{json,md}`, `results/probefix_matrix_n5.{json,md}` |
| Table 4, reference, voltage-thresholds and spike-plateau rows | `scripts/score/probefix_voltage_thresholds_aggregate.py` | `results/probefix_voltage_thresholds_n5.{md,json}` |
| Table 4, training-length and weight-management rows | `scripts/score/reextract_probefix_score.py`; per-seed pairing in `scripts/figures/make_voltage_controls_figure.py` | `results/reextract_probefix_scores.{md,json}`; `figures/fig_voltage_controls_data.json` |
| Table 4, architecture row | `scripts/train/run_seeds.sh conv_svm` (runs `scripts/extract/conv_svm_readout.py`, `scripts/score/classify_feats_torch.py`), then `scripts/score/conv_arch_logreg_n5.py` | `results/conv_svm_{trained,random}_s{1-5}.json`, `results/conv_arch_logreg_n5.{md,json}` |
| Table 4, Clopath's-neuron row | `scripts/score/clopath_port_probefix_score.py` | `results/clopath_port_probefix_scores.{md,json}` |
| Table S1, hyperparameters | none (configurations) | `run_configs/*.yaml`, `experiments/configs/*.yaml`, `conv/configs/conv_voltage_config.yaml` |
| Table S2, neuron labelling vs linear probe on eight datasets | `scripts/score/probefix_aggregate.py` | `results/probefix_matrix_n5.md` (its Table 3) |
| Table S3, scaffold matrix | `scripts/score/posthoc_both_conventions.py`, `scaffold_readout_table.py` | `results/posthoc_both_conventions.csv`, `results/scaffold_readout_table.csv`, `results/scaffold_readout_summary.md` |
| Table S4, voltage-STDP tuning | `scripts/score/reextract_probefix_score.py` | `results/reextract_probefix_scores.md` (section "Appendix C") |
| Table S5, weight management | `scripts/score/reextract_probefix_score.py` | `results/reextract_probefix_scores.md` (section "Table 3 weight-management row", named after an earlier draft) |
| Table S6, spike plateau | `scripts/score/reextract_probefix_score.py`; ratios in its note: `scripts/score/voltage_plateau_gate_probe.py` | `results/reextract_probefix_scores.md` (section "Appendix E: spike plateau"), `results/voltage_plateau_gate_probe/*.json`, `results/probefix_voltage_thresholds_n5.md` (the five-seed value in the note) |
| Table S7(a), threshold sweep | `scripts/score/probefix_voltage_thresholds_aggregate.py` | `results/probefix_voltage_thresholds_n5.md` |
| Table S7(b), plasticity-term activity | `scripts/score/voltage_gate_saturation_probe.py` on `results/gate_probes_2026-10/*.yaml`, then `scripts/score/make_section44_table.py` | `results/gate_probes_2026-10/*.json`, `results/section44_table.md` |
| Table S8, supervised ceiling | `scripts/score/supervised_ceiling_5seed.py` | `results/supervised_ceiling_5seed.{md,json}` |
| Table S9, Clopath's neuron | `scripts/score/clopath_port_probefix_score.py`, `appendix_j_counts.py` | `results/clopath_port_probefix_scores.md` (grid10k rows), `results/appendix_j_counts.md` |
| Table S10, larger homeostatic reference | `scripts/score/clopath_port_probefix_score.py`, `appendix_j_counts.py` | `results/clopath_port_probefix_scores.md` (thetatar10k rows), `results/appendix_j_counts.md` |
| Figure 2, pipeline | `scripts/figures/make_paper_figs.py` | `figures/fig1_pipeline.pdf` |
| Figure 4, neuron labelling vs linear probe on four scaffolds | `scripts/figures/make_paper_figs.py` | `figures/fig2_readout.pdf` |
| Figure 5, receptive fields | `scripts/figures/make_paper_figs.py` (needs `weights_final.npz` of three runs) | `figures/fig7_receptive_fields.pdf` |
| Figure S4, depression and potentiation thresholds | `scripts/figures/make_section44_mechanism_figure.py` (needs `CLOPATH_MODELDB_DIR`) | `figures/fig_section44_mechanism.pdf` |
| Figure S3, accuracy against training length | `scripts/figures/make_paper_figs.py` | `figures/fig4_forgetting.pdf` |
| Figure 6, scaffold ablation | `scripts/figures/make_paper_figs.py` | `figures/fig5_scaffold_fragility.pdf` |
| Figure 3, the protocol on one network; gain under each readout | `scripts/measure/fig_measurement.py` | `figures/fig_measurement.pdf` |
| Figure 1, evaluation protocol | `scripts/figures/make_protocol_figure.py` | `figures/fig8_protocol.pdf` |
| Figure S1, positive control | `scripts/figures/clopath_curve_reproduction.py` | `results/clopath_curve_reproduction.json`, `figures/fig6_clopath_curves.pdf` |
| Figure S2, voltage-STDP controls by seed | `scripts/figures/make_voltage_controls_figure.py` | `figures/fig_voltage_controls.pdf`, `figures/fig_voltage_controls_data.json` |
| Section 3.3, collapse counts and Remark 1 | `scripts/score/posthoc_both_conventions.py`, `scaffold_readout_table.py`, `identical_neuron_check.py`; `scripts/checks/verify_readout_bound.py` (prints the check) | `results/posthoc_both_conventions.csv`, `results/scaffold_readout_table.csv`, `results/scaffold_readout_summary.md`, `results/identical_neuron_check.csv` |
| Section 3.4, matched random baseline at 60,000 images | `scripts/score/matched_random_60k_table.py` | `results/matched_random_60k.json` |
| Section 3.7, convolutional network, surrogate-gradient readout (last epoch; best epoch in Section S16) | `scripts/score/conv_surrogate_readout_n5.py` | `results/conv_surrogate_readout_n5.{md,json}` |
| Section S16, share of spikes reset before θ₊ (tuned configuration) | `scripts/score/voltage_gate_saturation_probe.py --config-file run_configs/voltage_S_full_N400__mnist_adex_baseline.yaml` | `results/gate_tuned_seed{0-2}.json` |
| Section S15, potentiation : depression in the default configuration with τθ = 10⁶ ms | `scripts/score/voltage_gate_saturation_probe.py --config-file run_configs/voltage_S_full_N400.yaml` | `results/gate_scaffold_seed{0-4}.json` |
| Section 3.6 and Section S7, training length | `scripts/score/section45_probefix_mnist.py`, `section45_cosine_nmi.py`, `appendix_g_probefix.py` | `results/section45_probefix_mnist.{md,json}`, `results/section45_cosine_nmi.{md,json}`, `results/appendix_g_probefix.{md,json}` |
| Sections 3.4 and 3.5 | as Table 3, Table S3 and Figure 6 | as above |
| Section 3.3, the rule's gain under each readout on eight datasets | `scripts/measure/gain_by_readout.py` | `results/gain_by_readout.json` |
| Section 3.3, gain under each readout after 60,000 training images (30.0 against 16.2 points) | `scripts/measure/gain_by_readout_60k.py` | `results/gain_by_readout_60k.json` |
| Section 3.3, probe minus labelling at 5,000 to 60,000 training images | `scripts/measure/labelling_vs_training_length.py` | `results/labelling_vs_training_length.json` |
| Section 3.3, PathMNIST: one class for every labelled neuron | `scripts/measure/posthoc_degeneracy_check.py` | `results/posthoc_degeneracy.json` |
| Section 3.3, trained network under labelling below the untrained network under the probe | `scripts/measure/readout_vs_rule_paired.py` | `results/readout_vs_rule_paired.json` |
| Table 5, Section 3.7, Section 2.11 and Section S18, SoftHebb with the matched random baseline | `scripts/score/softhebb_table5.py`, `softhebb_summary.py` (from the shipped runs); the runs: `scripts/train/softhebb_runs.sh` (needs SoftHebb's code) | `results/softhebb/table5_values.json`, `results/softhebb/pilot_summary.{md,json}`, `results/softhebb/runs/*/result.json` |
| Section S4, positive control | `scripts/checks/clopath_positive_control.py`, `scripts/figures/clopath_curve_reproduction.py` | `results/clopath_positive_control.{json,md}`, `results/clopath_curve_reproduction.json` |
| Section S5, homeostasis | `scripts/score/reextract_probefix_score.py` | `results/reextract_probefix_scores.md` (section "Appendix E: homeostasis") |
| Section S9, Fisher ratio | `scripts/score/appendix_i_probefix.py` | `results/appendix_i_probefix.{md,json}` |
| Section S10, checks of the reimplementation | `scripts/checks/clopath_port_equivalence.py`, `clopath_port_validate.py`, `clopath_port_delay_check.py`; grid choice: `scripts/train/clopath_port_mnist.py calibrate`, `clopath_port_select_grid.py` | `results/clopath_port_validation/*.json`, `results/clopath_port_calibration/{calibration,grid}.json` |
| Section S10, counts | `scripts/score/appendix_j_counts.py` | `results/appendix_j_counts.{md,json}` |

`results/voltage_clopath_neuron_port_PREREG.md` and
`results/voltage_clopath_port_thetatar_PREREG.md` are the pre-registrations of the
two Clopath-neuron experiments. `results/clopath_positive_control.md` describes the
positive control.

## Checking a regenerated result

`scripts/checks/check_results.py` copies the code and the shipped tables into a
scratch folder, links the run outputs into it, reruns every scorer there in
dependency order, and compares each new table with the shipped one. Nothing in the
repository is written. Markdown and CSV files must match byte for byte; JSON
numbers must agree to 1e-9. It needs the run outputs (`experiments/runs/`,
`results/logs_conv5/`) and the datasets.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
    python scripts/checks/check_results.py --jobs 2 --workspace /path/to/scratch --resume
```

- `--list` prints every scorer, its command and its outputs.
- `--only NAME ...` checks some tables (with their dependencies).
- `--workspace DIR` keeps the scratch folder; with `--resume`, a restarted check
  skips the scorers that already finished there (each leaves `_logs/<name>.done`).
  The full check takes several CPU hours.
- `--figures REF_DIR` also redraws the figures and compares the PDFs with those in
  `REF_DIR`.

The script prints PASS, FAIL or SKIP per file and exits with status 0 when nothing
fails. With the library versions in `requirements.txt` most tables match exactly; a
probe fit can move by about one test image with another BLAS build, and the fits on the
encoded input by up to about one point per seed. Keep `--jobs` small: on the machine used
here, many simultaneous jobs gave irregular fits and occasional crashes.
The tables made by a GPU simulation (gate probes, convolutional readouts,
supervised ceiling, the Clopath calibration and equivalence checks) are not rerun;
the docstring of `check_results.py` lists them. The SoftHebb networks are not rerun
either; Table 5 is rebuilt from their shipped `result.json` files, which needs no run
data.

## Seeds

Every reported comparison uses five seeds, 0 to 4 (1 to 5 for the convolutional
voltage-STDP network; SoftHebb on CIFAR-10 uses three, 0 to 2). A seed fixes the initial weights, the order of the training
images and the Poisson draws. A trained network and its STDP-disabled control use
the same seed. Single-seed screens (Tables S4 to S6 and S7(b)) are marked as
such in the paper and in `run_configs/INDEX.md`.

## Licence

MIT, see `LICENSE`.

## How to cite

If you use this code, please cite the paper:

> Abin S and Raji Susan Mathew. Evaluating what unsupervised STDP learns: matched
> random baselines, linear readouts, and a controlled reappraisal of three
> plasticity rules.

Archived at Zenodo: https://doi.org/10.5281/zenodo.23683757 (this DOI always
points to the latest version).
