# Evaluating what unsupervised STDP learns

Code, configurations and run-selection maps for the paper *Evaluating what
unsupervised STDP learns: matched random baselines, linear readouts, and a
controlled reappraisal of three plasticity rules* (Abin S and Raji Susan Mathew).

The paper compares three STDP rules: pair-STDP (Diehl and Cook), triplet-STDP
(Pfister and Gerstner) and voltage-STDP (Clopath et al.). Each rule trains a
competitive, fully connected spiking layer of 400 neurons. Every trained network
is compared with the same network with STDP disabled (the matched random
baseline), and is read out with a linear probe as well as the usual post-hoc
labelling. The comparison covers eight datasets: MNIST, N-MNIST, EMNIST-Letters,
Fashion-MNIST and four MedMNIST sets. Controls include a two-layer convolutional
network and an exact reimplementation of Clopath's neuron and update rule.

**Labels.** Sections, tables and figures numbered with an S (for example Section S5 or
Table S5) are in the paper's Supplementary material. Script names and comments, the
results sheets in `results/`, and the section names quoted from them keep the labels of
an earlier draft: Appendices A to J are Sections S1 to S10, Tables A.1 to A.10 are
Tables S1 to S10, and Figures A.1 and A.2 are Figures S1 and S2. A few script messages carry labels
from still earlier drafts.

## What is in this repository

| folder | contents |
|---|---|
| `models/` | the fully connected STDP layer (`fc_stdp_layer.py`, all three rules) and the reimplementation of Clopath's neuron (`clopath_port.py`) |
| `experiments/` | `scripts/run_single.py` (trains and evaluates one network), `config.py`, `metrics.py`, `run_matrix.py`, and the base configurations in `configs/` |
| `conv/` | the two convolutional networks (`conv2layer_voltage_stdp.py`, `conv2layer_stdp_surrogate.py`) and their configurations |
| `data/` | dataset loaders; the datasets themselves are downloaded here at run time |
| `run_configs/` | the saved configuration of every network family behind the paper, and `INDEX.md`, which lists each family with its run directories and seeds |
| `scripts/train/` | launchers that train the networks |
| `scripts/extract/` | scripts that record spike counts from trained networks |
| `scripts/score/` | one scorer per results table; they fit the linear probe and write `results/` |
| `scripts/figures/` | the figure scripts |
| `scripts/checks/` | checks: regenerate and compare every table, plus the checks of the Clopath implementations |
| `results/` | the result tables the paper cites, the run lists the scorers read, and the pre-registrations |
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
  `scripts/figures/make_section44_mechanism_figure.py` (Figure 4) and the Octave
  comparisons of `scripts/checks/clopath_port_equivalence.py`. Both stop with
  these instructions when the variable is not set.

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

# the 240 networks of Table 2 and Figure 7 (3 rules x 8 datasets x 5 seeds x trained / STDP disabled)
bash scripts/train/probefix_matrix_n5.sh "0" 4

# the 50 voltage-STDP threshold and spike-plateau networks (Table 3, Table S7)
bash scripts/train/probefix_voltage_thresholds_n5.sh "0" 4

# the convolutional voltage-STDP network, seeds 1-5 (Table 3, Section 4.3)
SEEDS="1 2 3 4 5" bash scripts/train/run_seeds.sh conv_stdp
SEEDS="1 2 3 4" bash scripts/train/run_seeds.sh conv_trainedreadout
SEEDS="1 2 3 4 5" bash scripts/train/run_seeds.sh conv_randomfilters conv_svm

# the reimplementation of Clopath's neuron (Table 3, Tables S9 and S10)
SEEDS="0 1 2 3 4" bash scripts/train/run_seeds.sh port_grid10k port_explore_k0
bash scripts/train/clopath_port_thetatar_sweep.sh   # seeds 0-2; needs the check run of RUN_MAP.md, section 14
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

# pooled features of the pair-STDP convolutional control
# (needs the pair-STDP convolutional weights, see RUN_MAP.md)
bash scripts/train/conv_pair_trained_vs_random.sh cuda:0
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
python scripts/score/conv_pair_n5_table.py
python scripts/score/make_section44_table.py
python scripts/score/supervised_ceiling_5seed.py                  # GPU: trains the supervised CNNs
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
2026-10-26.

### 4. Figures

```bash
export PYTHONPATH=. FIG_OUT_DIR=figures            # FIG_OUT_DIR defaults to figures/
python scripts/figures/make_paper_figs.py           # Figures 1, 2, 3, 5, 6, 7
python scripts/figures/clopath_curve_reproduction.py --plot-only   # Figure S1 (without --plot-only it reruns the simulation)
python scripts/figures/make_protocol_figure.py      # Figure 8
python scripts/figures/make_voltage_controls_figure.py             # Figure S2
CLOPATH_MODELDB_DIR=/path/to/modeldb python scripts/figures/make_section44_mechanism_figure.py  # Figure 4
```

The figures are drawn from the tables in `results/`, except Figure 3 (receptive
fields), which needs the final weights of three trained networks
(`RUN_MAP.md`), and the two schematics (Figures 1 and 8), which need no data.

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
| Table 2, input decomposition | `scripts/score/input_baselines_probefix.py`, `input_baselines_probefix_table.py`, `probefix_aggregate.py`, `table2_probefix.py` | `results/table2_probefix.{md,json}`, `results/input_baselines_probefix.{json,md}`, `results/probefix_matrix_n5.{json,md}` |
| Table 3, reference, voltage-thresholds and spike-plateau rows | `scripts/score/probefix_voltage_thresholds_aggregate.py` | `results/probefix_voltage_thresholds_n5.{md,json}` |
| Table 3, training-length and weight-management rows | `scripts/score/reextract_probefix_score.py`; per-seed pairing in `scripts/figures/make_voltage_controls_figure.py` | `results/reextract_probefix_scores.{md,json}`; `figures/fig_voltage_controls_data.json` |
| Table 3, architecture row | `scripts/train/run_seeds.sh conv_svm` (runs `scripts/extract/conv_svm_readout.py`, `scripts/score/classify_feats_torch.py`), then `scripts/score/conv_arch_logreg_n5.py` | `results/conv_svm_{trained,random}_s{1-5}.json`, `results/conv_arch_logreg_n5.{md,json}` |
| Table 3, Clopath's-neuron row | `scripts/score/clopath_port_probefix_score.py` | `results/clopath_port_probefix_scores.{md,json}` |
| Table S1, hyperparameters | none (configurations) | `run_configs/*.yaml`, `experiments/configs/*.yaml`, `conv/configs/conv_voltage_config.yaml` |
| Table S2, post-hoc vs linear on eight datasets | `scripts/score/probefix_aggregate.py` | `results/probefix_matrix_n5.md` (its Table 3) |
| Table S3, scaffold matrix | `scripts/score/posthoc_both_conventions.py`, `scaffold_readout_table.py` | `results/posthoc_both_conventions.csv`, `results/scaffold_readout_table.csv`, `results/scaffold_readout_summary.md` |
| Table S4, voltage-STDP tuning | `scripts/score/reextract_probefix_score.py` | `results/reextract_probefix_scores.md` (section "Appendix C") |
| Table S5, weight management | `scripts/score/reextract_probefix_score.py` | `results/reextract_probefix_scores.md` (section "Table 3 weight-management row") |
| Table S6, spike plateau | `scripts/score/reextract_probefix_score.py`; ratios in its note: `scripts/score/voltage_plateau_gate_probe.py` | `results/reextract_probefix_scores.md` (section "Appendix E: spike plateau"), `results/voltage_plateau_gate_probe/*.json`, `results/probefix_voltage_thresholds_n5.md` (the five-seed value in the note) |
| Table S7(a), threshold sweep | `scripts/score/probefix_voltage_thresholds_aggregate.py` | `results/probefix_voltage_thresholds_n5.md` |
| Table S7(b), plasticity-term activity | `scripts/score/voltage_gate_saturation_probe.py` on `results/gate_probes_2026-10/*.yaml`, then `scripts/score/make_section44_table.py` | `results/gate_probes_2026-10/*.json`, `results/section44_table.md` |
| Table S8, supervised ceiling | `scripts/score/supervised_ceiling_5seed.py` | `results/supervised_ceiling_5seed.{md,json}` |
| Table S9, Clopath's neuron | `scripts/score/clopath_port_probefix_score.py`, `appendix_j_counts.py` | `results/clopath_port_probefix_scores.md` (grid10k rows), `results/appendix_j_counts.md` |
| Table S10, larger homeostatic reference | `scripts/score/clopath_port_probefix_score.py`, `appendix_j_counts.py` | `results/clopath_port_probefix_scores.md` (thetatar10k rows), `results/appendix_j_counts.md` |
| Figure 1, pipeline | `scripts/figures/make_paper_figs.py` | `figures/fig1_pipeline.pdf` |
| Figure 2, post-hoc vs linear readout | `scripts/figures/make_paper_figs.py` | `figures/fig2_readout.pdf` |
| Figure 3, receptive fields | `scripts/figures/make_paper_figs.py` (needs `weights_final.npz` of three runs) | `figures/fig7_receptive_fields.pdf` |
| Figure 4, depression and potentiation thresholds | `scripts/figures/make_section44_mechanism_figure.py` (needs `CLOPATH_MODELDB_DIR`) | `figures/fig_section44_mechanism.pdf` |
| Figure 5, accuracy against training length | `scripts/figures/make_paper_figs.py` | `figures/fig4_forgetting.pdf` |
| Figure 6, scaffold decomposition | `scripts/figures/make_paper_figs.py` | `figures/fig5_scaffold_fragility.pdf` |
| Figure 7, Δ-over-random on eight datasets | `scripts/figures/make_paper_figs.py` | `figures/fig3_delta_over_random.pdf` |
| Figure 8, evaluation protocol | `scripts/figures/make_protocol_figure.py` | `figures/fig8_protocol.pdf` |
| Figure S1, positive control | `scripts/figures/clopath_curve_reproduction.py` | `results/clopath_curve_reproduction.json`, `figures/fig6_clopath_curves.pdf` |
| Figure S2, voltage-STDP controls by seed | `scripts/figures/make_voltage_controls_figure.py` | `figures/fig_voltage_controls.pdf`, `figures/fig_voltage_controls_data.json` |
| Section 4.1, collapse counts and Remark 1 | `scripts/score/posthoc_both_conventions.py`, `scaffold_readout_table.py`, `identical_neuron_check.py`; `scripts/checks/verify_readout_bound.py` (prints the check) | `results/posthoc_both_conventions.csv`, `results/scaffold_readout_table.csv`, `results/scaffold_readout_summary.md`, `results/identical_neuron_check.csv` |
| Section 4.2, matched random baseline at 60,000 images | `scripts/score/matched_random_60k_table.py` | `results/matched_random_60k.json` |
| Section 4.3, convolutional readouts | `scripts/score/conv_surrogate_readout_n5.py`, `conv_pair_n5_table.py` | `results/conv_surrogate_readout_n5.{md,json}`, `results/conv_pair_n5.json` |
| Section 4.3, share of spikes reset before θ₊ (tuned configuration) | `scripts/score/voltage_gate_saturation_probe.py --config-file run_configs/voltage_S_full_N400__mnist_adex_baseline.yaml` | `results/gate_tuned_seed{0-2}.json` |
| Section 4.4, potentiation : depression in the default configuration with τθ = 10⁶ ms | `scripts/score/voltage_gate_saturation_probe.py --config-file run_configs/voltage_S_full_N400.yaml` | `results/gate_scaffold_seed{0-4}.json` |
| Section 4.5 and Section S7, training length | `scripts/score/section45_probefix_mnist.py`, `section45_cosine_nmi.py`, `appendix_g_probefix.py` | `results/section45_probefix_mnist.{md,json}`, `results/section45_cosine_nmi.{md,json}`, `results/appendix_g_probefix.{md,json}` |
| Sections 4.6 and 4.7 | as Table S3, Figure 6, Table 2 and Figure 7 | as above |
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
`results/logs_conv5/`, `results/conv_feats_*.npz`) and the datasets.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
    python scripts/checks/check_results.py --jobs 8 --workspace /path/to/scratch --resume
```

- `--list` prints every scorer, its command and its outputs.
- `--only NAME ...` checks some tables (with their dependencies).
- `--workspace DIR` keeps the scratch folder; with `--resume`, a restarted check
  skips the scorers that already finished there (each leaves `_logs/<name>.done`).
  The full check takes several CPU hours.
- `--figures REF_DIR` also redraws the figures and compares the PDFs with those in
  `REF_DIR`.

The script prints PASS, FAIL or SKIP per file and exits with status 0 when nothing
fails. With the library versions in `requirements.txt` the tables should match
exactly. With another BLAS build a probe fit can move by about one test image.
The tables made by a GPU simulation (gate probes, convolutional readouts,
supervised ceiling, the Clopath calibration and equivalence checks) are not rerun;
the docstring of `check_results.py` lists them.

## Seeds

Every reported comparison uses five seeds, 0 to 4 (1 to 5 for the convolutional
voltage-STDP network). A seed fixes the initial weights, the order of the training
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
