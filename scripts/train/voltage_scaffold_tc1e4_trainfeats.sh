#!/usr/bin/env bash
# Extract TRAIN features for the 20 corrected (tc_theta_decay=1e4) voltage
# scaffold cells, so Table A.3 / Figure 2 / §4.5 can be rebuilt on the paper's
# own train-fit protocol instead of a test-internal split.
#
# Why: scripts/score/scaffold_readout_table.py:36 maps each cell to the BARE run dir
# (voltage_S_*_N400_seed*), which are the superseded tc=1e6 runs. The corrected
# tc=1e4 re-runs (2026-04-28, see experiments/configs/rule_voltage.yaml:61-65)
# exist on disk but have no train_spike_counts.npz, so they cannot be scored
# with the train-fit probe the paper uses.
#
# No retraining: extract_train_features.py forward-passes 10k train images
# through each run's frozen checkpoint with plasticity off. Idempotent.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"   # or set PY=/path/to/env/bin/python
# Activate your environment first, or set VENV_ACTIVATE=/path/to/bin/activate
if [ -n "${VENV_ACTIVATE:-}" ]; then . "$VENV_ACTIVATE"; fi

DIRS=()
for sc in full inh_only thr_only none; do
  for s in 0 1 2 3 4; do
    d=experiments/runs/voltage_S_${sc}_N400_seed${s}__b_fix_tc1e4
    [ -d "$d" ] || d=experiments/runs/voltage_S_${sc}_N400_seed${s}__b_fix_tc1e4_60k
    [ -d "$d" ] && DIRS+=("$d")
  done
done
echo "extracting train features for ${#DIRS[@]} cells"
PYTHONPATH=. "$PY" -u scripts/extract/extract_train_features.py "${DIRS[@]}" \
  --n_subset 10000 --device "${DEVICE:-cuda:0}"
echo "######## TRAIN FEATURE EXTRACTION DONE ########"
