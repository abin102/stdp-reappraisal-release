#!/usr/bin/env bash
# PROBE-FIX MATRIX (2026-10-19). Re-runs the cross-dataset and medical Delta-over-random
# matrices so that BOTH readouts follow the standard protocol:
#   1. STDP on n_train = 10,000 training images (unchanged from the original matrices);
#   2. a frozen pass over those same 10,000 training images -> train_spike_counts.npz
#      (new run_single.py option evaluation.save_train_features=true);
#   3. a pass over the FULL test set -> test_spike_counts.npz (no 10,000-image cap).
# The linear probe and the post-hoc labels are then both fitted on the training-image
# spike counts and scored on the full test set (scripts/score/probefix_aggregate.py).
#
# Why: the original matrices (core_matrix_n10.sh, medical_matrix_n10.sh, voltage_b3_n10.sh)
# deleted each run's spike counts and checkpoint after scoring a 70/30 split of the
# (first <=10,000) test images, so the train-fit probe could not be applied to them.
# Their checkpoints cannot be reused either: run_single.py saves the checkpoint AFTER the
# test pass, so a reloaded network's adaptive threshold has already seen the test set.
#
# Grid: 3 rules x 8 datasets x seeds 0-4 x {trained, frozen(STDP off)} = 240 runs.
#   pair_exp, triplet : default full-scaffold configuration (as core/medical matrices)
#   voltage           : the tuned "B3" configuration (as voltage_b3 / medical matrices)
# Run dirs: experiments/runs/{rule}_S_full_N400_seed{s}_{dataset}_probefix_{mode} (no _{dataset} for MNIST)
# Nothing is deleted. Idempotent: a run with completed.flag is skipped.
#
# Usage: nohup bash scripts/train/probefix_matrix_n5.sh "0 1" 4 > results/logs_probefix/driver.out 2>&1 &
#   arg1 = GPUs to use (default "0"; the paper's runs used "0 1"), arg2 = parallel runs per GPU (default 4).
#   Writes results/logs_probefix/ALL_DONE when every run has finished.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"   # or set PY=/path/to/env/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOGS=results/logs_probefix; mkdir -p "$LOGS"; rm -f "$LOGS/ALL_DONE"
GPUS=(${1:-0}); SLOTS=${2:-4}
B3="--override stdp.A_plus=0.001 --override stdp.A_minus=0.003 --override stdp.tau_pre=16.8 --override scaffold.tc_theta_decay=1.0e4 --override scaffold.inh_weight=5.0 --override neuron.b_adapt=0.1 --override neuron.I_sp=0.2 --override weight_norm.norm=313.6"
COMMON="--override training.n_train=10000 --override training.n_test=1000000 --override evaluation.save_train_features=true"
RULES="pair_exp triplet voltage"
# Largest test sets first so the long TissueMNIST/EMNIST runs do not finish last.
DSETS="tissuemnist emnist_letters organamnist mnist nmnist fmnist pathmnist bloodmnist"

rd_of() { local t="_${2}"; [ "$2" = mnist ] && t=""   # run_single.py omits the tag for MNIST
          echo "experiments/runs/${1}_S_full_N400_seed${3}${t}_probefix_${4}"; }

launch() { # rule ds seed mode gpu
  local rule=$1 ds=$2 seed=$3 mode=$4 gpu=$5 rd cfg="" fz=""
  rd=$(rd_of "$rule" "$ds" "$seed" "$mode")
  local tag="${rule}.${ds}.s${seed}.${mode}"
  if [ -f "$rd/completed.flag" ]; then echo "[$(date '+%m-%d %H:%M')] SKIP $tag" >> "$LOGS/master.log"; echo SKIP; return; fi
  [ -d "$rd" ] && rm -rf "$rd"           # an unfinished earlier attempt of this exact run
  [ "$rule" = voltage ] && cfg="$B3"
  [ "$mode" = frozen ] && fz="--override training.frozen_init=true"
  echo "[$(date '+%m-%d %H:%M')] LAUNCH $tag on cuda:$gpu" >> "$LOGS/master.log"
  ( "$PY" -u experiments/scripts/run_single.py --rule "$rule" --scaffold full --seed "$seed" \
      --n_neurons 400 --device "cuda:$gpu" --dataset "$ds" $COMMON $fz $cfg \
      --name-suffix "probefix_${mode}" > "$LOGS/${tag}.log" 2>&1
    rc=$?; ok=no
    [ -f "$rd/completed.flag" ] && [ -f "$rd/train_spike_counts.npz" ] && [ -f "$rd/test_spike_counts.npz" ] && ok=yes
    echo "[$(date '+%m-%d %H:%M')] END $tag rc=$rc complete=$ok" >> "$LOGS/ends.log"
  ) </dev/null >/dev/null 2>&1 &
  echo $!
}

declare -A PIDS=()   # pid -> gpu
running_on() { local g=$1 n=0; for p in "${!PIDS[@]}"; do
  if kill -0 "$p" 2>/dev/null; then [ "${PIDS[$p]}" = "$g" ] && n=$((n+1)); else unset "PIDS[$p]"; fi; done; echo $n; }

JOBS=()
for ds in $DSETS; do for rule in $RULES; do for seed in 0 1 2 3 4; do for mode in trained frozen; do
  JOBS+=("$rule $ds $seed $mode"); done; done; done; done
echo "[$(date '+%m-%d %H:%M')] PROBEFIX START: ${#JOBS[@]} runs, GPUs ${GPUS[*]} x $SLOTS" >> "$LOGS/master.log"

for job in "${JOBS[@]}"; do
  read -r rule ds seed mode <<<"$job"
  while true; do
    placed=no
    for g in "${GPUS[@]}"; do
      if [ "$(running_on "$g")" -lt "$SLOTS" ]; then
        pid=$(launch "$rule" "$ds" "$seed" "$mode" "$g"); [ "$pid" != SKIP ] && PIDS[$pid]=$g
        placed=yes; break
      fi
    done
    [ "$placed" = yes ] && break
    sleep 30
  done
  sleep 3
done
while [ "$(running_on "${GPUS[0]}")" -gt 0 ] || { [ "${#GPUS[@]}" -gt 1 ] && [ "$(running_on "${GPUS[1]}")" -gt 0 ]; }; do sleep 60; done
n_ok=$(grep -c "complete=yes" "$LOGS/ends.log" 2>/dev/null || echo 0)
echo "[$(date '+%m-%d %H:%M')] PROBEFIX FINISHED: $n_ok runs complete (plus any SKIPs)" >> "$LOGS/master.log"
echo "finished $(date) ; complete=$n_ok" > "$LOGS/ALL_DONE"
