#!/usr/bin/env bash
# Re-extract training- and test-image spike counts (train/test_spike_counts_pf.npz) for finished runs,
# one sequential worker per GPU. A directory with meta.json is a Clopath-port run
# (scripts/extract/clopath_port_reextract.py); any other is a run_single run
# (scripts/extract/reextract_features.py). Both skip a run whose _pf files already exist.
#
# Usage:  GPUS="0 1" bash scripts/extract/reextract_all.sh RUN_DIR [RUN_DIR ...]
#         GPUS="0 1" bash scripts/extract/reextract_all.sh -f results/run_lists/jobs_runsingle.txt [-f LIST ...]
#   LIST = one run directory per line, relative to the repo root (results/run_lists/*.txt are the
#   lists behind the paper: jobs_runsingle, jobs_homeo, jobs_seeds34, jobs_port).
# Env: PY (python, default "python"), GPUS (default "0"), LOG_DIR (default logs_reextract).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"; GPUS="${GPUS:-0}"; LOG_DIR="${LOG_DIR:-logs_reextract}"
export PYTHONPATH=.
DIRS=()
while [ $# -gt 0 ]; do
  if [ "$1" = -f ]; then mapfile -t -O "${#DIRS[@]}" DIRS < <(grep -v '^\s*\(#\|$\)' "$2"); shift 2
  else DIRS+=("$1"); shift; fi
done
[ ${#DIRS[@]} -gt 0 ] || { sed -n '2,11p' "$0"; exit 1; }
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/master.log"; }

GPU_LIST=($GPUS); NG=${#GPU_LIST[@]}
log "start: ${#DIRS[@]} run directories on GPUs $GPUS"
for i in "${!GPU_LIST[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]}
    for ((j = i; j < ${#DIRS[@]}; j += NG)); do
      d=${DIRS[$j]}; tag=$(echo "${d#experiments/runs/}" | tr / _)
      if [ -f "$d/meta.json" ]; then script=scripts/extract/clopath_port_reextract.py
      else script=scripts/extract/reextract_features.py; fi
      if "$PY" -u "$script" "$d" --device cuda:0 >> "$LOG_DIR/$tag.log" 2>&1 && [ -f "$d/test_spike_counts_pf.npz" ]
      then log "OK   $d"; else log "FAIL $d (see $LOG_DIR/$tag.log)"; fi
    done
  ) &
done
wait
log "done"
