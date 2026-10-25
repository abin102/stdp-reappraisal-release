#!/usr/bin/env bash
# Clopath port on MNIST: the pre-registered grid (results/voltage_clopath_neuron_port_PREREG.md).
# One process per seed on GPU 1; each runs 9 cells x {trained 10x, trained 1x} + 9 STDP-off
# networks. Idempotent: a seed with completed.flag is skipped; an interrupted seed resumes
# from progress.pt (every 1,000 images).
# Usage: N_TRAIN=10000 TAG=grid10k GRID=results/clopath_port_calibration/grid.json SEEDS="0 1 2" \
#        bash scripts/train/clopath_port_grid.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES="${GPU:-0}"   # physical GPU (the runs used GPU 1); cuda:0 inside
N_TRAIN=${N_TRAIN:-10000}; N_TEST=${N_TEST:-10000}; TAG=${TAG:-grid10k}
GRID=${GRID:-results/clopath_port_calibration/grid.json}; SEEDS=${SEEDS:-0 1 2}
LOG=logs_clopath_port; mkdir -p "$LOG"
echo "[$(date '+%F %T')] $TAG start: seeds $SEEDS, n_train $N_TRAIN, grid $GRID ($(sha256sum "$GRID" | cut -c1-16)), prereg $(sha256sum results/voltage_clopath_neuron_port_PREREG.md | cut -c1-16)" | tee -a "$LOG/master.log"
PIDS=()
for s in $SEEDS; do
  out="experiments/runs/clopath_port/$TAG/seed$s"
  if [ -f "$out/completed.flag" ]; then echo "[$(date '+%F %T')] SKIP $TAG s$s" | tee -a "$LOG/master.log"; continue; fi
  echo "[$(date '+%F %T')] LAUNCH $TAG s$s" | tee -a "$LOG/master.log"
  ( "$PY" -u scripts/train/clopath_port_mnist.py train --seed "$s" --device cuda:0 --grid "$GRID" \
        --n_train "$N_TRAIN" --n_test "$N_TEST" --out "$out" > "$LOG/${TAG}_seed$s.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] END $TAG s$s rc=$rc $( [ -f "$out/completed.flag" ] && echo completed || echo NOT-COMPLETED )" >> "$LOG/master.log"
  ) </dev/null >/dev/null 2>&1 &
  PIDS+=($!)
  sleep 20
done
wait "${PIDS[@]}" 2>/dev/null
echo "[$(date '+%F %T')] $TAG all done" | tee -a "$LOG/master.log"
