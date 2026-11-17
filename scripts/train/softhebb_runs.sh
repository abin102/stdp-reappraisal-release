#!/bin/bash
# SoftHebb with the matched random baseline (Section 3.9, Table 5, Methods 2.13, Section S18): the 114 runs, one after
# another on one GPU. Release copy of experiments/softhebb_pilot/run_pilot.sh; the paths changed, and the wait for other
# users' jobs on the GPU (a courtesy on the shared machine of the original runs) was removed.
# Needs SoftHebb's demo at commit f425230 in third_party/SoftHebb (or SOFTHEBB_DIR), see results/softhebb/README.md.
# A finished run is skipped; a failed run stops the queue.
#   PHASE=short: 10,000 training images (medical sets, seeds 0-4), then CIFAR-10 (whole split, seeds 0-2);
#   PHASE=full: the whole medical training splits (seeds 0-4; TissueMNIST not run); PHASE=all (default): both, then scoring.
# Every dataset, size and seed gets trained, random and pixels.
#   DEVICE=cuda:0 PHASE=all nohup bash scripts/train/softhebb_runs.sh > /dev/null 2>&1 &
# Runs go to OUT (default experiments/runs/softhebb); the summary of those runs is written to OUT as well.
# The shipped runs are in results/softhebb/runs/.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1   # repo root
PY="${PY:-python}"   # or set PY=/path/to/env/bin/python
export PYTHONPATH=.
DEVICE=${DEVICE:-cuda:0}; PHASE=${PHASE:-all}; OUT=${OUT:-experiments/runs/softhebb}
mkdir -p "$OUT"; LOG=$OUT/softhebb_runs.log
echo "### SoftHebb runs start $(date +%F_%T) on $DEVICE, phase $PHASE; SHA-256 of the scripts and third-party demo:" >> $LOG
sha256sum scripts/train/softhebb_run.py scripts/score/softhebb_summary.py scripts/train/softhebb_runs.sh \
    "${SOFTHEBB_DIR:-third_party/SoftHebb}/demo.py" >> $LOG
one() {  # dataset mode n_train seed
    [ -f "$OUT/$1__$2__n$3__seed$4/result.json" ] && return 0
    $PY -u scripts/train/softhebb_run.py --dataset $1 --mode $2 --n_train $3 --seed $4 --device $DEVICE --out "$OUT" >> $LOG 2>&1 \
        || { echo "### STOP: $1 $2 n$3 seed$4 failed $(date +%F_%T)" >> $LOG; exit 1; }
}
MODES="trained random pixels"
tenk()  { for seed in 0 1 2 3 4; do for ds in bloodmnist organamnist pathmnist tissuemnist; do for mode in $MODES; do one $ds $mode 10000 $seed; done; done; done; }
cifar() { for seed in 0 1 2; do for mode in $MODES; do one cifar10 $mode 0 $seed; done; done; }
whole() { for seed in 0 1 2 3 4; do for ds in bloodmnist organamnist pathmnist; do for mode in $MODES; do one $ds $mode 0 $seed; done; done; done; }
if [ "$PHASE" != full ]; then tenk; cifar; fi
if [ "$PHASE" != short ]; then whole; fi
$PY scripts/score/softhebb_summary.py --runs "$OUT" --out "$OUT" >> $LOG 2>&1
echo "### SoftHebb runs phase $PHASE done $(date +%F_%T)" >> $LOG
