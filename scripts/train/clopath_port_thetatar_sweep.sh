#!/usr/bin/env bash
# Homeostatic-target sweep for the exact Clopath port (results/voltage_clopath_port_thetatar_PREREG.md).
# Trained networks only (--no_random; the STDP-off networks are reused from grid10k), all 9 cells,
# amplitudes 10x and 1x, theta_tar 240, 960, 3840, 15360 mV^2 (60 = the first grid, not rerun).
# Jobs (PREREG Section 7):
#   seed0_all  seed 0, all four targets (72 networks)      GPU 1, first
#   seed1_a    seed 1, theta_tar 240, 960 (36 networks)    GPU 1, second
#   seed2_all  seed 2, all four targets (72 networks)      GPU 0, first  (guest gate)
#   seed1_b    seed 1, theta_tar 3840, 15360 (36 networks) GPU 0, second (guest gate)
# One process per GPU at a time (two processes on one GPU are slower than one after the other).
# A GPU-0 job starts only if nvidia-smi lists nothing on GPU 0 except gnome-remote-desktop-daemon;
# otherwise it is left for GPU 1, which runs every job not completed and not running at the end of
# its own queue. Guest processes are never touched. A job that fails is retried once (it resumes
# from progress.pt). Idempotent: completed jobs are skipped.
# Usage: nohup bash scripts/train/clopath_port_thetatar_sweep.sh > logs_clopath_port/thetatar_sweep.nohup 2>&1 &
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"   # or set PY=/path/to/env/bin/python
export PYTHONPATH=.
TAG=thetatar10k; N_TRAIN=${N_TRAIN:-10000}; N_TEST=${N_TEST:-10000}
GRID=results/clopath_port_calibration/grid.json
PREREG=results/voltage_clopath_port_thetatar_PREREG.md
PREREG_SHA_AT_LAUNCH=7dd1c61571c2dc21d87adad156ea518e26275f9b45d8681502b42e0845f83ded
# Machine-specific: the original machine shared GPU 0 with another user, so GPU-0 jobs started only
# when that GPU (UUID below) was free. Set GPU0_UUID for your machine.
GPU0_UUID=${GPU0_UUID:-GPU-82c115ae-7651-2b22-b9f9-b4a8c443e84d}
LOG=logs_clopath_port; RUNS=experiments/runs/clopath_port/$TAG; LOCKS=$RUNS/.locks
mkdir -p "$LOG" "$LOCKS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/master.log"; }

declare -A SEED=( [seed0_all]=0 [seed1_a]=1 [seed1_b]=1 [seed2_all]=2 )
declare -A TT=( [seed0_all]=240,960,3840,15360 [seed1_a]=240,960 [seed1_b]=3840,15360 [seed2_all]=240,960,3840,15360 )

prereg_ok() {   # every job stamps sha256(PREREG) into meta.json at its start: the file must not change
  [ "$(sha256sum "$PREREG" | cut -d' ' -f1)" = "$PREREG_SHA_AT_LAUNCH" ]
}

gpu0_others() {   # compute processes on GPU 0 other than the remote-desktop daemon; "ERR" if nvidia-smi fails
  local q
  q=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null) || { echo ERR; return; }
  echo "$q" | grep "$GPU0_UUID" | grep -v gnome-remote-desktop-daemon || true
}

run_job() {   # gpu name
  local gpu=$1 name=$2 out="$RUNS/$2" attempt rc
  if [ -f "$out/completed.flag" ]; then log "SKIP $TAG/$name (completed)"; return 0; fi
  if ! mkdir "$LOCKS/$name" 2>/dev/null; then log "SKIP $TAG/$name on GPU $gpu (running elsewhere)"; return 0; fi
  if ! prereg_ok; then log "ABORT $TAG/$name: PREREG changed since launch"; rmdir "$LOCKS/$name"; return 1; fi
  for attempt in 1 2; do
    log "LAUNCH $TAG/$name GPU $gpu attempt $attempt: seed ${SEED[$name]}, theta_tar ${TT[$name]}, n_train $N_TRAIN, n_test $N_TEST"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train/clopath_port_mnist.py train --seed "${SEED[$name]}" --device cuda:0 \
        --grid "$GRID" --theta_tar "${TT[$name]}" --no_random --prereg "$PREREG" \
        --n_train "$N_TRAIN" --n_test "$N_TEST" --out "$out" >> "$LOG/${TAG}_$name.log" 2>&1 </dev/null
    rc=$?
    log "END $TAG/$name GPU $gpu attempt $attempt rc=$rc $( [ -f "$out/completed.flag" ] && echo completed || echo NOT-COMPLETED )"
    [ -f "$out/completed.flag" ] && break
  done
  rmdir "$LOCKS/$name"
}

queue_gpu1() {
  run_job 1 seed0_all
  run_job 1 seed1_a
  for name in seed2_all seed1_b; do run_job 1 "$name"; done   # whatever GPU 0 did not take
}

queue_gpu0() {
  local name others
  for name in seed2_all seed1_b; do
    others=$(gpu0_others)
    if [ -n "$others" ]; then
      log "GPU 0 not free (${others//$'\n'/; }): $TAG/$name left for GPU 1"
      continue
    fi
    run_job 0 "$name"
  done
}

prereg_ok || { log "ABORT $TAG: PREREG changed and no frozen copy found"; exit 1; }
[ -f experiments/runs/clopath_port/thetatar_repro/seed0/completed.flag ] && \
  [ -f results/clopath_port_thetatar/check_default_reproduction.json ] && \
  grep -q '"all_identical": true' results/clopath_port_thetatar/check_default_reproduction.json || \
  { log "ABORT $TAG: check 2 (default reproduction) has not passed"; exit 1; }
log "$TAG start: prereg ${PREREG_SHA_AT_LAUNCH:0:16}, code models/clopath_port.py:$(sha256sum models/clopath_port.py | cut -c1-16) scripts/train/clopath_port_mnist.py:$(sha256sum scripts/train/clopath_port_mnist.py | cut -c1-16)"
queue_gpu1 &
P1=$!
sleep 30
queue_gpu0 &
P0=$!
wait $P0
wait $P1
# GPU 1's queue may have skipped a job that GPU 0 was still running; nothing is left only if all four completed
for name in seed0_all seed1_a seed1_b seed2_all; do
  [ -f "$RUNS/$name/completed.flag" ] || log "WARNING $TAG/$name not completed"
done
log "$TAG all done"
