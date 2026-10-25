#!/usr/bin/env bash
# PROBE-FIX ADD-ON (2026-10-19): the voltage-STDP threshold and spike-plateau experiments
# (Section 4.4, Table A.7a, Table 3 rows 4-5) re-run under the train-fit protocol of
# scripts/train/probefix_matrix_n5.sh, so they can be scored like the reference network.
#
# Every run is the voltage-STDP MNIST reference of the probe-fix matrix
# (voltage_S_full_N400_seed{s}_probefix_trained: tuned B3 configuration, STDP on 10,000
# training images, frozen pass over those images, full test set) with ONE plasticity
# setting changed. The thresholds and the plateau act only inside the plasticity rule
# (stdp_repro/code/models/fc_stdp_layer.py), so with STDP disabled every variant is the
# same network: the random baseline is voltage_S_full_N400_seed{s}_probefix_frozen.
#
# Grid: 10 settings x seeds 0-4 = 50 trained runs.
#   theta- in {-68, -65, -62, -60, -58, -55, -50} mV   (old: voltage_theta_ltd_sweep_3seed.sh)
#   theta+ = -52 mV                                    (old: voltage_theta_ltp_shift.sh)
#   theta+ = -52, theta- = -65 mV (Table 3 row 4)      (old: thltd_m65_thltp_m52)
#   Clopath's spike plateau (Table 3 row 5)            (old: voltage_plateau.sh, T_ap0.001)
# Run dirs: experiments/runs/voltage_S_full_N400_seed{s}_probefix_{label}
# Nothing is deleted. Idempotent: a run with completed.flag is skipped.
#
# Usage: nohup bash scripts/train/probefix_voltage_thresholds_n5.sh "0 1" 4 > results/logs_probefix_thr/driver.out 2>&1 &
#   arg1 = GPUs to use (default "0"; the paper's runs used "0 1"), arg2 = parallel runs per GPU (default 4).
#   Writes results/logs_probefix_thr/ALL_DONE when every run has finished.
# Aggregate: python scripts/score/probefix_voltage_thresholds_aggregate.py
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"   # or set PY=/path/to/env/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOGS=results/logs_probefix_thr; mkdir -p "$LOGS"; rm -f "$LOGS/ALL_DONE"
GPUS=(${1:-0}); SLOTS=${2:-4}
# Identical to scripts/train/probefix_matrix_n5.sh (voltage rule, MNIST, trained).
B3="--override stdp.A_plus=0.001 --override stdp.A_minus=0.003 --override stdp.tau_pre=16.8 --override scaffold.tc_theta_decay=1.0e4 --override scaffold.inh_weight=5.0 --override neuron.b_adapt=0.1 --override neuron.I_sp=0.2 --override weight_norm.norm=313.6"
COMMON="--override training.n_train=10000 --override training.n_test=1000000 --override evaluation.save_train_features=true"

setting() { # label -> overrides
  case "$1" in
    thltd_m68) echo "--override stdp.theta_ltd=-68.0" ;;
    thltd_m65) echo "--override stdp.theta_ltd=-65.0" ;;
    thltd_m62) echo "--override stdp.theta_ltd=-62.0" ;;
    thltd_m60) echo "--override stdp.theta_ltd=-60.0" ;;
    thltd_m58) echo "--override stdp.theta_ltd=-58.0" ;;
    thltd_m55) echo "--override stdp.theta_ltd=-55.0" ;;
    thltd_m50) echo "--override stdp.theta_ltd=-50.0" ;;
    thltp_m52) echo "--override stdp.theta_ltp=-52.0" ;;
    thltd_m65_thltp_m52) echo "--override stdp.theta_ltd=-65.0 --override stdp.theta_ltp=-52.0" ;;
    plateau) echo "--override stdp.spike_plateau=true --override stdp.gate_cap_mv=1000" ;;
  esac
}
LABELS="thltd_m68 thltd_m65 thltd_m62 thltd_m60 thltd_m58 thltd_m55 thltd_m50 thltp_m52 thltd_m65_thltp_m52 plateau"

launch() { # label seed gpu
  local label=$1 seed=$2 gpu=$3
  local rd="experiments/runs/voltage_S_full_N400_seed${seed}_probefix_${label}" tag="${label}.s${seed}"
  if [ -f "$rd/completed.flag" ]; then echo "[$(date '+%m-%d %H:%M')] SKIP $tag" >> "$LOGS/master.log"; echo SKIP; return; fi
  [ -d "$rd" ] && rm -rf "$rd"           # an unfinished earlier attempt of this exact run
  echo "[$(date '+%m-%d %H:%M')] LAUNCH $tag on cuda:$gpu" >> "$LOGS/master.log"
  ( "$PY" -u experiments/scripts/run_single.py --rule voltage --scaffold full --seed "$seed" \
      --n_neurons 400 --device "cuda:$gpu" --dataset mnist $COMMON $B3 $(setting "$label") \
      --name-suffix "probefix_${label}" > "$LOGS/${tag}.log" 2>&1
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
for seed in 0 1 2 3 4; do for label in $LABELS; do JOBS+=("$label $seed"); done; done
echo "[$(date '+%m-%d %H:%M')] THRESHOLD ADD-ON START: ${#JOBS[@]} runs, GPUs ${GPUS[*]} x $SLOTS" >> "$LOGS/master.log"

for job in "${JOBS[@]}"; do
  read -r label seed <<<"$job"
  while true; do
    placed=no
    for g in "${GPUS[@]}"; do
      if [ "$(running_on "$g")" -lt "$SLOTS" ]; then
        pid=$(launch "$label" "$seed" "$g"); [ "$pid" != SKIP ] && PIDS[$pid]=$g
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
echo "[$(date '+%m-%d %H:%M')] THRESHOLD ADD-ON FINISHED: $n_ok runs complete (plus any SKIPs)" >> "$LOGS/master.log"
echo "finished $(date) ; complete=$n_ok" > "$LOGS/ALL_DONE"
