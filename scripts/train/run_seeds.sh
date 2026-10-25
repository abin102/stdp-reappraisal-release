#!/usr/bin/env bash
# Train network families over a list of seeds, one sequential worker per GPU.
#
# Usage:  SEEDS="0 1 2 3 4" GPUS="0 1" bash scripts/train/run_seeds.sh FAMILY [FAMILY ...]
#   FAMILY = the stem of a run_configs/<FAMILY>.yaml (run through scripts/train/run_config.py), or
#   conv_stdp            conv voltage-STDP + surrogate readout       -> checkpoints/..._paper_seed<S>
#   conv_trainedreadout  fresh surrogate readout on conv_stdp seed S -> checkpoints/..._TRAINEDREADOUT_seed<S>
#   conv_randomfilters   surrogate readout on random filters         -> checkpoints/..._RANDOMFILTERS_seed<S>
#   conv_svm             linear readout of trained (needs conv_stdp) and random conv features
#                                                                    -> results/conv_svm_{trained,random}_s<S>.json
#   port_grid10k, port_explore_k0   Clopath port                     -> experiments/runs/clopath_port/<tag>/seed<S>
# Which family feeds which paper element, and its seeds: run_configs/INDEX.md.
# Env: PY (python, default "python"), SEEDS (default "0 1 2 3 4"), GPUS (default "0"),
#      LOG_DIR (default logs_run_seeds). Completed runs are skipped, so re-running resumes.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"; SEEDS="${SEEDS:-0 1 2 3 4}"; GPUS="${GPUS:-0}"; LOG_DIR="${LOG_DIR:-logs_run_seeds}"
export PYTHONPATH=. WANDB_MODE=disabled
[ $# -gt 0 ] || { sed -n '2,17p' "$0"; exit 1; }
mkdir -p "$LOG_DIR"
CONV=conv2l_voltage_clopath_faithful_mnist
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/master.log"; }

run_job() {   # family seed  (runs on cuda:0 = the GPU in CUDA_VISIBLE_DEVICES)
  local f=$1 s=$2
  case $f in
  conv_stdp)
    "$PY" -u conv/conv2layer_voltage_stdp.py --config conv/configs/conv_voltage_config.yaml \
        --seed "$s" --name-suffix "_paper_seed$s" --device cuda:0 ;;
  conv_trainedreadout)
    local d=checkpoints/${CONV}_TRAINEDREADOUT_seed$s
    [ -f "$d/best_readout.pth" ] && { echo "SKIP (done): $d"; return 0; }
    mkdir -p "$d" && cp -n "checkpoints/${CONV}_paper_seed$s/ckpt_L2_ep1.pt" "$d/conv_voltage_stdp_ckpt.pt" &&
    "$PY" -u conv/conv2layer_voltage_stdp.py --config conv/configs/conv_voltage_config.yaml \
        --seed "$s" --name-suffix "_TRAINEDREADOUT_seed$s" --device cuda:0 ;;
  conv_randomfilters)
    local c=conv/configs/conv_seed5/randomfilters_seed$s.yaml
    [ -f "$c" ] || c=run_configs/conv/randomfilters_seed$s.yaml
    "$PY" -u conv/conv2layer_voltage_stdp.py --config "$c" --device cuda:0 ;;
  conv_svm)
    local lab extra
    for lab in trained_s$s random_s$s; do
      [ -f "results/conv_svm_$lab.json" ] && continue
      if [ "$lab" = "trained_s$s" ]; then extra="--ckpt checkpoints/${CONV}_paper_seed$s/conv_voltage_stdp_ckpt.pt"
      else extra="--random --seed $s"; fi
      "$PY" -u scripts/extract/conv_svm_readout.py --net voltage --label "$lab" --device cuda:0 \
          --n-train 12000 --n-test 10000 $extra &&
      "$PY" -u scripts/score/classify_feats_torch.py "$lab" cuda:0 || return 1
      rm -f "results/conv_feats_$lab.npz"
    done ;;
  port_grid10k)
    "$PY" -u scripts/train/clopath_port_mnist.py train --seed "$s" --device cuda:0 \
        --grid results/clopath_port_calibration/grid.json --n_train 10000 --n_test 10000 \
        --out "experiments/runs/clopath_port/grid10k/seed$s" ;;
  port_explore_k0)
    "$PY" -u scripts/train/clopath_port_mnist.py train --seed "$s" --device cuda:0 \
        --grid run_configs/clopath_port/grid_explore_k0.json --n_train 10000 --n_test 5000 \
        --out "experiments/runs/clopath_port/explore_k0/seed$s" ;;
  *)
    [ -f "run_configs/$f.yaml" ] || { echo "unknown family: $f (see run_configs/INDEX.md)"; return 2; }
    "$PY" -u scripts/train/run_config.py "run_configs/$f.yaml" --seed "$s" --device cuda:0 ;;
  esac
}

JOBS=(); for f in "$@"; do for s in $SEEDS; do JOBS+=("$f $s"); done; done
GPU_LIST=($GPUS); NG=${#GPU_LIST[@]}
log "start: ${#JOBS[@]} jobs ($*; seeds $SEEDS) on GPUs $GPUS"
for i in "${!GPU_LIST[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]}
    for ((j = i; j < ${#JOBS[@]}; j += NG)); do
      set -- ${JOBS[$j]}
      if run_job "$1" "$2" >> "$LOG_DIR/$1_seed$2.log" 2>&1; then log "OK   $1 seed $2 (GPU ${GPU_LIST[$i]})"
      else log "FAIL $1 seed $2 (GPU ${GPU_LIST[$i]}), see $LOG_DIR/$1_seed$2.log"; fi
    done
  ) &
done
wait
log "done"
