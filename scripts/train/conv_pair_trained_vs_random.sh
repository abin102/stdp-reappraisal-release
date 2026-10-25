#!/usr/bin/env bash
# Conv trained-vs-random for PAIR-STDP at n=5.
#
# Section 4.3 establishes for VOLTAGE that trained conv features are no more
# linearly separable than random conv features (results/conv_arch_logreg_n5.json).
# This extends the same control to pair-STDP, whose five trained seeds are the
# weights conv2l_stdp_ckpt_pairgrade_seed{0..4}.pt at the repository root
# (trained with conv/conv2layer_stdp_surrogate.py; not shipped, see RUN_MAP.md).
#
# No STDP training happens here: each run loads frozen weights (or a random init)
# and does a forward pass to extract the 11008-d pooled conv features.
#
# FAILURE HANDLING (added 2026-09-10 after a silent 8/10 run).
# The 2026-09-09 sweep lost two cells to transient CUDA faults ("illegal memory
# access" on trained_s1, "unspecified launch failure" on random_s4) and still
# printed "complete", so the log was indistinguishable from a clean run and the
# missing cells were only caught by listing the output files. This version:
#   * verifies every .npz actually loads with the expected shapes,
#   * deletes partial/corrupt output instead of leaving it to be skipped later,
#   * retries a failed cell once after a GPU cool-down (the faults are transient
#     -- the same code path succeeded on eight other cells in the same sweep),
#   * prints an explicit per-cell PASS/FAIL summary and exits non-zero if any
#     cell is missing, so a failed sweep cannot be mistaken for a finished one.
# Re-running is safe and cheap: verified cells are skipped, so only the missing
# ones are recomputed.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
PY="${PY:-python}"
export PYTHONPATH=.
GPU="${1:-${DEVICE:-cuda:0}}"
N_TRAIN=12000
N_TEST=10000
N_FEAT=11008

ok=(); bad=()

# A cell counts as done only if the npz loads and carries the expected arrays at
# the expected shapes. np.savez_compressed writes the archive last, but a crash
# mid-write (or a truncated file on a full disk) would otherwise be skipped
# silently on the next run.
verify() {
  "$PY" - "$1" "$N_TRAIN" "$N_TEST" "$N_FEAT" <<'PYEOF' 2>/dev/null
import sys, numpy as np
p, ntr, nte, nf = sys.argv[1], *map(int, sys.argv[2:5])
z = np.load(p)
assert set(z.files) >= {"Xtr","ytr","Xte","yte"}, z.files
assert z["Xtr"].shape == (ntr, nf), z["Xtr"].shape
assert z["Xte"].shape == (nte, nf), z["Xte"].shape
assert z["ytr"].shape == (ntr,) and z["yte"].shape == (nte,)
PYEOF
}

run_cell() {                      # $1=label  $2=ckpt|RANDOM  $3=seed
  local label=$1 ck=$2 seed=$3
  local out="results/conv_feats_${label}.npz"

  if [ -f "$out" ]; then
    if verify "$out"; then
      echo "[$(date +%H:%M:%S)] $label -- already present and verified, skipping"
      ok+=("$label"); return 0
    fi
    echo "[$(date +%H:%M:%S)] $label -- existing file is corrupt/partial, deleting"
    rm -f "$out"
  fi

  for attempt in 1 2; do
    echo "[$(date +%H:%M:%S)] $label (attempt $attempt/2)"
    if "$PY" -u scripts/extract/extract_pair_feats.py "$label" "$ck" "$GPU" "$N_TRAIN" "$seed"; then
      if verify "$out"; then
        echo "[$(date +%H:%M:%S)] $label -- OK"
        ok+=("$label"); return 0
      fi
      echo "[$(date +%H:%M:%S)] $label -- exited 0 but output failed verification"
    else
      echo "[$(date +%H:%M:%S)] $label -- extractor exited $?"
    fi
    rm -f "$out"
    if [ "$attempt" -eq 1 ]; then
      echo "[$(date +%H:%M:%S)] $label -- cooling down 120s before retry"
      sleep 120
    fi
  done

  echo "[$(date +%H:%M:%S)] $label -- FAILED after 2 attempts"
  bad+=("$label"); return 1
}

for s in 0 1 2 3 4; do
  ck="conv2l_stdp_ckpt_pairgrade_seed${s}.pt"
  if [ ! -f "$ck" ]; then
    echo "MISSING checkpoint $ck -- cannot extract trained_s${s}"
    bad+=("pairN5_trained_s${s} (no checkpoint)")
  else
    run_cell "pairN5_trained_s${s}" "$ck"  "$s"
  fi
  run_cell "pairN5_random_s${s}"  RANDOM  "$s"
done

echo
echo "================ SUMMARY ================"
echo "verified cells : ${#ok[@]}/10"
for c in "${ok[@]:-}";  do [ -n "$c" ] && echo "   PASS  $c"; done
for c in "${bad[@]:-}"; do [ -n "$c" ] && echo "   FAIL  $c"; done
if [ "${#bad[@]}" -gt 0 ]; then
  echo "-----------------------------------------"
  echo "INCOMPLETE: ${#bad[@]} cell(s) missing. Re-run this script to retry only those."
  echo "Do NOT build results/conv_pair_n5.json from a partial set: the"
  echo "trained-vs-random delta needs matched seed pairs."
  exit 1
fi
echo "all 10 cells present and verified"
echo "[$(date +%H:%M:%S)] pair conv extraction complete"
