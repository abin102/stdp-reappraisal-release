#!/usr/bin/env python3
"""Matched random-initialisation baseline for the 60k-trained MNIST pair-STDP
networks (Section 4.2: random 69.7 %, trained 85.9 %, +16.2 points; Figure 6 dashed line).

Trained : experiments/runs/pair_exp_S_full_N400_seed{0..4}            (60k train, STDP on)
Random  : experiments/runs/pair_exp_S_full_N400_seed{0..4}_frozen60k  (60k train, STDP off,
          training.frozen_init=true; identical config otherwise)

Readouts:
  train_fit : LogisticRegression(max_iter=100000, C=1.0) fit on the 10k
              train_spike_counts.npz, scored on the full 10k test set
              (the Section 4.1 protocol, scripts/score/scaffold_readout_table.py).
              Needs train_spike_counts.npz (scripts/extract/extract_train_features.py).
  posthoc   : the run's own post-hoc accuracy (metrics.json "accuracy").
(The release no longer computes the retired 70/30 test-split probe or the 10k random
baseline of the old ten-seed matrix.)

Usage (repo root): PYTHONPATH=. python scripts/score/matched_random_60k_table.py [--runs DIR]
Writes results/matched_random_60k.json and prints a table. Fails loudly on a missing cell.
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

from threadpoolctl import threadpool_limits
threadpool_limits(1)   # single-threaded BLAS: probe fits reproduce bit for bit (scripts/score/_blas.py)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "matched_random_60k.json"
SEEDS = range(5)
CELLS = {"trained60k": "pair_exp_S_full_N400_seed{}",
         "random60k": "pair_exp_S_full_N400_seed{}_frozen60k"}


def train_fit(d):
    tr, te = np.load(d / "train_spike_counts.npz"), np.load(d / "test_spike_counts.npz")
    clf = LogisticRegression(max_iter=100000, C=1.0, n_jobs=4).fit(tr["train_spike_counts"], tr["train_labels"])
    return 100 * clf.score(te["test_spike_counts"], te["test_labels"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "experiments" / "runs"))
    R = Path(ap.parse_args().runs)
    res = {"train_fit": {}, "posthoc": {}}
    for key, pat in CELLS.items():
        tf, ph = [], []
        for s in SEEDS:
            d = R / pat.format(s)
            if not (d / "completed.flag").exists():
                sys.exit(f"ERROR: missing or incomplete run {d}")
            ph.append(100 * json.load(open(d / "metrics.json"))["accuracy"])
            if not (d / "train_spike_counts.npz").exists():
                sys.exit(f"ERROR: {d}/train_spike_counts.npz missing (scripts/extract/extract_train_features.py)")
            tf.append(train_fit(d))
        res["train_fit"][key] = tf; res["posthoc"][key] = ph
    json.dump(res, open(OUT, "w"), indent=1)

    def row(name, a):
        a = np.asarray(a); return f"{name:11s} " + "  ".join(f"{v:5.1f}" for v in a) + f"   mean {a.mean():5.1f} ± {a.std(ddof=1):.1f}"
    for prot in ("train_fit", "posthoc"):
        print(f"\n{prot}  (seeds 0-4)")
        for k, v in res[prot].items():
            print(row(k, v))
        print(row("delta60k", np.subtract(res[prot]["trained60k"], res[prot]["random60k"])))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
