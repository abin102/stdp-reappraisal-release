#!/usr/bin/env python3
"""Post-hoc readout under both conventions, for every run the paper quotes one for.

  Diehl-Cook (2015): assignment = argmax over classes of the MEAN spike count per
                     training image; prediction = per-class sum of assigned-neuron
                     counts divided by the number of neurons assigned to that class.
  summed variant   : both divisions omitted (what this repository's training pipeline
                     computes; the paper reports the Diehl-Cook rule above as primary).

Both are rebuilt from each run's saved `class_rates` and `test_spike_counts`, so no
retraining or feature extraction happens here.

Covers: the 60 MNIST scaffold networks (Section 4.1, Figure 2, Table A.3). The release
dropped the 88 networks of the retired tier-1 table (old Table A.2) and its run list
(results/tier1_source_map.txt); no current table uses them.

Since 2026-10-27 the Diehl-Cook labels are formed from the FROZEN spike counts of the first
10,000 training images (train_spike_counts.npz, the same counts the linear probe is fitted on), as
Diehl and Cook (2015) and Methods 3.10 describe, and neurons that never fire on those images are
left unassigned. Before, they came from `class_rates`, accumulated during training while the
weights were still changing, and silent neurons fell to class 0; that value is kept as
`posthoc_dc_trainingpass`.

Writes results/posthoc_both_conventions.csv and prints the per-group tables.

Usage (repo root): PYTHONPATH=. python scripts/score/posthoc_both_conventions.py [--runs DIR]
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

from experiments.scripts.run_single import load_dataset

R = Path("experiments/runs")
OUT = Path("results/posthoc_both_conventions.csv")
_counts = {}


def class_counts(ds):
    """Training-set class counts; N-MNIST inherits MNIST's labels (same 60k digits)."""
    if ds not in _counts:
        key = "mnist" if ds == "nmnist" else ds
        tr, _ = load_dataset("./data", key)
        y = np.array(tr.targets) if hasattr(tr, "targets") else np.array([int(tr[i][1]) for i in range(len(tr))])
        _counts[ds] = np.bincount(y).astype(np.float64)
    return _counts[ds]


def both(run, ds, frozen=False):
    d = R / run
    f = d / "test_spike_counts_full.npz"          # 26-letter EMNIST re-extraction
    z = np.load(f if f.exists() else d / "test_spike_counts.npz")
    X, y = z["test_spike_counts"].astype(np.float64), z["test_labels"]
    # the 26-letter re-extraction stores only test features; the per-class training
    # counts that define the labels live in the run's original file
    src = z if "class_rates" in z.files else np.load(d / "test_spike_counts.npz")
    cr = np.asarray(src["class_rates"], dtype=np.float64)
    cnt = class_counts(ds)
    # EMNIST-Letters ships 1-indexed targets (bin 0 empty) while the runs store 0-based
    # labels, so drop the empty leading bin before pairing counts with columns
    if len(cnt) > cr.shape[1] and cnt[0] == 0:
        cnt = cnt[1:]
    C = cr.shape[1] if cr.ndim == 2 and cr.shape[1] <= len(cnt) else len(cnt)
    if cr.shape[0] == C and cr.shape[1] != C:
        cr = cr.T
    cr, cnt = cr[:, :C], cnt[:C]

    def score(asg, divide):
        v = np.zeros((len(y), C))
        for c in range(C):
            m = asg == c
            if m.any():
                v[:, c] = X[:, m].sum(1) / (m.sum() if divide else 1.0)
        pred = v.argmax(1)
        return 100.0 * (pred == y).mean(), pred
    a_sum, pred_sum = score(cr.argmax(1), False)
    a_dc, pred_dc = score((cr / cnt[None, :]).argmax(1), True)
    a_dc_tp = a_dc
    if frozen:
        zt = np.load(d / "train_spike_counts.npz")
        Xtr, ytr = zt["train_spike_counts"].astype(np.float64), zt["train_labels"]
        means = np.stack([Xtr[ytr == c].mean(0) if (ytr == c).any() else np.zeros(Xtr.shape[1])
                          for c in range(C)], 1)
        asg = means.argmax(1)
        asg[Xtr.sum(0) == 0] = -1                   # never fires on the labelling images
        a_dc, pred_dc = score(asg, True)
    conc = lambda p: 100.0 * np.bincount(p, minlength=C).max() / len(p)
    return dict(run=run, dataset=ds, n_test=len(y), n_classes=C,
                posthoc_dc=round(a_dc, 2), posthoc_dc_trainingpass=round(a_dc_tp, 2),
                label_source="frozen_train_counts" if frozen else "class_rates_training_pass",
                posthoc_summed=round(a_sum, 2),
                difference_pp=round(a_dc - a_sum, 2),
                pct_images_one_class_dc=round(conc(pred_dc), 1),
                pct_images_one_class_summed=round(conc(pred_sum), 1))


def _check_runs_present():
    """experiments/runs/ is gitignored, so a fresh clone has no run data."""
    probe = R / "pair_exp_S_full_N400_seed0" / "test_spike_counts.npz"
    if not probe.exists():
        sys.exit(
            f"ERROR: missing run data ({probe})\n\n"
            "This script rebuilds both post-hoc conventions from each run's saved\n"
            "spike counts in experiments/runs/, which is gitignored -- a clone of this\n"
            "repository has the analysis code and the summary tables in results/, but\n"
            "not the trained networks themselves.\n\n"
            "To regenerate the runs:  experiments/scripts/run_single.py (see README)\n"
            "To read the already-computed numbers instead, without retraining:\n"
            "  results/posthoc_both_conventions.csv (Figure 2, Table A.2, Table A.3)"
        )


def main():
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(R), help="folder holding the run directories")
    R = Path(ap.parse_args().runs)
    _check_runs_present()
    rows = []
    # --- scaffold matrix ----------------------------------------------------
    for rule in ("pair_exp", "triplet", "voltage"):
        for sc in ("S_full", "S_inh_only", "S_thr_only", "S_none"):
            for s in range(5):
                # Voltage uses the CORRECTED re-runs (tc_theta_decay = 1e4); the bare
                # names are the superseded tc=1e6 cells. Must match
                # scripts/score/scaffold_readout_table.py:cell_dir, or the figure's
                # per-run join on "run" breaks. See results/voltage_audit_2026-09.md.
                name = f"{rule}_{sc}_N400_seed{s}"
                if rule == "voltage":
                    for suffix in ("__b_fix_tc1e4", "__b_fix_tc1e4_60k"):
                        if (R / f"{name}{suffix}").is_dir():
                            name = f"{name}{suffix}"
                            break
                r = both(name, "mnist", frozen=True)
                r.update(group="scaffold", rule=rule, scaffold=sc, seed=s); rows.append(r)
    cols = ["group", "rule", "dataset", "scaffold", "seed", "run", "n_test", "n_classes",
            "posthoc_dc", "posthoc_dc_trainingpass", "label_source", "posthoc_summed", "difference_pp",
            "pct_images_one_class_dc", "pct_images_one_class_summed"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])

    def agg(sel, key):
        g = defaultdict(list)
        for r in rows:
            if sel(r):
                g[key(r)].append(r)
        for k in sorted(g):
            v = g[k]
            dc = np.array([x["posthoc_dc"] for x in v]); su = np.array([x["posthoc_summed"] for x in v])
            print(f"  {str(k):34s} D&C {dc.mean():6.2f}±{dc.std(ddof=1) if len(dc)>1 else 0:4.2f} | "
                  f"summed {su.mean():6.2f}±{su.std(ddof=1) if len(su)>1 else 0:4.2f} | diff {dc.mean()-su.mean():+6.2f} pp")

    print("\n=== scaffold matrix (MNIST, 5 seeds) ===")
    agg(lambda r: r["group"] == "scaffold", lambda r: (r["rule"], r["scaffold"]))
    print(f"\nwrote {OUT} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
