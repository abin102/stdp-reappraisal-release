#!/usr/bin/env python3
"""Section 4.5 (MNIST, training length) under the paper's single linear probe, from the
re-extracted spike counts (scripts/extract/reextract_features.py, results/logs_reextract/), plus the
validation of the re-extraction against fresh probe-fix runs.

Probe: multinomial logistic regression, L2, C = 1, raw counts, max_iter=100000, fitted on the
spike counts of 10,000 training images and scored on the full 10,000-image MNIST test set.

Curves (seeds 0-4, full scaffold):
  * pair-/triplet-STDP and default voltage-STDP: {rule}_S_full_N400_seed{s}_flen{5k,10k,20k};
    60k = the scaffold-matrix networks ({rule}_S_full_N400_seed{s}; voltage-STDP
    voltage_S_full_N400_seed0__b_fix_tc1e4_60k and _seed{1,2}__b_fix_tc1e4), as in
    results/forgetting_curve.json.
  * tuned voltage-STDP: voltage_S_full_N400_seed{s}_{vhz2000,vhz5000,mnist_vhz60000}.
Spike-count sources per network:
  * "new"      train_spike_counts_pf.npz -> test_spike_counts_pf.npz (both re-extracted);
  * "mixed"    train_spike_counts_pf.npz -> test_spike_counts.npz (original test pass);
  * "stage1"   train_spike_counts.npz (first 10,000) -> test_spike_counts.npz; the only source
               for the pair-/triplet-STDP 60k networks, which were not re-extracted;
  * "old"      the retired probe: stratified 70/30 split (random_state 0) of the original test
               counts, which reproduces results/forgetting_curve.json.
Pair-/triplet-STDP thresholds decay slowly (tau 1e7 ms), so a re-extracted pass starts after the
original test pass and fires 12-17% less; "new" and "mixed" bracket that effect.

Validation: voltage-STDP thltd_ref and plateau_T_ap0.001 (seeds 0-2), re-extracted from the old
runs, against the fresh probe-fix runs with the same seed and configuration
(voltage_S_full_N400_seed{s}_probefix_{trained,plateau}).

Outputs: results/section45_probefix_mnist.json, results/section45_probefix_mnist.md
Usage (repo root): python scripts/score/section45_probefix_mnist.py
"""
import json, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from scripts.score._blas import fixed_threads
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
R = Path("experiments/runs")
MAX_ITER = 100000
SEEDS = (0, 1, 2, 3, 4)       # five seeds since 2026-10-23 (seeds 3-4 added by scripts/run_from_template.py)
VAL_SEEDS = (0, 1, 2)         # the validation pairs exist for seeds 0-2 only
OUT = Path("results/section45_probefix_mnist")


def load(f):
    z = np.load(f, allow_pickle=True)
    X = [z[k] for k in z.files if z[k].ndim == 2][0].astype(np.float64)
    y = [z[k] for k in z.files if z[k].ndim == 1 and "label" in k][0]
    return X, y


def fit(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=MAX_ITER, C=1.0).fit(Xtr, ytr)
    return 100 * clf.score(Xte, yte), int(clf.n_iter_.max()) < MAX_ITER


def score(d):
    d = R / d
    out = {"run": d.name}
    Xo, yo = load(d / "test_spike_counts.npz")
    a, b, ya, yb = train_test_split(Xo, yo, test_size=0.3, stratify=yo, random_state=0)
    out["old"], _ = fit(a, ya, b, yb)
    if (d / "train_spike_counts_pf.npz").exists():
        Xt, yt = load(d / "train_spike_counts_pf.npz")
        Xn, yn = load(d / "test_spike_counts_pf.npz")
        assert len(yt) == 10000 and len(yn) == 10000, d
        out["new"], out["new_conv"] = fit(Xt, yt, Xn, yn)
        out["mixed"], _ = fit(Xt, yt, Xo, yo)
        out["spikes_test_orig"], out["spikes_test_new"] = Xo.sum(1).mean(), Xn.sum(1).mean()
    elif (d / "train_spike_counts.npz").exists():
        Xs, ys = load(d / "train_spike_counts.npz")
        assert len(ys) >= 10000 and len(yo) == 10000, d
        out["stage1"], out["stage1_conv"] = fit(Xs[:10000], ys[:10000], Xo, yo)
    return out


def curve_runs():
    rows = []
    for rule in ("pair_exp", "triplet", "voltage"):
        for s in SEEDS:
            for k, n in (("5k", 5000), ("10k", 10000), ("20k", 20000)):
                rows.append((rule, "default", n, s, f"{rule}_S_full_N400_seed{s}_flen{k}"))
            if rule == "voltage":
                d60 = "voltage_S_full_N400_seed0__b_fix_tc1e4_60k" if s == 0 else f"voltage_S_full_N400_seed{s}__b_fix_tc1e4"
            else:
                d60 = f"{rule}_S_full_N400_seed{s}"
            rows.append((rule, "default", 60000, s, d60))
    for s in SEEDS:
        for t, n in (("vhz2000", 2000), ("vhz5000", 5000), ("mnist_vhz60000", 60000)):
            rows.append(("voltage", "tuned", n, s, f"voltage_S_full_N400_seed{s}_{t}"))
    return rows


def validation_pairs():
    return [(f"{name} seed {s}", f"voltage_S_full_N400_seed{s}_{old}", f"voltage_S_full_N400_seed{s}_probefix_{new}")
            for name, old, new in (("reference", "thltd_ref", "trained"), ("spike plateau", "plateau_T_ap0.001", "plateau"))
            for s in VAL_SEEDS]


def fresh(d):
    Xt, yt = load(R / d / "train_spike_counts.npz")
    Xn, yn = load(R / d / "test_spike_counts.npz")
    return fit(Xt, yt, Xn, yn)[0], Xn.sum(1).mean()


def main():
    rows = curve_runs()
    res = Parallel(n_jobs=12)(delayed(fixed_threads(score, 1))(r[-1]) for r in rows)
    by = {r["run"]: r for r in res}
    vp = validation_pairs()
    val_old = Parallel(n_jobs=6)(delayed(fixed_threads(score, 1))(o) for _, o, _ in vp)
    val_new = Parallel(n_jobs=6)(delayed(fixed_threads(fresh, 1))(n) for _, _, n in vp)

    def cell(rule, cfg, n, key):
        v = [by[d][key] for (r, c, m, s, d) in rows if (r, c, m) == (rule, cfg, n) and key in by[d]]
        return (float(np.mean(v)), float(np.std(v, ddof=1)), len(v)) if v else None

    def best(rule, cfg, n):
        """single-probe value: re-extracted counts if present, else stage1 (pair/triplet 60k)."""
        return cell(rule, cfg, n, "new") or cell(rule, cfg, n, "stage1")

    curves = []
    for rule, cfg, ns in (("pair_exp", "default", (5000, 10000, 20000, 60000)),
                          ("triplet", "default", (5000, 10000, 20000, 60000)),
                          ("voltage", "default", (5000, 10000, 20000, 60000)),
                          ("voltage", "tuned", (2000, 5000, 60000))):
        for n in ns:
            curves.append({"rule": rule, "config": cfg, "n_train": n, "single_probe": best(rule, cfg, n),
                           "mixed": cell(rule, cfg, n, "mixed"), "old_7030": cell(rule, cfg, n, "old")})
    val = [{"pair": name, "reextracted_old_run": o.get("new"), "fresh_probefix_run": f[0],
            "diff": o.get("new") - f[0], "spikes_reextracted": o.get("spikes_test_new"), "spikes_fresh": f[1]}
           for (name, _, _), o, f in zip(vp, val_old, val_new)]
    nonconv = [r["run"] for r in res + val_old if r.get("new_conv") is False or r.get("stage1_conv") is False]
    json.dump({"probe": f"logreg L2 C=1 raw counts max_iter={MAX_ITER}; fit 10,000 training images, full test set",
               "curves": curves, "validation": val, "per_run": res, "not_converged": nonconv},
              open(f"{OUT}.json", "w"), indent=1)

    f = lambda c: "--" if c is None else f"{c[0]:.1f} ± {c[1]:.1f}"
    L = ["# Section 4.5 under the single linear probe (MNIST, full scaffold, seeds 0-4)", "",
         "Generated by `scripts/score/section45_probefix_mnist.py`; probe and sources in its docstring.",
         "Single probe = re-extracted training and test counts; the pair-/triplet-STDP 60k networks were not",
         "re-extracted and use their saved training counts with the original test pass (Table A.3).", "",
         "| rule | configuration | training images | single probe | mixed (orig. test) | old 70/30 |",
         "|---|---|---:|---:|---:|---:|"]
    lab = {"pair_exp": "pair-STDP", "triplet": "triplet-STDP", "voltage": "voltage-STDP"}
    for c in curves:
        L.append(f"| {lab[c['rule']]} | {c['config']} | {c['n_train']:,} | {f(c['single_probe'])} | "
                 f"{f(c['mixed'])} | {f(c['old_7030'])} |")
    L += ["", "## Validation of the re-extraction (voltage-STDP, single probe)", "",
          "| network | re-extracted old run | fresh probe-fix run | difference | spikes/test image (re-extr. / fresh) |",
          "|---|---:|---:|---:|---:|"]
    for v in val:
        L.append(f"| {v['pair']} | {v['reextracted_old_run']:.2f} | {v['fresh_probefix_run']:.2f} | "
                 f"{v['diff']:+.2f} | {v['spikes_reextracted']:.1f} / {v['spikes_fresh']:.1f} |")
    L += ["", f"Fits not converged at max_iter {MAX_ITER}: {len(nonconv)}" + (f" ({', '.join(nonconv)})" if nonconv else "")]
    open(f"{OUT}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
