#!/usr/bin/env python3
"""Score every network re-extracted by scripts/extract/reextract_features.py (results/logs_reextract/
jobs_runsingle.txt, 232 run_single networks; + jobs_homeo.txt, 18 homeostasis networks) with the paper's single linear probe, next to the
retired probes on the original test pass.

Single probe (Methods 3.9): multinomial logistic regression, L2, C = 1, raw counts,
max_iter=100000, fitted on train_spike_counts_pf.npz (frozen pass over the first 10,000 images of
the run's own presentation order) and scored on test_spike_counts_pf.npz (full test set, shuffled
order seed + 3000).
Retired probes, on the ORIGINAL test_spike_counts.npz of each run (as the old paper numbers):
  * old_v     "voltage-STDP probe": stratified 70/30 split (random_state 0) of the first 5,000 test
              images, max_iter 2000 (scripts/voltage_results_table.py COMMON);
  * old_full  stratified 70/30 split (random_state 0) of all saved test images, max_iter 2000
              (the training-length probe of Section 4.5 / Appendix G).
EMNIST-Letters and N-MNIST store their test sets sorted by class, and the original passes presented
them in that order, so old_v / old_full on those datasets carry that artifact.

Outputs
  results/reextract_probefix_scores.json  one record per network
  results/reextract_probefix_scores.md    (1) per-setting summary (seeds pooled), grouped by where the
                                          numbers are used in the paper; (2) per-network table
Pairing each trained setting with its STDP-disabled network (Delta) and the paper-table layouts are
NOT done here.
Usage (repo root): python scripts/score/reextract_probefix_score.py
"""
import json, re, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
from joblib import Parallel, delayed
from scripts.score._blas import fixed_threads
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
JOBS = Path("results/run_lists/jobs_runsingle.txt")
JOBS_EXTRA = [Path("results/run_lists/jobs_homeo.txt"),    # 18 homeostasis networks added 2026-10-21
              Path("results/run_lists/jobs_seeds34.txt")]  # seeds 3-4 of every 3-seed family, 2026-10-22
OUT = Path("results/reextract_probefix_scores")
MAX_ITER, OLD_ITER, N_OLD_V = 100000, 2000, 5000

# (paper use, regex on the run name); first match wins
FAMILIES = [
    ("Validation: re-extraction vs fresh probe-fix runs", r"_thltd_ref$"),
    ("Appendix E: homeostasis on the tuned network (60k)", r"(tier0_|r1volt)"),
    ("Appendix E: spike plateau (Table A.6); Table 3 spike-plateau row", r"_plateau_"),
    ("Table 3 weight-management row; Appendix E (Table A.5)", r"_fixref_"),
    ("Appendix C: hyperparameter search (Table A.4)", r"(phaseB|i4_smooth|mnist_adex|c_conv_parity)"),
    ("Section 4.6 / Figure 6 / Table A.3: voltage-STDP scaffolds at 60k", r"b_fix_tc1e4"),
    ("Section 4.6 / Appendix G: voltage-STDP scaffolds at 5k and 20k", r"(peak5k|mid20k)"),
    ("Table 3 training-length row (60k, trained vs STDP disabled)", r"vb3_(trained|frozen)60k$"),
    ("Section 4.5 / Figure 5: training length, default configurations", r"flen"),
    ("Section 4.5 / Appendix G: tuned voltage-STDP, 2k / 5k / 60k", r"vhz"),
    ("Appendix G: tuned voltage-STDP at 60k, other datasets", r"(_f3_fmnist_voltage_tuned|_nmnist_n5|_n5_organa_tuned|_n5_blood_tuned|smoke_tuned)$"),
]


def family(name):
    for lab, p in FAMILIES:
        if re.search(p, name):
            return lab
    return "other"


def load(f):
    z = np.load(f, allow_pickle=True)
    X = [z[k] for k in z.files if z[k].ndim == 2][0].astype(np.float64)
    y = [z[k] for k in z.files if z[k].ndim == 1 and "label" in k][0]
    return X, y


def fit(Xa, ya, Xb, yb, max_iter):
    clf = LogisticRegression(max_iter=max_iter, C=1.0).fit(Xa, ya)
    return 100 * clf.score(Xb, yb), bool(clf.n_iter_.max() < max_iter)


def strat(X, y, max_iter):
    a, b, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    return fit(a, ya, b, yb, max_iter)[0]


def score(d):
    d = Path(d)
    cfg = yaml.safe_load(open(d / "config.yaml"))
    tr = cfg.get("training", {}) or {}
    Xt, yt = load(d / "train_spike_counts_pf.npz")
    Xn, yn = load(d / "test_spike_counts_pf.npz")
    Xo, yo = load(d / "test_spike_counts.npz")
    new, conv = fit(Xt, yt, Xn, yn, MAX_ITER)
    return {"run": d.name, "family": family(d.name),
            "setting": re.sub(r"_seed\d+_", "_seed*_", d.name),
            "seed": int(re.search(r"_seed(\d+)", d.name).group(1)),
            "rule": d.name.split("_S_")[0], "dataset": (cfg.get("data", {}) or {}).get("dataset", "mnist"),
            "n_train_stdp": tr.get("n_train"), "stdp_disabled": bool(tr.get("frozen_init")),
            "n_probe_train": len(yt), "n_test": len(yn), "n_test_orig": len(yo),
            "new": new, "new_converged": conv,
            "old_v": strat(Xo[:N_OLD_V], yo[:N_OLD_V], OLD_ITER), "old_full": strat(Xo, yo, OLD_ITER),
            "spikes_train": float(Xt.sum(1).mean()), "spikes_test": float(Xn.sum(1).mean()),
            "spikes_test_orig": float(Xo.sum(1).mean())}


def main():
    jobs = [l.strip() for f in [JOBS] + [j for j in JOBS_EXTRA if j.exists()] for l in open(f) if l.strip()]
    missing = [d for d in jobs if not (Path(d) / "test_spike_counts_pf.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} networks have no re-extracted counts, e.g. {missing[:3]}")
    res = Parallel(n_jobs=16)(delayed(fixed_threads(score, 1))(d) for d in jobs)
    res.sort(key=lambda r: (r["family"], r["setting"], r["seed"]))
    json.dump({"probe": f"logreg L2 C=1 raw counts max_iter={MAX_ITER}; fit 10,000 training images, full test set",
               "old_v": f"stratified 70/30 (random_state 0) of the first {N_OLD_V} original test images, max_iter {OLD_ITER}",
               "old_full": f"stratified 70/30 (random_state 0) of all original test images, max_iter {OLD_ITER}",
               "runs": res}, open(f"{OUT}.json", "w"), indent=1)

    groups = defaultdict(list)
    for r in res:
        groups[(r["family"], r["setting"])].append(r)
    ms = lambda v: f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
    fams = [lab for lab, _ in FAMILIES] + ["other"]
    nc = [r["run"] for r in res if not r["new_converged"]]
    L = ["# Re-extracted networks scored with the single linear probe", "",
         "Generated by `scripts/score/reextract_probefix_score.py` (definitions in its docstring). One row per setting,",
         "seeds pooled (mean ± s.d.; one value if n = 1). **new** = single probe (10,000 training images -> full",
         "test set, re-extracted counts). **old_v** = retired voltage-STDP probe (70/30 of the first 5,000",
         "original test images); **old_full** = 70/30 of all original test images. EMNIST-Letters and N-MNIST old",
         "values come from class-sorted test passes. Trained-vs-disabled pairing (Delta) is not computed here.", "",
         f"{len(res)} networks; fits not converged at max_iter {MAX_ITER}: {len(nc)}" + (f" ({', '.join(nc)})" if nc else ""), ""]
    for fam in fams:
        keys = [k for k in groups if k[0] == fam]
        if not keys:
            continue
        L += [f"## {fam}", "", "| setting | dataset | STDP images | STDP | n | new | old_v | old_full | spikes/test image |",
              "|---|---|--:|---|--:|--:|--:|--:|--:|"]
        for k in sorted(keys, key=lambda k: k[1]):
            g = groups[k]; r0 = g[0]
            L.append(f"| `{k[1]}` | {r0['dataset']} | {r0['n_train_stdp']:,} | {'off' if r0['stdp_disabled'] else 'on'} | "
                     f"{len(g)} | {ms([r['new'] for r in g])} | {ms([r['old_v'] for r in g])} | "
                     f"{ms([r['old_full'] for r in g])} | {np.mean([r['spikes_test'] for r in g]):.1f} |")
        L.append("")
    L += ["## Per-network values", "", "| run | new | old_v | old_full | spikes/test image (new / orig.) | test images |",
          "|---|--:|--:|--:|--:|--:|"]
    for r in res:
        L.append(f"| `{r['run']}` | {r['new']:.2f} | {r['old_v']:.2f} | {r['old_full']:.2f} | "
                 f"{r['spikes_test']:.1f} / {r['spikes_test_orig']:.1f} | {r['n_test']:,} |")
    open(f"{OUT}.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}.md / .json: {len(res)} networks, {len(groups)} settings, not converged {len(nc)}")


if __name__ == "__main__":
    main()
