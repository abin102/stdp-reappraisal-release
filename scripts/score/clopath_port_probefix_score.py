#!/usr/bin/env python3
"""Score the exact reimplementation of Clopath's neuron (experiments/runs/clopath_port/{explore_k0,
grid10k,thetatar10k}) with the paper's single linear probe, from the counts re-extracted by
scripts/extract/clopath_port_reextract.py, next to the retired probe of scripts/score/clopath_port_score.py.

Single probe: multinomial logistic regression, L2, C = 1, raw counts, max_iter=100000, fitted on
train_spike_counts_pf.npz (frozen pass over the first 10,000 images of the run's presentation order)
and scored on test_spike_counts_pf.npz (full 10,000-image MNIST test set). One fit per network.
Retired probe (old): scripts/score/clopath_port_score.py "strat5k" on the original test_spike_counts.npz
(stratified 70/30 split, random_state 0, of the first 5,000 test images, max_iter 2000).
Gap = trained - STDP-disabled network of the same seed, input gain g and inhibition k_inh
(thetatar10k has no STDP-disabled networks of its own and uses grid10k's, which are identical:
same init_weights(seed), g, k_inh, plasticity off). Mean and sample s.d. over seeds.
Check of the re-extraction: spikes per test image, re-extracted vs original pass, per network
(different Poisson draws and starting state, so close but not equal).

Outputs: results/clopath_port_probefix_scores.json, results/clopath_port_probefix_scores.md
Usage (repo root): python scripts/score/clopath_port_probefix_score.py [--allow-missing]
"""
import json, os, sys, warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scripts.score._blas import fixed_threads  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

warnings.filterwarnings("ignore")
JOBS = Path("results/run_lists/jobs_port.txt")
OUT = Path("results/clopath_port_probefix_scores")
MAX_ITER = 100000


def fit_net(spec, Xt, yt, Xn, yn, Xo, yo):
    Xt, Xn, Xo = (X.astype(np.float64) for X in (Xt, Xn, Xo))
    clf = LogisticRegression(max_iter=MAX_ITER, C=1.0).fit(Xt, yt)
    a, c = Xo[:5000], yo[:5000]
    i, j = train_test_split(np.arange(len(c)), test_size=0.3, random_state=0, stratify=c)
    old = 100 * LogisticRegression(max_iter=100000, C=1.0).fit(a[i], c[i]).score(a[j], c[j])
    return {**spec, "new": 100 * clf.score(Xn, yn), "converged": bool(clf.n_iter_.max() < MAX_ITER),
            "old": old, "spikes_test": float(Xn.sum(1).mean()), "spikes_test_orig": float(Xo.sum(1).mean()),
            "spikes_train": float(Xt.sum(1).mean()), "n_train": len(yt), "n_test": len(yn)}


def dir_scores(d):
    meta = json.load(open(d / "meta.json"))
    tr = np.load(d / "train_spike_counts_pf.npz", allow_pickle=True)
    te = np.load(d / "test_spike_counts_pf.npz", allow_pickle=True)
    og = np.load(d / "test_spike_counts.npz", allow_pickle=True)
    Ct, Cn, Co = tr["counts"], te["counts"], og["counts"]
    assert len(meta["specs"]) == len(Ct) == len(Cn) == len(Co), d
    assert json.loads(str(te["specs"])) == meta["specs"], d
    specs = [{**s, "seed": int(meta["seed"]), "group": d.parent.name, "dir": str(d)} for s in meta["specs"]]
    return Parallel(n_jobs=16)(delayed(fixed_threads(fit_net, 1))(s, Ct[b], tr["labels"], Cn[b], te["labels"], Co[b], og["labels"])
                               for b, s in enumerate(specs))


def main():
    allow_missing = "--allow-missing" in sys.argv
    dirs = [Path(l.strip()) for l in open(JOBS) if l.strip()]
    missing = [d for d in dirs if not (d / "test_spike_counts_pf.npz").exists()]
    if missing and not allow_missing:
        raise SystemExit(f"not re-extracted yet: {[str(m) for m in missing]}")
    dirs = [d for d in dirs if d not in missing]
    res = [r for d in dirs for r in dir_scores(d)]

    rand = {(r["group"] if r["group"] != "thetatar10k" else "grid10k", r["seed"], r["g"], r["k_inh"]): r
            for r in res if r["kind"] == "random"}
    for r in res:
        if r["kind"] == "trained":
            q = rand[("grid10k" if r["group"] == "thetatar10k" else r["group"], r["seed"], r["g"], r["k_inh"])]
            r["gap_new"], r["gap_old"] = r["new"] - q["new"], r["old"] - q["old"]
    json.dump({"probe": f"logreg L2 C=1 raw counts max_iter={MAX_ITER}; fit 10,000 training images, full test set",
               "old": "strat5k of scripts/score/clopath_port_score.py on the original test pass", "networks": res},
              open(f"{OUT}.json", "w"), indent=1)

    ms = lambda v: f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
    cells = defaultdict(list)
    for r in res:
        cells[(r["group"], r.get("theta_tar", 60.0) if r["kind"] == "trained" else None, r["cell"], r["amp"], r["kind"])].append(r)
    nc = [f"{r['dir']}#{r['cell']}" for r in res if not r["converged"]]
    ratio = [r["spikes_test"] / r["spikes_test_orig"] for r in res if r["spikes_test_orig"] > 1]
    L = ["# Exact reimplementation of Clopath's neuron: single linear probe", "",
         "Generated by `scripts/score/clopath_port_probefix_score.py` (definitions in its docstring). new = single probe;",
         "old = retired probe (70/30 of the first 5,000 original test images). Gap = trained - STDP disabled",
         "(same seed, g, k_inh). Mean ± s.d. over seeds. u_ref^2 = homeostatic reference (60 mV^2 = Clopath's).", "",
         (f"NOT YET SCORED (re-extraction unfinished): {', '.join(str(m) for m in missing)}" if missing else "All 10 run directories scored."), "",
         f"{len(res)} networks; fits not converged: {len(nc)}" + (f" ({', '.join(nc)})" if nc else ""),
         f"Re-extraction check: spikes per test image re-extracted / original, networks with > 1 spike per image: "
         f"median {np.median(ratio):.3f}, range {min(ratio):.3f}-{max(ratio):.3f} (n = {len(ratio)}).", "",
         "| runs | u_ref^2 (mV^2) | cell | g | k_inh | amplitudes | network | seeds | new | old | gap new | gap old | spikes/test image (new / orig.) |",
         "|---|--:|---|--:|--:|---|---|--:|--:|--:|--:|--:|--:|"]
    for k in sorted(cells, key=lambda k: (k[0], k[1] or 0, k[4], k[2], str(k[3]))):
        g = cells[k]; r0 = g[0]; tr = r0["kind"] == "trained"
        L.append(f"| {k[0]} | {k[1] if tr else '--'} | {k[2]} | {r0['g']} | {r0['k_inh']} | {k[3] or '--'} | "
                 f"{'trained' if tr else 'STDP disabled'} | {len(g)} | {ms([r['new'] for r in g])} | {ms([r['old'] for r in g])} | "
                 f"{ms([r['gap_new'] for r in g]) if tr else '--'} | {ms([r['gap_old'] for r in g]) if tr else '--'} | "
                 f"{np.mean([r['spikes_test'] for r in g]):.1f} / {np.mean([r['spikes_test_orig'] for r in g]):.1f} |")
    open(f"{OUT}.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}.md / .json: {len(res)} networks, {len(cells)} cells, not converged {len(nc)}")


if __name__ == "__main__":
    main()
