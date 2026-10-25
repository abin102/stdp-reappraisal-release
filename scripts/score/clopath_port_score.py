#!/usr/bin/env python3
"""Score the Clopath-port MNIST grid (results/voltage_clopath_neuron_port_PREREG.md).

Primary probe (decisions): the voltage-STDP probe of results/voltage_matched_random_60k.md:
  LogisticRegression(max_iter=100000, C=1.0) on spike counts of the first 5,000 test images,
  stratified 70/30 split (train_test_split, random_state=0, stratify=labels).
Secondary (reported only): perm70, the same classifier on a 70/30 RandomState(0)
  permutation of all presented test images (the Section 4.7 readout).
Gap = trained - STDP-off (same seed, same g and k_inh), in points; mean and sample SD
(ddof=1) over seeds. A cell beats random if mean gap > 0 and mean gap > 2 x SD.

Usage: python scripts/score/clopath_port_score.py --runs experiments/runs/clopath_port/grid10k \
          --out results/voltage_clopath_neuron_port [--probe strat5k|perm70]
Writes <out>.json and <out>_tables.md (the narrative .md is written by hand around them).
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scripts.score._blas import fixed_threads  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)


def probes(X, y):
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X = X.astype(np.float64)
    out = {}
    a, b = X[:5000], y[:5000]
    tr, te = train_test_split(np.arange(len(b)), test_size=0.3, random_state=0, stratify=b)
    out["strat5k"] = 100 * LogisticRegression(max_iter=100000, C=1.0).fit(a[tr], b[tr]).score(a[te], b[te])
    n = len(y); o = np.random.RandomState(0).permutation(n); k = int(0.7 * n)
    out["perm70"] = 100 * LogisticRegression(max_iter=100000, C=1.0).fit(X[o[:k]], y[o[:k]]).score(X[o[k:]], y[o[k:]])
    out["spikes_per_test_image"] = float(X.sum(1).mean())
    out["active_fraction_test"] = float((X > 0).mean())
    out["silent_test_images"] = float((X.sum(1) == 0).mean())
    out["neurons_ever_active_test"] = int((X.sum(0) > 0).sum())
    return out


def ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory with seed*/ run folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", default="strat5k", choices=["strat5k", "perm70"],
                    help="probe used for the decision column")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    seeds = sorted(glob.glob(os.path.join(args.runs, "seed*")))
    if not seeds:   # without this an empty table and "beats random: False" would be written
        sys.exit(f"ERROR: no seed*/ run folders in {args.runs}; nothing written.\n"
                 "The runs are made by scripts/train/clopath_port_grid.sh (experiments/runs/ is not in\n"
                 "the release copies). The scored grid is already in\n"
                 "results/voltage_clopath_neuron_port.{json,md} and _tables.md.")
    nets = []
    for sd in seeds:
        if not os.path.isfile(os.path.join(sd, "completed.flag")):
            sys.exit(f"not completed: {sd}")
        meta = json.load(open(os.path.join(sd, "meta.json")))
        z = np.load(os.path.join(sd, "test_spike_counts.npz"))
        specs = json.loads(str(z["specs"]))
        assert specs == meta["specs"], f"spec mismatch in {sd}"
        W = np.load(os.path.join(sd, "weights.npz"))
        st = np.load(os.path.join(sd, "train_stats.npz"))
        ntr = st["spikes"].shape[0]; last = slice(max(ntr - 1000, 0), ntr)
        for j, s in enumerate(specs):
            rec = dict(seed=meta["seed"], run=sd, idx=j, **s,
                       spikes_per_train_image_last1000=float(st["spikes"][last, j].mean()),
                       spikes_per_train_image_first100=float(st["spikes"][:100, j].mean()),
                       active_fraction_train_last1000=float(st["active"][last, j].mean()))
            if s["kind"] == "trained":
                w = W["W_final"][j]
                rec.update(w_mean=float(w.mean()), w_sd=float(w.std()), frac_w0=float((w <= 0).mean()),
                           frac_wmax=float((w >= 3.0).mean()),
                           ltp_per_image_last1000=float(st["ltp"][last, j].mean()),
                           ltd_per_image_last1000=float(st["ltd"][last, j].mean()))
                wn = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
                cs = wn.T @ wn
                rec["mean_pairwise_cosine"] = float((cs.sum() - np.trace(cs)) / (cs.shape[0] * (cs.shape[0] - 1)))
            nets.append((rec, z["counts"][j], z["labels"]))
    print(f"{len(nets)} networks from {len(seeds)} seed folders; fitting probes ...", flush=True)
    res = Parallel(n_jobs=args.jobs)(delayed(fixed_threads(probes, 1))(c, y) for _, c, y in nets)
    rows = []
    for (rec, _, _), r in zip(nets, res):
        rec.update(r); rows.append(rec)
    # paired comparison per (cell, amplitude)
    cells = sorted({r["cell"] for r in rows})
    amps = sorted({r["amp"] for r in rows if r["kind"] == "trained"})
    summ = []
    for c in cells:
        rnd = {r["seed"]: r for r in rows if r["cell"] == c and r["kind"] == "random"}
        for a in amps:
            trn = {r["seed"]: r for r in rows if r["cell"] == c and r["amp"] == a and r["kind"] == "trained"}
            ss = sorted(set(trn) & set(rnd))
            gaps = [trn[s][args.probe] - rnd[s][args.probe] for s in ss]
            m, sd, n = ms(gaps)
            beats = bool(m is not None and n >= 2 and m > 0 and (m > 2 * sd if sd and np.isfinite(sd) else m > 0))
            row = dict(cell=c, amp=a, g=trn[ss[0]]["g"], k_inh=trn[ss[0]]["k_inh"], seeds=ss, probe=args.probe,
                       trained=[trn[s][args.probe] for s in ss], random=[rnd[s][args.probe] for s in ss],
                       gaps=gaps, gap_mean=m, gap_sd=sd, n=n, beats_random=beats)
            for key in ("strat5k", "perm70", "spikes_per_test_image", "active_fraction_test",
                        "spikes_per_train_image_last1000", "silent_test_images"):
                row[f"trained_{key}"] = ms([trn[s][key] for s in ss])
                row[f"random_{key}"] = ms([rnd[s][key] for s in ss])
            for key in ("w_mean", "w_sd", "frac_w0", "frac_wmax", "mean_pairwise_cosine",
                        "ltp_per_image_last1000", "ltd_per_image_last1000"):
                row[f"trained_{key}"] = ms([trn[s][key] for s in ss])
            summ.append(row)
    out = dict(runs=args.runs, probe=args.probe, networks=rows, cells=summ,
               any_cell_beats_random=any(r["beats_random"] for r in summ))
    json.dump(out, open(args.out + ".json", "w"), indent=2, default=float)

    def f(t, d=1):
        m, sd, n = t
        return "–" if m is None else (f"{m:.{d}f} ± {sd:.{d}f}" if n > 1 else f"{m:.{d}f}")
    L = [f"Probe for decisions: **{args.probe}**. Mean ± sample SD over seeds "
         f"{sorted({r['seed'] for r in rows})}; gap paired by seed, in points.", "",
         "| cell | g | k_inh (mV) | amplitudes | trained | STDP off | gap | beats random? | "
         "trained perm70 | STDP-off perm70 | spikes/test img trained | STDP off |",
         "|---|--:|--:|---|--:|--:|--:|---|--:|--:|--:|--:|"]
    for r in summ:
        L.append(f"| {r['cell']} | {r['g']:g} | {r['k_inh']:g} | {r['amp']} | {f(ms(r['trained']))} | "
                 f"{f(ms(r['random']))} | {f((r['gap_mean'], r['gap_sd'], r['n']))} | "
                 f"{'yes' if r['beats_random'] else 'no'} | {f(r['trained_perm70'])} | {f(r['random_perm70'])} | "
                 f"{f(r['trained_spikes_per_test_image'], 0)} | {f(r['random_spikes_per_test_image'], 0)} |")
    L += ["", "Weights and activity of the trained networks (end of training):", "",
          "| cell | amplitudes | spikes/train img, last 1,000 | first 100 | w mean | w SD | at 0 | at 3 | "
          "mean pairwise cosine | potentiation/img | depression/img |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in summ:
        L.append(f"| {r['cell']} | {r['amp']} | {f(r['trained_spikes_per_train_image_last1000'], 0)} | "
                 f"{f(ms([n['spikes_per_train_image_first100'] for n in rows if n['cell'] == r['cell'] and n['amp'] == r['amp']]), 0)} | "
                 f"{f(r['trained_w_mean'], 3)} | {f(r['trained_w_sd'], 3)} | {f(r['trained_frac_w0'], 3)} | "
                 f"{f(r['trained_frac_wmax'], 3)} | {f(r['trained_mean_pairwise_cosine'], 3)} | "
                 f"{f(r['trained_ltp_per_image_last1000'], 3)} | {f(r['trained_ltd_per_image_last1000'], 3)} |")
    open(args.out + "_tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:len(summ) + 4]))
    print(f"any cell beats random: {out['any_cell_beats_random']}")


if __name__ == "__main__":
    main()
