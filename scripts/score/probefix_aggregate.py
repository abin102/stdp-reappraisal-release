#!/usr/bin/env python3
"""Aggregate the probe-fix matrix (scripts/train/probefix_matrix_n5.sh) into results tables.

Protocol (identical for every run, 3 rules x 8 datasets x seeds 0-4 x trained/random):
  * STDP on 10,000 training images (random = same network with STDP disabled);
  * frozen pass over those 10,000 training images  -> train_spike_counts.npz;
  * pass over the FULL test set                    -> test_spike_counts.npz.
Readouts, both fitted on the training-image spike counts and scored on the full test set:
  * linear probe : multinomial logistic regression, L2, C = 1 (fixed in advance), raw counts;
  * post-hoc     : Diehl & Cook (2015) rule. Each neuron is labelled with the class of its
                   highest mean spike count over the training images; a test image goes to
                   the class with the largest summed count of its labelled neurons divided
                   by the number of neurons carrying that class. Neurons that never fire
                   on the training images get no label.
Delta-over-random = trained minus random accuracy, per seed; mean +- s.d. over seeds.

Outputs
  results/probefix_matrix_n5.json  per-run accuracies + per-cell summaries
  results/probefix_matrix_n5.md    tables: (1) Delta-over-random with the linear probe,
                                   (2) old 70/30 ten-seed values vs new (only if
                                   the retired result files are present), (3) post-hoc vs
                                   linear on identical data, (4) run inventory.
Usage (repo root): python scripts/score/probefix_aggregate.py [--allow-missing]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from scripts.score._blas import fixed_threads
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
R = Path("experiments/runs")
RULES = [("pair_exp", "pair-STDP"), ("triplet", "triplet-STDP"), ("voltage", "voltage-STDP")]
DSETS = [("mnist", "MNIST"), ("nmnist", "N-MNIST"), ("emnist_letters", "EMNIST-Letters"),
         ("fmnist", "Fashion-MNIST"), ("organamnist", "OrganAMNIST"), ("bloodmnist", "BloodMNIST"),
         ("tissuemnist", "TissueMNIST"), ("pathmnist", "PathMNIST")]
SEEDS = range(5)
MAX_ITER = 100000   # raised from 2000 on 2026-10-19: 25 voltage-STDP fits (high spike counts) stopped at 2000


def run_dir(rule, ds, seed, mode):
    tag = "" if ds == "mnist" else f"_{ds}"
    return R / f"{rule}_S_full_N400_seed{seed}{tag}_probefix_{mode}"


def posthoc_dc(Xtr, ytr, Xte, yte, n_cls):
    means = np.stack([Xtr[ytr == c].mean(0) if (ytr == c).any() else np.zeros(Xtr.shape[1])
                      for c in range(n_cls)], 1)
    asg = means.argmax(1)
    asg[Xtr.sum(0) == 0] = -1
    v = np.zeros((len(yte), n_cls))
    for c in range(n_cls):
        m = asg == c
        if m.any():
            v[:, c] = Xte[:, m].sum(1) / m.sum()
    return 100.0 * float((v.argmax(1) == yte).mean())


def score(rule, ds, seed, mode):
    d = run_dir(rule, ds, seed, mode)
    ok = (d / "completed.flag").exists() and (d / "train_spike_counts.npz").exists() \
        and (d / "test_spike_counts.npz").exists()
    if not ok:
        return dict(rule=rule, dataset=ds, seed=seed, mode=mode, run=d.name, complete=False)
    zt, ze = np.load(d / "train_spike_counts.npz"), np.load(d / "test_spike_counts.npz")
    Xtr, ytr = zt["train_spike_counts"].astype(np.float64), zt["train_labels"]
    Xte, yte = ze["test_spike_counts"].astype(np.float64), ze["test_labels"]
    n_cls = int(max(ytr.max(), yte.max())) + 1
    clf = LogisticRegression(max_iter=MAX_ITER, C=1.0).fit(Xtr, ytr)
    lin = 100.0 * clf.score(Xte, yte)
    return dict(rule=rule, dataset=ds, seed=seed, mode=mode, run=d.name, complete=True,
                n_train=int(len(ytr)), n_test=int(len(yte)), n_classes_test=int(len(np.unique(yte))),
                linear=float(lin), converged=bool(clf.n_iter_.max() < MAX_ITER),
                posthoc=posthoc_dc(Xtr, ytr, Xte, yte, n_cls),
                mean_spikes_test=float(Xte.sum(1).mean()))


def ms(a):
    a = np.asarray(a, float)
    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0) if len(a) else (np.nan, np.nan)


# The previous ten-seed values (retired 70/30 probe) are not part of the release. Table 2 of the
# .md output (previous vs new) is built only when all four files are present, and skipped otherwise.
OLD_FILES = ["results/emnist_full26_n10.json", "results/core_matrix.json", "results/medical_matrix.json",
             "results/voltage_b3_n3.json"]
HAVE_OLD = all(Path(p).is_file() for p in OLD_FILES)


def old_delta(rule, ds):
    """The paper's previous ten-seed values (70/30 split of the first <=10,000 test images)."""
    if not HAVE_OLD:
        return np.nan, np.nan
    if ds == "emnist_letters":
        e = json.load(open("results/emnist_full26_n10.json"))["rules"][rule]
        return e["delta_mean"], e["delta_sd"]
    core = json.load(open("results/core_matrix.json"))
    med = json.load(open("results/medical_matrix.json"))
    vb3 = json.load(open("results/voltage_b3_n3.json"))
    if rule in core and ds in core[rule]:
        return core[rule][ds]["delta"], core[rule][ds].get("delta_sd", np.nan)
    if rule in med and ds in med[rule]:
        return med[rule][ds]["delta"], med[rule][ds].get("delta_sd", np.nan)
    if rule == "voltage" and ds in vb3:
        return vb3[ds]["delta_mean"], vb3[ds]["delta_sd"]
    return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true",
                    help="build the tables from the runs finished so far")
    args = ap.parse_args()
    jobs = [(r, d, s, m) for r, _ in RULES for d, _ in DSETS for s in SEEDS for m in ("trained", "frozen")]
    res = Parallel(n_jobs=12)(delayed(fixed_threads(score, 1))(*j) for j in jobs)
    missing = [x["run"] for x in res if not x["complete"]]
    if missing and not args.allow_missing:
        sys.exit(f"ERROR: {len(missing)} of {len(res)} runs incomplete, e.g. {missing[:3]}. "
                 "Wait for scripts/train/probefix_matrix_n5.sh, or pass --allow-missing.")
    by = {(x["rule"], x["dataset"], x["seed"], x["mode"]): x for x in res if x["complete"]}

    cells = {}
    for r, _ in RULES:
        for d, _ in DSETS:
            seeds = [s for s in SEEDS if (r, d, s, "trained") in by and (r, d, s, "frozen") in by]
            tl = [by[(r, d, s, "trained")]["linear"] for s in seeds]
            rl = [by[(r, d, s, "frozen")]["linear"] for s in seeds]
            tp = [by[(r, d, s, "trained")]["posthoc"] for s in seeds]
            rp = [by[(r, d, s, "frozen")]["posthoc"] for s in seeds]
            cells[f"{r}|{d}"] = dict(
                rule=r, dataset=d, n=len(seeds), seeds=seeds,
                n_test=by[(r, d, seeds[0], "trained")]["n_test"] if seeds else None,
                trained_linear=ms(tl), random_linear=ms(rl),
                delta_linear=ms(np.subtract(tl, rl)) if seeds else (np.nan, np.nan),
                same_sign=bool(len(seeds) > 0 and (np.all(np.subtract(tl, rl) > 0) or np.all(np.subtract(tl, rl) < 0))),
                trained_posthoc=ms(tp), random_posthoc=ms(rp),
                gap_trained=ms(np.subtract(tl, tp)) if seeds else (np.nan, np.nan),
                delta_old=old_delta(r, d))
    json.dump(dict(protocol=__doc__.split("Outputs")[0].strip(), runs=res, cells=cells),
              open("results/probefix_matrix_n5.json", "w"), indent=1)

    f = lambda t, p=1: "—" if t is None or np.isnan(t[0]) else f"{t[0]:.{p}f} ± {t[1]:.{p}f}"
    L = ["# Probe-fix matrix: one protocol for both readouts (5 seeds)", "",
         "Generated by `scripts/score/probefix_aggregate.py` from the runs of `scripts/train/probefix_matrix_n5.sh`.",
         "Every network: STDP on 10,000 training images (random = STDP disabled); frozen pass over",
         "those training images; pass over the **full** test set. Linear probe (logistic regression,",
         "C = 1) and post-hoc labels (Diehl & Cook rule) are both fitted on the training-image spike",
         "counts and scored on the full test set. Values: mean ± s.d. over seeds 0–4 (n per cell shown).",
         f"Logistic regression: L2, C = 1, raw counts, max_iter = {MAX_ITER}; fits not converged: "
         f"{sum(1 for x in res if x.get('complete') and not x.get('converged', True))} of {sum(1 for x in res if x.get('complete'))}.", "",
         "Notes. (1) Test images are presented in a fixed random order (seed + 3000) since 2026-10-19 04:38:",
         "EMNIST-Letters and N-MNIST store their test sets sorted by class, which corrupted test spike counts",
         "(the adaptive threshold adapts during evaluation); the 29 TissueMNIST runs from before the fix used",
         "stored order, which for TissueMNIST is already random. The previous EMNIST and N-MNIST values came",
         "from sorted passes. (2) OrganAMNIST and PathMNIST have a genuine train/test shift: raw-pixel logistic",
         "regression scores 58.3 train->test vs 74.7 within-train (OrganA) and 40.3 vs 28.3 (PathMNIST, greyscale;",
         "test class mix differs, majority class 17.2%). (3) 'same sign' = Delta has the same sign in all seeds.", ""]
    if missing:
        L += [f"**Incomplete: {len(missing)} of {len(res)} runs missing** — tables use finished runs only.", ""]
    L += ["## Table 1. Δ-over-random with the linear probe (accuracy, %)", "",
          "| rule | dataset | n | test images | trained | random | Δ (trained − random) | same sign |",
          "|---|---|--:|--:|--:|--:|--:|:-:|"]
    for r, rl in RULES:
        for d, dl in DSETS:
            c = cells[f"{r}|{d}"]
            L.append(f"| {rl} | {dl} | {c['n']} | {c['n_test'] or '—'} | {f(c['trained_linear'])} | "
                     f"{f(c['random_linear'])} | {f(c['delta_linear'])} | {'yes' if c['same_sign'] else 'no'} |")
    L += ["", "## Table 2. Δ-over-random: previous paper values vs this protocol", ""]
    if not HAVE_OLD:
        L += ["Skipped: the previous ten-seed values (retired 70/30 probe; " + ", ".join(OLD_FILES) +
              ") are not part of this release."]
    else:
        L += ["Previous: ten seeds, probe fitted on 70% of the (first ≤10,000) test images and scored on",
              "the other 30%. New: five seeds, probe fitted on 10,000 training images, full test set.", "",
              "| rule | dataset | previous Δ (10 seeds, 70/30) | new Δ (5 seeds, train-fit) | change |",
              "|---|---|--:|--:|--:|"]
        for r, rl in RULES:
            for d, dl in DSETS:
                c = cells[f"{r}|{d}"]; o, nw = c["delta_old"], c["delta_linear"]
                ch = "—" if np.isnan(nw[0]) or np.isnan(o[0]) else f"{nw[0] - o[0]:+.1f}"
                L.append(f"| {rl} | {dl} | {f(o)} | {f(nw)} | {ch} |")
    L += ["", "## Table 3. Post-hoc vs linear readout on identical data (trained networks)", "",
          "Both readouts fitted on the same training-image spike counts, scored on the same test set.", "",
          "| rule | dataset | post-hoc | linear probe | gap (linear − post-hoc) |", "|---|---|--:|--:|--:|"]
    for r, rl in RULES:
        for d, dl in DSETS:
            c = cells[f"{r}|{d}"]
            L.append(f"| {rl} | {dl} | {f(c['trained_posthoc'])} | {f(c['trained_linear'])} | {f(c['gap_trained'])} |")
    L += ["", "## Table 4. Run inventory", "",
          f"Complete runs: {len(res) - len(missing)} of {len(res)}."]
    if missing:
        L += ["", "Missing:", ""] + [f"- `{m}`" for m in missing]
    open("results/probefix_matrix_n5.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
