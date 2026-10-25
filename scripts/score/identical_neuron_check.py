#!/usr/bin/env python3
"""Evidence for the identical-neuron collapse (Section 4.1, item 2).

For the 20 pair/triplet MNIST scaffold cells without lateral inhibition
({pair_exp,triplet} x {S_none,S_thr_only} x seeds 0-4) this computes, from the
saved run artefacts, every number item 2 quotes:

  rf_cos_start / rf_cos_end   mean pairwise receptive-field cosine similarity of
                              the initial weights (rebuilt from the run's seed)
                              and of the final weights
  firing_rate_cv              std/mean of per-neuron mean test firing rate
  spread_across_neurons       median over test images (with any spike) of the
                              std/mean of spike counts across neurons -- 0 means
                              every neuron fires the same count on that image
  max_distinct_counts         over ALL 10k test images, the largest number of
                              distinct spike counts found among the 400 neurons
                              on any single image -- 2 means no image ever splits
                              the population more than two ways
  pct_images_one_count        share of all 10k test images on which all 400
                              neurons emit exactly the same count
  corr_count_ink              Pearson r, over the 10k test images, between the
                              population spike count and the image's summed
                              pixel intensity ("amount of ink")
  n_label_classes             how many classes the labeller's neuron labels span
  pct_images_to_one_class     share of test images the labeller sends to its
                              most frequent answer
  linear_all400               from results/scaffold_readout_table.csv
  posthoc                     from results/posthoc_both_conventions.csv
                              (`posthoc_dc` = the Diehl-Cook convention with
                              both divisions, as printed in Table A.3; the
                              `posthoc` column of scaffold_readout_table.csv
                              is the superseded convention)
  linear_count_only           linear readout given only the population count
  linear_ink_only             linear readout given only the image's summed
                              pixel intensity -- no network involved
  linear_ink_plus_count       both together

All low-dimensional readouts use the SAME 10,000 training images as the run's
own train_spike_counts.npz (train order rebuilt from the seed and verified by
label match), are evaluated on the full 10k MNIST test set, and use
StandardScaler + LogisticRegression(max_iter=100000, C=1.0).

Writes results/identical_neuron_check.csv and prints a summary.
"""
import copy, csv, sys, warnings
from pathlib import Path
import numpy as np
import torch, yaml

from threadpoolctl import threadpool_limits
threadpool_limits(1)   # single-threaded BLAS: probe fits reproduce bit for bit (scripts/score/_blas.py)
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
from models.fc_stdp_layer import FCStdpLayer

RUNS = ROOT / "experiments" / "runs"
CELLS = [f"{r}_{sc}_N400_seed{s}" for r in ("pair_exp", "triplet")
         for sc in ("S_none", "S_thr_only") for s in range(5)]


def rf_cos(W):
    W = np.asarray(W, dtype=np.float64)
    Wn = W / np.linalg.norm(W, axis=0, keepdims=True).clip(min=1e-12)
    S = Wn.T @ Wn
    return float(S[~np.eye(S.shape[0], dtype=bool)].mean())


def fit(Ftr, ytr, Fte, yte):
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=100000, C=1.0).fit(sc.transform(Ftr), ytr)
    return 100.0 * clf.score(sc.transform(Fte), yte)


def main():
    tr = datasets.MNIST(ROOT / "data", train=True, download=True)
    te = datasets.MNIST(ROOT / "data", train=False, download=True)
    ink_tr = (tr.data.numpy().reshape(len(tr), -1) / 255.0).sum(1)
    ink_te = (te.data.numpy().reshape(len(te), -1) / 255.0).sum(1)
    y_tr_all = tr.targets.numpy()
    table = {r["run"]: r for r in csv.DictReader(open(ROOT / "results" / "scaffold_readout_table.csv"))}
    posthoc_dc = {r["run"]: float(r["posthoc_dc"])
                  for r in csv.DictReader(open(ROOT / "results" / "posthoc_both_conventions.csv"))
                  if r["group"] == "scaffold" and r["dataset"] == "mnist"}
    absent = [c for c in CELLS if c not in posthoc_dc]
    if absent:
        sys.exit(f"ERROR: results/posthoc_both_conventions.csv has no Diehl-Cook post-hoc "
                 f"value for {len(absent)} of {len(CELLS)} cells, e.g. {absent[:3]} -- "
                 f"regenerate it with scripts/score/posthoc_both_conventions.py")

    rows = []
    for run in CELLS:
        d = RUNS / run
        if not d.is_dir():
            sys.exit(f"ERROR: missing cell {d}")
        cfg = yaml.safe_load(open(d / "config.yaml"))
        g = torch.Generator().manual_seed(int(cfg["experiment"]["seed"]) + 1000)
        idx = np.array(torch.randperm(len(tr), generator=g).tolist()[: int(cfg["training"]["n_train"])][:10000])
        trz, tez = np.load(d / "train_spike_counts.npz"), np.load(d / "test_spike_counts.npz")
        if not (trz["train_labels"] == y_tr_all[idx]).all():
            sys.exit(f"ERROR: {run}: rebuilt training order does not match saved train labels")
        Xte, yte = tez["test_spike_counts"].astype(np.float64), tez["test_labels"]
        c_tr, c_te = trz["train_spike_counts"].sum(1), Xte.sum(1)
        ytr = y_tr_all[idx]

        ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
        c0 = copy.deepcopy(ck["config"]); c0["experiment"]["device"] = "cpu"
        w0 = FCStdpLayer(c0, device=torch.device("cpu")).w.numpy()

        act = Xte.sum(1) > 0
        Xs = np.sort(Xte, axis=1)
        n_distinct = 1 + (np.diff(Xs, axis=1) != 0).sum(1)
        asg = tez["assignments"]
        votes = np.zeros((len(yte), 10))
        for c in range(10):
            mk = asg == c
            if mk.any():
                votes[:, c] = Xte[:, mk].sum(1)
        pred = votes.argmax(1)
        m = __import__("json").load(open(d / "metrics.json"))
        rows.append(dict(
            run=run,
            rf_cos_start=round(rf_cos(w0), 4),
            rf_cos_end=round(rf_cos(ck["w"]), 4),
            firing_rate_cv=round(float(m["firing_rate_cv"]), 4),
            spread_across_neurons=round(float(np.median(Xte[act].std(1) / Xte[act].mean(1))), 4),
            max_distinct_counts=int(n_distinct.max()),
            pct_images_one_count=round(100.0 * float((n_distinct == 1).mean()), 2),
            n_label_classes=int((np.bincount(asg[asg >= 0], minlength=10) > 0).sum()),
            pct_images_to_one_class=round(100.0 * np.bincount(pred, minlength=10).max() / len(pred), 2),
            images_with_spikes_pct=round(100.0 * act.mean(), 2),
            corr_count_ink=round(float(np.corrcoef(c_te, ink_te)[0, 1]), 4),
            posthoc=round(posthoc_dc[run], 2),
            linear_all400=round(100.0 * float(table[run]["linear"]), 2),
            linear_count_only=round(fit(c_tr[:, None], ytr, c_te[:, None], yte), 2),
            linear_ink_only=round(fit(ink_tr[idx, None], ytr, ink_te[:, None], yte), 2),
            linear_ink_plus_count=round(fit(np.c_[ink_tr[idx], c_tr], ytr, np.c_[ink_te, c_te], yte), 2),
        ))
        print(f"  {run:32s} done", flush=True)

    out = ROOT / "results" / "identical_neuron_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def rng(k):
        v = [r[k] for r in rows]; return f"{min(v)} - {max(v)}"
    print(f"\nwrote {out}  ({len(rows)} cells)")
    for k in ("n_label_classes", "pct_images_to_one_class", "rf_cos_start", "rf_cos_end", "firing_rate_cv", "spread_across_neurons",
              "max_distinct_counts", "pct_images_one_count", "corr_count_ink",
              "posthoc", "linear_all400", "linear_count_only", "linear_ink_only", "linear_ink_plus_count"):
        print(f"  {k:24s} {rng(k)}")
    gap = [abs(r["linear_all400"] - r["linear_count_only"]) for r in rows]
    beat = sum(r["linear_count_only"] <= r["linear_ink_only"] for r in rows)
    print(f"  |all400 - count_only|    max {max(gap):.2f} pp")
    print(f"  count_only <= ink_only   in {beat}/{len(rows)} cells")


if __name__ == "__main__":
    main()
