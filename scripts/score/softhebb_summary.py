"""Score the SoftHebb runs (Section 3.9, Section S18). Release copy of experiments/softhebb_pilot/score_softhebb.py;
only the input and output folders differ. results/softhebb/pilot_README_2026-11-16.md holds the rules: stage 1 fixed
2026-11-11, stage 2 fixed 2026-11-16, each before its runs.

Gain = trained minus random test accuracy at the last epoch, paired by seed; input gap = trained minus pixels.
  Reproduction bar (stage 1): CIFAR-10, whole training split, seed 0, trained: at least 78.3% (the paper's 80.3% minus
  2 points). If it is missed, every reading is marked "port not validated".
  Stage 1 reading, kept as registered (seeds 0-2, 10,000 training images): "adds" if the gain is positive in 3 of 3
  seeds, "does not add" if in 0 of 3, otherwise "mixed".
  Stage 2 readings (seeds 0-4, 10,000 training images, now with TissueMNIST): training: "adds" if trained beats random
  in at least 4 of 5 seeds, "does not add" if in at most 1 of 5, otherwise "mixed"; input: "beats its input" if trained
  beats pixels in at least 4 of 5 seeds, "does not beat its input" if in at most 1 of 5, otherwise "mixed".
  A reading needs all runs it uses.
Reported without a reading: CIFAR-10 seeds 0-2, random against pixels, the whole splits, means and SDs over seeds.

  PYTHONPATH=. python scripts/score/softhebb_summary.py [--runs DIR] [--out DIR]
  (defaults: --runs results/softhebb/runs, --out results/softhebb; writes pilot_summary.md and pilot_summary.json)
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(REPO, "results", "softhebb", "runs")
MED = ("organamnist", "bloodmnist", "pathmnist", "tissuemnist")
WHOLE = ("organamnist", "bloodmnist", "pathmnist")
S3, S5, CIFAR_SEEDS, BAR, PAPER = (0, 1, 2), (0, 1, 2, 3, 4), (0, 1, 2), 78.3, "80.31 ± 0.14 (4 runs)"


def acc(ds, mode, n, seed):
    p = os.path.join(RUNS, f"{ds}__{mode}__n{n}__seed{seed}", "result.json")
    return json.load(open(p))["test_acc_last"] if os.path.exists(p) else None


def fmt(v):
    return ", ".join("–" if x is None else f"{x:.2f}" for x in v)


def ms(v):
    """mean ± SD over the seeds present (SD with n - 1), and how many there are when some are missing."""
    x = [y for y in v if y is not None]
    if not x:
        return "waiting"
    s = f"{np.mean(x):.2f}" + (f" ± {np.std(x, ddof=1):.2f}" if len(x) > 1 else "")
    return s if len(x) == len(v) else f"{s} ({len(x)} of {len(v)})"


def wins(a, b):
    """seeds where a > b, over the seeds where both exist; None if any is missing."""
    if None in a or None in b:
        return None
    return int((np.array(a) > np.array(b)).sum())


def read5(k, yes, no):
    return "waiting" if k is None else yes if k >= 4 else no if k <= 1 else "mixed"


def read3(k):
    return "waiting" if k is None else "adds" if k == 3 else "does not add" if k == 0 else "mixed"


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS, help="folder of <dataset>__<mode>__n<N>__seed<S>/result.json")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "softhebb"))
    a = ap.parse_args()
    RUNS, out = a.runs, a.out
    if not os.path.isdir(RUNS):
        sys.exit(f"{RUNS} not found: it should hold the run folders written by scripts/train/softhebb_run.py "
                 "(shipped in results/softhebb/runs/).")
    L = ["# SoftHebb pilot: trained against untrained weights and against the pixels", "",
         "Test accuracy (%) at the last readout epoch, full official test split. Mean ± SD over seeds.", ""]
    J = {}

    rep, rep_rand = acc("cifar10", "trained", 0, 0), acc("cifar10", "random", 0, 0)
    ok = rep is not None and round(rep, 2) >= BAR
    tag = "" if ok else (" (port not validated)" if rep is not None else " (reproduction check waiting)")
    c = {m: [acc("cifar10", m, 0, s) for s in CIFAR_SEEDS] for m in ("trained", "random", "pixels")}
    L += ["## CIFAR-10 (whole training split): reproduction check", "",
          f"- Stage-1 bar, seed 0 trained: {'waiting' if rep is None else f'{rep:.2f}%'} (bar {BAR}%): "
          f"{'waiting' if rep is None else 'met' if ok else 'MISSED: port not validated'}. Paper: {PAPER}.", "",
          "| | Seeds 0-2 | Mean ± SD |", "|---|---|---|"]
    for m in ("trained", "random", "pixels"):
        L.append(f"| {m} | {fmt(c[m])} | {ms(c[m])} |")
    g = None if None in c["trained"] or None in c["random"] else (np.array(c["trained"]) - np.array(c["random"])).tolist()
    L += ["", f"Gain per seed (trained minus random): {'waiting' if g is None else ', '.join(f'{x:+.2f}' for x in g)}", ""]
    J["cifar10"] = dict(bar=BAR, seed0_trained=rep, seed0_random=rep_rand, met=ok, paper=PAPER, **c, gain=g)

    # --- 10,000 training images: the readings ---
    L += ["## Medical sets, 10,000 training images (the STDP paper's matrix size): readings", "",
          "| Dataset | Trained | Random | Pixels | Gain per seed (trained minus random) | Trained > random | "
          "Training, 5 seeds | Training, stage 1 (seeds 0-2) | Trained > pixels | Input, 5 seeds | Random > pixels |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    J["n10000"] = {}
    for ds in MED:
        t, r, p = ([acc(ds, m, 10000, s) for s in S5] for m in ("trained", "random", "pixels"))
        k_tr, k_tp, k_rp = wins(t, r), wins(t, p), wins(r, p)
        k3 = wins(t[:3], r[:3])
        rd5 = read5(k_tr, "adds", "does not add") + ("" if k_tr is None else tag)
        rd3 = ("n/a (not in stage 1)" if ds == "tissuemnist" else read3(k3) + ("" if k3 is None else tag))
        ri = read5(k_tp, "beats its input", "does not beat its input") + ("" if k_tp is None else tag)
        gs = "waiting" if k_tr is None else ", ".join(f"{a - b:+.2f}" for a, b in zip(t, r))
        n = lambda k: "waiting" if k is None else f"{k} of 5"
        L.append(f"| {ds} | {ms(t)} | {ms(r)} | {ms(p)} | {gs} | {n(k_tr)} | {rd5} | {rd3} | {n(k_tp)} | {ri} | {n(k_rp)} |")
        J["n10000"][ds] = dict(trained=t, random=r, pixels=p, trained_gt_random=k_tr, trained_gt_pixels=k_tp,
                               random_gt_pixels=k_rp, reading_training_5=rd5, reading_training_stage1=rd3,
                               reading_input_5=ri)
    L.append("")

    # --- whole splits: reported, no reading ---
    L += ["## Medical sets, whole training split (reported, no reading)", "",
          "| Dataset | Trained | Random | Pixels | Gain per seed (trained minus random) | Trained > random | Trained > pixels |",
          "|---|---|---|---|---|---|---|"]
    J["whole"] = {}
    for ds in WHOLE:
        t, r, p = ([acc(ds, m, 0, s) for s in S5] for m in ("trained", "random", "pixels"))
        pairs = [(a, b) for a, b in zip(t, r) if a is not None and b is not None]
        gs = ", ".join(f"{a - b:+.2f}" for a, b in pairs) or "waiting"
        cnt = lambda x, y: f"{sum(a > b for a, b in zip(x, y) if a is not None and b is not None)} of " \
                           f"{sum(a is not None and b is not None for a, b in zip(x, y))}"
        L.append(f"| {ds} | {ms(t)} | {ms(r)} | {ms(p)} | {gs} | {cnt(t, r)} | {cnt(t, p)} |")
        J["whole"][ds] = dict(trained=t, random=r, pixels=p)
    L.append("")

    open(os.path.join(out, "pilot_summary.md"), "w").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(out, "pilot_summary.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
