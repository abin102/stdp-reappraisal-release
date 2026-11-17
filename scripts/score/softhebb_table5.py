"""Values of Table 5 (SoftHebb, Section 3.9) from results/softhebb/runs/*/result.json: mean and SD (n - 1) over seeds,
gain = trained minus random per seed then averaged, counts of seeds. Writes results/softhebb/table5_values.json.
Release copy of experiments/softhebb_pilot/table5_values.py; only the input and output folders differ.
  PYTHONPATH=. python scripts/score/softhebb_table5.py [--runs DIR] [--out DIR]"""
import argparse, json, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ap = argparse.ArgumentParser()
_ap.add_argument("--runs", default=os.path.join(REPO, "results", "softhebb", "runs"))
_ap.add_argument("--out", default=os.path.join(REPO, "results", "softhebb"))
_a = _ap.parse_args()
RUNS, OUT = _a.runs, _a.out
if not os.path.isdir(RUNS):
    sys.exit(f"{RUNS} not found: it should hold the run folders written by scripts/train/softhebb_run.py "
             "(shipped in results/softhebb/runs/).")
A = lambda ds, m, n, s: json.load(open(os.path.join(RUNS, f"{ds}__{m}__n{n}__seed{s}", "result.json")))["test_acc_last"]
ROWS = [("cifar10", 0, range(3)), ("organamnist", 10000, range(5)), ("bloodmnist", 10000, range(5)),
        ("pathmnist", 10000, range(5)), ("tissuemnist", 10000, range(5)), ("organamnist", 0, range(5)),
        ("bloodmnist", 0, range(5)), ("pathmnist", 0, range(5))]
out = []
for ds, n, seeds in ROWS:
    v = {m: np.array([A(ds, m, n, s) for s in seeds]) for m in ("trained", "random", "pixels")}
    n_img = json.load(open(os.path.join(RUNS, f"{ds}__trained__n{n}__seed0", "result.json")))["n_train"]
    g, tp, rp = v["trained"] - v["random"], v["trained"] - v["pixels"], v["random"] - v["pixels"]
    row = dict(dataset=ds, n_train=n_img, seeds=len(v["trained"]),
               **{f"{m}_mean": float(x.mean()) for m, x in v.items()}, **{f"{m}_sd": float(x.std(ddof=1)) for m, x in v.items()},
               gain_mean=float(g.mean()), gain_min=float(g.min()), gain_max=float(g.max()), trained_gt_random=int((g > 0).sum()),
               trained_minus_pixels=float(tp.mean()), random_minus_pixels=float(rp.mean()),
               trained_gt_pixels=int((tp > 0).sum()), random_gt_pixels=int((rp > 0).sum()))
    out.append(row)
    print(f"{ds:12s} {n_img:6d} tr {row['trained_mean']:.2f}±{row['trained_sd']:.2f} ra {row['random_mean']:.2f}±{row['random_sd']:.2f} "
          f"px {row['pixels_mean']:.2f}±{row['pixels_sd']:.2f} | gain {row['gain_mean']:+.3f} [{row['gain_min']:+.2f},{row['gain_max']:+.2f}] "
          f"{row['trained_gt_random']}/{row['seeds']} | tr-px {row['trained_minus_pixels']:.2f} ra-px {row['random_minus_pixels']:.2f} "
          f"share of tr-px from random {100*row['random_minus_pixels']/row['trained_minus_pixels']:.0f}%")
json.dump(out, open(os.path.join(OUT, "table5_values.json"), "w"), indent=1)

# Markdown rows of Table 5 (one decimal; the values above are the source)
NAME = dict(cifar10="CIFAR-10", organamnist="OrganAMNIST", bloodmnist="BloodMNIST", pathmnist="PathMNIST", tissuemnist="TissueMNIST")
pm = lambda r, m: f"${r[m + '_mean']:.1f}\\pm{r[m + '_sd']:.1f}$"
sg = lambda x: ("+" if x >= 0 else "-") + f"{abs(x):.1f}"
for r in out:
    print(f"| {NAME[r['dataset']]} | ${r['n_train']:,}$ | {pm(r, 'trained')} | {pm(r, 'random')} | {pm(r, 'pixels')} | "
          f"${sg(r['gain_mean'])}$ | {r['trained_gt_random']}/{r['seeds']} |".replace(",", "{,}"))
