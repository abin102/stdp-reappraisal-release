"""Conv trained-vs-random for pair-STDP at n=5.

Section 4.3 establishes for VOLTAGE that trained conv features are no more
linearly separable than random conv features (results/conv_arch_logreg_n5.json,
scripts/score/conv_arch_logreg_n5.py, n=5).
This runs the same control for pair-STDP, whose five trained conv seeds exist as
weights at the repository root (conv2l_stdp_ckpt_pairgrade_seed{0..4}.pt; see RUN_MAP.md).

Readout matches scripts/extract/conv_svm_readout.py: logistic regression and a linear SVM on the
11008-d pooled conv features, fit on Xtr and scored on Xte.
"""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from threadpoolctl import threadpool_limits
threadpool_limits(1)   # single-threaded BLAS: probe fits reproduce bit for bit (scripts/score/_blas.py)

R = Path("results")


def score(tag):
    f = R / f"conv_feats_{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    lr = LogisticRegression(max_iter=100000, C=1.0, n_jobs=4).fit(Xtr, ytr).score(Xte, yte) * 100
    sv = LinearSVC(max_iter=100000).fit(Xtr, ytr).score(Xte, yte) * 100
    return lr, sv


def main():
    rows = []
    for s in range(5):
        t = score(f"pairN5_trained_s{s}")
        r = score(f"pairN5_random_s{s}")
        if t is None or r is None:
            print(f"  seed {s}: incomplete (trained={t is not None}, random={r is not None})")
            continue
        rows.append((s, t[0], r[0], t[1], r[1]))
        print(f"  seed {s}: logreg trained {t[0]:6.2f}  random {r[0]:6.2f}  "
              f"delta {t[0]-r[0]:+6.2f} | svm trained {t[1]:6.2f}  random {r[1]:6.2f}  "
              f"delta {t[1]-r[1]:+6.2f}", flush=True)
    if not rows:
        print("no complete seeds yet"); return

    a = np.array(rows, dtype=float)
    n = len(rows)
    d_lr, d_sv = a[:, 1] - a[:, 2], a[:, 3] - a[:, 4]
    sd = lambda v: float(np.std(v, ddof=1)) if n > 1 else 0.0
    out = {"logreg": {"trained": a[:, 1].mean(), "trained_sd": sd(a[:, 1]),
                      "random": a[:, 2].mean(), "random_sd": sd(a[:, 2]),
                      "delta": d_lr.mean(), "delta_sd": sd(d_lr), "n": n},
           "linear_svm": {"trained": a[:, 3].mean(), "trained_sd": sd(a[:, 3]),
                          "random": a[:, 4].mean(), "random_sd": sd(a[:, 4]),
                          "delta": d_sv.mean(), "delta_sd": sd(d_sv), "n": n}}
    print(f"\n  PAIR conv, n={n}")
    print(f"    logreg : trained {out['logreg']['trained']:.2f}±{out['logreg']['trained_sd']:.2f}  "
          f"random {out['logreg']['random']:.2f}±{out['logreg']['random_sd']:.2f}  "
          f"delta {out['logreg']['delta']:+.2f}±{out['logreg']['delta_sd']:.2f}")
    print(f"    svm    : trained {out['linear_svm']['trained']:.2f}±{out['linear_svm']['trained_sd']:.2f}  "
          f"random {out['linear_svm']['random']:.2f}±{out['linear_svm']['random_sd']:.2f}  "
          f"delta {out['linear_svm']['delta']:+.2f}±{out['linear_svm']['delta_sd']:.2f}")
    if (R / "conv_arch_logreg_n5.json").exists():
        v = json.load(open(R / "conv_arch_logreg_n5.json"))
        print("\n  for comparison, VOLTAGE (results/conv_arch_logreg_n5.json, n=5):")
        print(f"    logreg : delta {v['gap_mean_sd'][0]:+.2f}±{v['gap_mean_sd'][1]:.2f}")
    json.dump(out, open(R / "conv_pair_n5.json", "w"), indent=2)
    print(f"\n  wrote {R/'conv_pair_n5.json'}")


if __name__ == "__main__":
    main()
