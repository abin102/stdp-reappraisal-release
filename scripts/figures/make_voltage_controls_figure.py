#!/usr/bin/env python3
"""Figure A.2 (Table 3 controls): voltage-STDP trained against random, one row per control.

A dot plot on one accuracy axis. Each row is one fully connected voltage-STDP setting of
Table 3 (MNIST, seeds 0-4). It shows the trained networks (filled triangles), the networks
with STDP disabled and their weights left at the random initialisation (open circles), and
the mean of each set (tick). A faint line joins each trained network to the random network
of the same seed and settings, so a reader can check that every trained point lies left of
its own random one. The right-hand column prints the mean of the per-seed gaps
(trained - random, in points).

Readout: the single linear probe of Section 3.9 (logistic regression, L2, C = 1, raw counts,
fit on the spike counts of 10,000 training images, scored on the full test set); spread is
the sample SD (ddof=1).

Sources (no accuracy is typed in here):
  results/probefix_voltage_thresholds_n5.json  reference (trained vs frozen), thresholds of
                                               the convolutional network, spike plateau
  results/reextract_probefix_scores.json       60,000-image tuned network and its matched
                                               random network; homeostatic scaling

Not plotted: the convolutional-architecture row of Table 3 (a different network and
readout, scripts/score/conv_arch_logreg_n5.py) and the exact reimplementation of Clopath's
neuron (Appendix J).

The x axis runs from 20 to 80 % (the script stops if a value falls outside).

Outputs: $FIG_OUT_DIR (default figures/)/fig_voltage_controls.{pdf,png}
         and fig_voltage_controls_data.json (the plotted per-seed values and their sources).
Run from the repo root:
  PYTHONPATH=. python scripts/figures/make_voltage_controls_figure.py
"""
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from scripts.figures import paper_style as ps  # noqa: E402

ps.apply()
warnings.filterwarnings("ignore")

OUT_DIR = os.environ.get("FIG_OUT_DIR", "figures")
STEM = "fig_voltage_controls"
SEEDS = (0, 1, 2, 3, 4)

# trained = voltage-STDP colour of every paper figure (Okabe-Ito bluish green, marker "^").
# random = a darker step of the paper's neutral grey: the dataviz validator gives
# ps.GREY (#9A9A9A) against this green a protan Delta E of 5.6 (FAIL) and #6E6E6E 8.4 (PASS),
# and contrast on white 2.8:1 against 5.1:1. Shape (filled triangle vs open circle) is the
# second cue.
C_TR = ps.RULE_C["voltage"]
C_RD = "#6E6E6E"
C_LINK = "#CFCFCF"
MUTED = "#555555"


def load(path):
    if not os.path.isfile(path):
        sys.exit(f"missing {path}: this figure is built from it")
    return json.load(open(path))


# ---------------------------------------------------------------- checks
def ms(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1)


def check(ok, what):
    if not ok:
        sys.exit(f"cross-check failed: {what}")


# 2026-10-20: every row now uses the single linear probe (fit on 10,000 training images, full
# test set), per seed, from results/probefix_voltage_thresholds_n5.json (fresh probe-fix runs,
# seeds 0-4) and results/reextract_probefix_scores.json (re-extracted saved networks).
_PF = load("results/probefix_voltage_thresholds_n5.json")["runs"]
_RE = {r["run"]: r["new"] for r in load("results/reextract_probefix_scores.json")["runs"]}


def pf(label):
    v = sorted((r["seed"], r["linear"]) for r in _PF if r["label"] == label)
    check([s for s, _ in v] == list(range(5)), f"{label}: seeds {[s for s, _ in v]}")
    return [x for _, x in v]


V = "voltage_S_full_N400_seed{}_"
rd5 = pf("frozen")
tr60 = [_RE[V.format(s) + ("mnist_vhz60000" if s in (1, 2) else "vb3_trained60k")] for s in range(5)]   # seeds 0-4 (author, 10-22)
rd60 = [_RE[V.format(s) + "vb3_frozen60k"] for s in range(5)]
rows = [
    ("Table 3", "reference: tuned network, 10,000 images (5 seeds)", pf("trained"), rd5,
     "probefix_voltage_thresholds_n5.json: trained vs frozen"),
    ("Table 3", "tuned network, 60,000 images (5 seeds)", tr60, rd60,
     "reextract_probefix_scores.json: vb3_trained60k (seeds 1-2: mnist_vhz60000) vs vb3_frozen60k"),
    ("Table 3", "thresholds of the convolutional network (5 seeds)", pf("thltd_m65_thltp_m52"), rd5,
     "probefix_voltage_thresholds_n5.json: thltd_m65_thltp_m52 vs frozen"),
    ("Table 3", "rule computed with Clopath's spike plateau (5 seeds)", pf("plateau"), rd5,
     "probefix_voltage_thresholds_n5.json: plateau vs frozen"),
    ("Table 3", "homeostatic scaling instead of $L_1$ renormalisation (5 seeds)",
     [_RE[V.format(s) + "fixref_B_clopath_ref2400_ap0.002_h0.9"] for s in range(5)],
     [_RE[V.format(s) + "fixref_frozen_h0.9"] for s in range(5)],
     "reextract_probefix_scores.json: fixref_B_clopath_ref2400_ap0.002_h0.9 vs fixref_frozen_h0.9"),
]
for g, lab, tr, rd, src in rows:
    check(len(tr) == len(rd) and len(tr) in (len(SEEDS), 5, 10), f"{lab}: unexpected seed count")

# ---------------------------------------------------------------- printout
print("Plotted values (accuracy %, mean ± sample SD over seeds; gap = trained − random "
      "of the same seed, in points)")
print(f"{'row':42s} {'trained':>13s} {'random':>13s} {'gap':>14s}   per-seed trained / random")
records = []
for g, lab, tr, rd, src in rows:
    gap = np.subtract(tr, rd)
    (mt, st), (mr, sr), (mg, sg) = ms(tr), ms(rd), ms(gap)
    print(f"{lab:42s} {mt:5.1f} ± {st:4.1f}  {mr:5.1f} ± {sr:4.1f}  {mg:+6.1f} ± {sg:4.1f}   "
          f"{', '.join(f'{v:.2f}' for v in tr)} / {', '.join(f'{v:.2f}' for v in rd)}"
          f"{'' if np.all(gap < 0) else '   <- a trained seed is not below its random seed'}")
    records.append(dict(group=g, label=lab, trained=list(map(float, tr)), random=list(map(float, rd)),
                        gap=list(map(float, gap)), trained_mean=mt, trained_sd=st, random_mean=mr,
                        random_sd=sr, gap_mean=mg, gap_sd=sg, source=src))
n_left = sum(int(np.all(np.subtract(r["trained"], r["random"]) < 0)) for r in records)
print(f"rows where every trained seed lies below the random network of its own seed: "
      f"{n_left} of {len(records)}")

# ---------------------------------------------------------------- figure
groups = []
for g, *_ in rows:
    if g not in groups:
        groups.append(g)
# y layout, top to bottom: a heading slot per group, then its rows
Y, heads, y = [], [], 0.0
for g in groups:
    heads.append((g, y)); y += 0.85
    for r in records:
        if r["group"] == g:
            Y.append(y); y += 1.0
    y += 0.25
y_max = y - 0.25

def offsets(n):                      # vertical spread of the seeds within a row
    return np.array([-0.2, 0.0, 0.2]) if n == 3 else np.linspace(-0.3, 0.3, n)
# x range: a dot plot has no bar length, so the axis need not start at 0; starting at 20
# (below the lowest plotted value) gives each point a third more width than 0-80, which
# separates the overlapping markers of the closest rows. Stop if a value falls outside.
X_LO, X_HI = 20, 80
allv = np.concatenate([np.r_[r["trained"], r["random"]] for r in records])
check(X_LO < allv.min() and allv.max() < X_HI, f"values {allv.min():.1f}-{allv.max():.1f} outside the x range")
fig, ax = plt.subplots(figsize=(ps.FULL_W, 0.205 * y_max + 0.62), layout="constrained")
# gap column: numbers and header right-aligned 26 pt to the right of the axes, so the
# one-line header can reach left over the empty heading row instead of sitting on row 1
gap_tr = ax.get_yaxis_transform() + ScaledTranslation(26 / 72, 0, fig.dpi_scale_trans)
for r, yr in zip(records, Y):
    OFF = offsets(len(r["trained"]))
    for k in range(len(r["trained"])):
        ax.plot([r["trained"][k], r["random"][k]], [yr + OFF[k]] * 2, color=C_LINK, lw=0.5,
                zorder=1, solid_capstyle="butt")
    ax.plot(r["random"], yr + OFF, ls="none", marker="o", ms=3.3, mfc="white", mec=C_RD,
            mew=0.8, zorder=3)
    ax.plot(r["trained"], yr + OFF, ls="none", marker="^", ms=3.8, mfc=C_TR, mec="white",
            mew=0.35, zorder=3)
    for m, c in ((r["random_mean"], C_RD), (r["trained_mean"], C_TR)):
        ax.plot([m, m], [yr - 0.36, yr + 0.36], color=c, lw=1.3, zorder=2, solid_capstyle="butt")
    ax.text(1.0, yr, f"{r['gap_mean']:+.1f}".replace("-", "−"), transform=gap_tr,
            ha="right", va="center", color=ps.INK, fontsize=ps.FS)
for g, yh in heads:
    pass                      # (2026-10-13) no group heading: a single group, Table 3
for (_, yh) in heads[1:]:
    ps.guide_line(ax, yh - 0.55, axis="y")
ax.text(1.0, heads[0][1] + 0.1, "Δ (points)", transform=gap_tr, ha="right",
        va="center", color=MUTED, fontsize=ps.FS)

ax.set_yticks(Y, [r["label"] for r in records])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_ylim(y_max - 0.35, -0.6)
ax.set_xlim(X_LO, X_HI)
ax.set_xticks(range(X_LO, X_HI + 1, 10))
ax.set_xlabel("accuracy, linear probe (%)")

handles = [Line2D([], [], ls="none", marker="^", ms=3.8, mfc=C_TR, mec="white", mew=0.35),
           Line2D([], [], ls="none", marker="o", ms=3.3, mfc="white", mec=C_RD, mew=0.8),
           Line2D([], [], ls="none", marker="|", ms=7, mew=1.3, color=ps.INK)]
# a figure legend above the axes: constrained layout reserves its full width, which an
# axes legend wider than its axes does not get (it was clipped at the right edge)
fig.legend(handles, ["voltage-STDP, trained", "same network, STDP disabled", "mean over seeds"],
           loc="outside upper center", ncol=3)

os.makedirs(OUT_DIR, exist_ok=True)
ps.save(fig, f"{OUT_DIR}/{STEM}")
plt.close(fig)
json.dump(dict(readout="single linear probe: logistic regression, L2, C = 1, raw counts, fit on 10,000 "
                       "training images, full test set; mean and sample SD over seeds 0-4",
               rows=records), open(f"{OUT_DIR}/{STEM}_data.json", "w"), indent=1)
print(f"wrote {OUT_DIR}/{STEM}.pdf, .png and {STEM}_data.json")
