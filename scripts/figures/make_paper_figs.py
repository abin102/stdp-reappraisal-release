"""Generate the paper's data figures into $FIG_OUT_DIR (default figures/).
  fig1 pipeline schematic     (Methods)                 printed as Figure 1
  fig2 readout bias           (4.1)  post-hoc vs linear readout, 3 rules x 4 scaffolds   Figure 2
  fig3 delta-over-random      (4.7)  8 datasets x 3 rules                                Figure 7
  fig4 forgetting curves      (4.5)  accuracy vs training length                         Figure 5
  fig5 scaffold + fragility   (4.6)  decomposition + FC/conv/random                      Figure 6
  fig7 learned receptive fields (4.2)                                                    Figure 3
fig6 (Clopath curves, Figure A.1) is drawn by scripts/figures/clopath_curve_reproduction.py,
fig8 (protocol schematic, Figure 8) by scripts/figures/make_protocol_figure.py,
fig_section44_mechanism (Figure 4) by scripts/figures/make_section44_mechanism_figure.py and
fig_voltage_controls (Figure A.2) by scripts/figures/make_voltage_controls_figure.py.
Each figure is written as a vector PDF (what the manuscript includes) and a 600 dpi PNG.
Style, sizes and colours come from scripts/figures/paper_style.py (Elsevier artwork sizing).
Run: PYTHONPATH=. python scripts/figures/make_paper_figs.py
"""
import csv, json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import NullLocator
from scripts.figures import paper_style as ps
ps.apply()

OUT = os.environ.get("FIG_OUT_DIR", "figures"); os.makedirs(OUT, exist_ok=True)
C = ps.RULE_C
RULES = ["pair_exp", "triplet", "voltage"]
LBL = {"pair_exp": "pair", "triplet": "triplet", "voltage": "voltage"}
SCAF = ["full", "inh_only", "none", "thr_only"]
SCAF_T = {"full": "S_full", "inh_only": "S_inh", "none": "S_none", "thr_only": "S_thr"}
# Figures carry no overall title (the caption describes them); panels keep only the
# labels the caption or text refers to.

# ---------- FIG 2 readout ----------
# Source: results/scaffold_readout_table.csv (scripts/score/scaffold_readout_table.py):
# both readouts on identical frozen features, linear probe fit on TRAIN features --
# the matched protocol Section 4.1 reports. Replaces the pre-2026-08 test-split
# experiments/runs/readout_comparison.csv.
rows = list(csv.DictReader(open("results/scaffold_readout_table.csv")))
# The `posthoc` column is Diehl & Cook's own rule (means over training images, means
# over assigned neurons), carried into that table from results/posthoc_both_conventions.csv
# (`posthoc_dc`) since 2026-10-05. Cross-checked here so a table rebuilt by older code
# cannot silently put the summed variant back into the figure.
_dc = {r["run"]: float(r["posthoc_dc"]) / 100 for r in
       csv.DictReader(open("results/posthoc_both_conventions.csv"))
       if r["group"] == "scaffold" and r["dataset"] == "mnist"}
_off = [r["run"] for r in rows if abs(float(r["posthoc"]) - _dc[r["run"]]) > 1e-9]
if _off:
    raise SystemExit(
        f"FIG 2: the post-hoc column of scaffold_readout_table.csv disagrees with the "
        f"Diehl-Cook values in posthoc_both_conventions.csv for {len(_off)} of {len(rows)} "
        f"cells, e.g. {_off[:3]} -- rebuild it with scripts/score/scaffold_readout_table.py")
def agg(rule, sc, col):
    v = [float(r[col])*100 for r in rows if r["run"].startswith(f"{rule}_S_{sc}_")]
    return (np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0)) if v else (np.nan, 0)
fig, ax = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.25), sharey=True, layout="constrained")
x = np.arange(4); w = 0.38
for k, rule in enumerate(RULES):
    ph = [agg(rule, s, "posthoc") for s in SCAF]; lr = [agg(rule, s, "linear") for s in SCAF]
    ax[k].bar(x-w/2, [m[0] for m in ph], w, yerr=[m[1] for m in ph], color=ps.GREY,
              label="post-hoc readout", error_kw=ps.ERR)
    ax[k].bar(x+w/2, [m[0] for m in lr], w, yerr=[m[1] for m in lr], color=C[rule],
              label="linear readout", error_kw=ps.ERR)
    ax[k].axhline(11.35, ls=(0, (1, 1.5)), color=ps.INK, lw=0.6)
    ax[k].set_xticks(x, [SCAF_T[s] for s in SCAF]); ax[k].tick_params(axis="x", length=0)
    # the panel's rule is the only label kept: the caption does not say which panel is which
    ax[k].set_title(ps.RULE_L[rule])
    ax[k].legend(loc="upper right")
ax[0].set_ylabel("test accuracy (%)"); ax[0].set_ylim(0, 100)
ps.save(fig, f"{OUT}/fig2_readout"); plt.close(fig)

# ---------- FIG 3 delta-over-random ----------
# Five seeds, single linear probe (fit on 10,000 training images, full test set): the probe-fix
# matrix, results/probefix_matrix_n5.json (scripts/score/probefix_aggregate.py). Replaced the ten-seed
# 70/30 test-split values (core_matrix, medical_matrix, voltage_b3_n3, emnist_full26_n10) on 2026-10-20.
_pf = json.load(open("results/probefix_matrix_n5.json"))["cells"]
DS = [("mnist","MNIST"),("nmnist","N-MNIST"),("emnist_letters","EMNIST"),("fmnist","FMNIST"),
      ("organamnist","OrganA"),("bloodmnist","Blood"),("tissuemnist","Tissue"),("pathmnist","Path")]
def delta(rule, ds):
    c = _pf[f"{rule}|{ds}"]; assert c["n"] == 5, (rule, ds)
    return c["delta_linear"][0], c["delta_linear"][1]
fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.35), layout="constrained")
xm = np.arange(len(DS)); w = 0.27
for k, rule in enumerate(RULES):
    mu=[]; sd=[]
    for ds,_ in DS:
        d,s = delta(rule, ds); mu.append(d); sd.append(s)
    ax.bar(xm+(k-1)*w, mu, w, yerr=sd, color=C[rule], label=ps.RULE_L[rule], error_kw=ps.ERR)
ps.zero_line(ax)
ax.set_xticks(xm, [l for _,l in DS]); ax.tick_params(axis="x", length=0)
# the caption's argument contrasts the four standard and four medical (MedMNIST) datasets
ps.guide_line(ax, 3.5)
for xg, lab in ((1.5, "standard"), (5.5, "medical (MedMNIST)")):
    ax.text(xg, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            color="#555555")
ax.set_ylabel("Δ-over-random (points)")
ax.legend(loc="lower right")
ps.save(fig, f"{OUT}/fig3_delta_over_random"); plt.close(fig)

# ---------- FIG 4 forgetting ----------
# Values come from results/forgetting_curve.json (3 seeds per point, one config per
# rule, voltage at the corrected tc_theta_decay=1e4) rather than being typed in here.
# The previous hard-coded sweep was the superseded pair: a 5k cell at tc=1e7 against a
# 60k cell at tc=1e6. See results/voltage_audit_2026-09.md.
# A single panel is printed at Elsevier's single-column width (90 mm).
# 2026-10-21: single linear probe (fit on 10,000 training images, full test set), every point from
# spike counts recorded the same way after training: results/section45_probefix_mnist.json
# (scripts/score/section45_probefix_mnist.py). Replaced forgetting_curve.json (70/30 test split).
_fc = [c for c in json.load(open("results/section45_probefix_mnist.json"))["curves"] if c["config"] == "default"]
sweep = {r: [[c["n_train"] // 1000, *c["single_probe"]] for c in _fc if c["rule"] == r] for r in RULES}
assert all(len(v) == 4 and all(p[3] == 5 for p in v) for v in sweep.values()), sweep
fig, ax = plt.subplots(figsize=(ps.SINGLE_W, 2.2), layout="constrained")
# pair- and triplet-STDP lie within a point of each other up to 20k: triplet is drawn
# first with large hollow squares, pair on top with small filled circles that sit inside
# them, so both stay visible. Error bars (s.d.) are mostly smaller than the markers.
STY = {"triplet":  dict(ms=4.6, mfc="none", mew=0.9, zorder=2),
       "voltage":  dict(ms=4.2, mfc="none", mew=0.9, zorder=2),
       "pair_exp": dict(ms=2.6, zorder=3)}
for rule in ("triplet", "voltage", "pair_exp"):
    pts = sweep[rule]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; es=[p[2] for p in pts]
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=C[rule], elinewidth=0.6, capthick=0.6,
                capsize=1.2, zorder=STY[rule]["zorder"])
    ax.plot(xs, ys, "-", marker=ps.RULE_M[rule], color=C[rule], label=ps.RULE_L[rule], **STY[rule])
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticks([5,10,20,60], ["5k","10k","20k","60k"])
ax.set_xlabel("training images (log scale)"); ax.set_ylabel("linear-readout accuracy (%)")
ax.set_ylim(0, 100)
h, l = ax.get_legend_handles_labels(); order = [l.index(ps.RULE_L[r]) for r in RULES]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="center right", bbox_to_anchor=(1.0, 0.55))
ps.save(fig, f"{OUT}/fig4_forgetting"); plt.close(fig)

# ---------- FIG 5 scaffold decomposition + architectural fragility ----------
# (a) all three rules from the scaffold matrix (results/scaffold_readout_table.csv,
#     train-fit linear probe, 5 seeds) -- the same networks as Figure 2. The earlier
#     version plotted pair/triplet from the superseded test-split table and voltage
#     from a hard-coded "tc-fix" variant that appears nowhere else in the paper.
# (b) trained fully connected: pair/triplet = scaffold matrix S_full (train-fit);
#     voltage = the tuned configuration's ceiling (Section 4.3). Convolutional values
#     from the conv result sheets. Random initialisation: pair/triplet = the matched
#     60k random networks (results/matched_random_60k.json, train-fit); voltage = the
#     AdEx-B3 frozen control (results/frozen_ctl_results.md).
FIG5_OUT = os.environ.get("FIG5_OUT", f"{OUT}/fig5_scaffold_fragility")
_mr = json.load(open("results/matched_random_60k.json"))["train_fit"]["random60k"]
RAND60 = float(np.mean(_mr))
# 2026-10-13: panel (b) (architectural fragility) removed. It plotted superseded voltage-STDP
# values (24.4 / 72.5), was not cited in the text, and its point is made by the architecture
# row of Table 3. The figure is now the scaffold decomposition alone.
fig, ax0 = plt.subplots(figsize=(ps.FULL_W * 0.72, 2.35), layout="constrained")
xs = np.arange(4); w = 0.27
for k, rule in enumerate(RULES):
    mu=[agg(rule,s,"linear")[0] for s in SCAF]; sd=[agg(rule,s,"linear")[1] for s in SCAF]
    ax0.bar(xs+(k-1)*w, mu, w, yerr=sd, color=C[rule], label=ps.RULE_L[rule], error_kw=ps.ERR)
ax0.axhline(RAND60, ls="--", color=ps.GREY, lw=0.8)
ax0.text(0.99, RAND60-1.5, f"pair/triplet, STDP disabled ({RAND60:.1f}%)", color="#6E6E6E",
         ha="right", va="top", transform=ax0.get_yaxis_transform())
ax0.set_xticks(xs, [SCAF_T[s] for s in SCAF]); ax0.tick_params(axis="x", length=0)
ax0.set_ylim(0, 100)
ax0.set_ylabel("accuracy, train-fit probe (%)")
ax0.legend(loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.02))
ps.save(fig, FIG5_OUT); plt.close(fig)

# ---------- FIG 1 pipeline schematic ----------
# Drawn at its printed width (the full text width), so the font sizes below are the
# sizes the reader sees. Two rows: encoding -> network on top, readout -> class
# underneath. Box widths follow their text, so the layout adapts to the font. All text
# is one size (FS_LABEL), and the two scaffold mechanisms use the schematic accents
# rather than rule colours. Box, line and arrow styling is shared with Figure 8.
W1, H1 = ps.FULL_W, 2.50
fig = plt.figure(figsize=(W1, H1)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W1); ax.set_ylim(0, H1); ax.axis("off")
FS1 = ps.FS_LABEL
INH_C, THR_C = ps.ACC1, ps.ACC2
rend = fig.canvas.get_renderer()
def tsize(txt, fs, **kw):                      # text extent in inches
    t = ax.text(0, 0, txt, fontsize=fs, ha="center", va="center", **kw)
    bb = t.get_window_extent(renderer=rend); t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi
PAD, LW, HEAD = 0.0, 0.8, 7         # box padding, one line width for everything, arrowhead size
def box(xl, yc, txt, h=None, w=None):
    """Draw a rounded box (the shape and padding of Figure 8's boxes); return its outer
    edges, so arrows can start and end exactly on the visible border."""
    tw, th = tsize(txt, FS1); w = w or tw + 0.20; h = h or th + 0.15
    ax.add_patch(FancyBboxPatch((xl, yc - h/2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#EEF2F5", ec=ps.INK, lw=LW, zorder=1, clip_on=False))
    ax.text(xl + w/2, yc, txt, ha="center", va="center", fontsize=FS1, color=ps.INK, zorder=3)
    return xl - PAD, xl + w + PAD, yc - h/2 - PAD, yc + h/2 + PAD
def arrow(x1, y1, x2, y2, color=ps.INK, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=HEAD,
                                 lw=LW, color=color, shrinkA=0, shrinkB=0, zorder=4, clip_on=False, **kw))
def hgap(txt): return tsize(txt, FS1)[0] + 0.20
MUTED = "#555555"
Y1, Y2 = 1.52, 0.30
l1 = "Poisson\nencode"; l2 = "$W$ (784×400)\nfully connected"
b1 = box(0.08, Y1, "Image\n28×28\n(grayscale)")
x = b1[1] + hgap(l1); arrow(b1[1], Y1, x, Y1)
ax.text((b1[1]+x)/2, Y1+0.07, l1, ha="center", va="bottom", fontsize=FS1, color=MUTED)
b2 = box(x + PAD, Y1, "784 spike\ntrains\n350×1 ms", h=b1[3]-b1[2]-2*PAD)
x = b2[1] + hgap(l2); arrow(b2[1], Y1, x, Y1)
ax.text((b2[1]+x)/2, Y1+0.07, l2, ha="center", va="bottom", fontsize=FS1, color=MUTED)
b3 = box(x + PAD, Y1, "400 LIF / AdEx\nneurons\n+ STDP", h=(b1[3]-b1[2]-2*PAD)+0.14)
xc3 = (b3[0]+b3[1])/2
# lateral inhibition: double-headed arrow ending on the neuron box's top border
arrow(xc3, b3[3]+0.26, xc3, b3[3], color=INH_C, style="<|-|>")
ax.text(b3[1], b3[3]+0.29, "lateral inhibition (WTA)\nbetween neurons", ha="right", va="bottom",
        fontsize=FS1, color=INH_C)
# adaptive threshold: a self-loop that leaves and re-enters the box at its border
arrow(b3[0]+0.18, b3[2], b3[0], b3[2]+0.18, color=THR_C, connectionstyle="arc3,rad=-1.9")
ax.text(b3[0]-0.19, b3[2]-0.07, "adaptive threshold $\\theta_j$\nwithin each neuron", ha="right", va="top",
        fontsize=FS1, color=THR_C)
# row 2: readout under the network, class to its left
tw4, _ = tsize("Linear readout\n(vs post-hoc)", FS1)
b4 = box(b3[1] - PAD - (tw4 + 0.20), Y2, "Linear readout\n(vs post-hoc)")
arrow(xc3, b3[2], xc3, b4[3])
ax.text(xc3+0.07, (b3[2]+b4[3])/2, "spike\ncounts", ha="left", va="center", fontsize=FS1, color=MUTED)
tw5, _ = tsize("Class", FS1)
b5 = box(b4[0] - 0.48 - (tw5 + 0.28), Y2, "Class", w=tw5+0.28, h=b4[3]-b4[2]-2*PAD)
arrow(b4[0], Y2, b5[1], Y2)
# centre the drawing horizontally on the page width, then save at that exact width
x_lo, x_hi = b1[0] - 0.02, b3[1] + 0.02
ax.set_xlim((x_lo + x_hi)/2 - W1/2, (x_lo + x_hi)/2 + W1/2)
ps.save(fig, f"{OUT}/fig1_pipeline"); plt.close(fig)

# ---------- FIG 7 learned receptive fields ----------
# This is the one figure that needs the trained weights themselves rather than a
# summary table, and weights_final.npz is gitignored (see README "What is not in
# this repository"). Missing weights therefore skip the figure with an explanation
# instead of crashing: previously load_w returned None, the guard below chose
# indices safely, and W[:, nj] then raised "NoneType is not subscriptable".
def load_w(path):
    try: return np.load(path)["w"]                      # (784, 400)
    except Exception: return None
rng = np.random.RandomState(1)
rowspec = [("random\ninitialisation", None), ("pair-STDP", "pair_exp"),
           ("triplet-STDP", "triplet"), ("voltage-STDP", "voltage")]
ncol = 12
# The networks of the scaffold matrix (S_full, seed 0). Voltage-STDP: the tc_theta_decay = 1e4 network
# that Table A.3 and Figure 6 use, not the superseded 1e6 run voltage_S_full_N400_seed0.
RF_RUN = {"pair_exp": "pair_exp_S_full_N400_seed0", "triplet": "triplet_S_full_N400_seed0",
          "voltage": "voltage_S_full_N400_seed0__b_fix_tc1e4_60k"}
_needed = [f"experiments/runs/{RF_RUN[r]}/weights_final.npz"
           for _, r in rowspec if r is not None]
_missing = [p for p in _needed if load_w(p) is None]
if _missing:
    print(f"SKIPPING fig7_receptive_fields — needs trained weights that are not "
          f"in this repository ({len(_missing)} of {len(_needed)} missing, e.g. "
          f"{_missing[0]}).\n  Reproduce them with experiments/scripts/run_single.py "
          f"(see README), or omit this figure; every other figure is regenerable "
          f"from the committed tables in results/.")
else:
  # square tiles across the full text width, an 18 mm column on the left for row labels;
  # the grid is inset by one frame width (E) so the outer frames are not clipped
  LEFT, GAP, RGAP, E = 18 * ps.MM, 1.0 * ps.MM, 1.6 * ps.MM, 0.4 / 72
  tile = (ps.FULL_W - LEFT - (ncol - 1) * GAP - E) / ncol
  H7 = len(rowspec) * tile + (len(rowspec) - 1) * RGAP + 2 * E
  fig = plt.figure(figsize=(ps.FULL_W, H7))
  ax = np.empty((len(rowspec), ncol), dtype=object)
  for i in range(len(rowspec)):
      for j in range(ncol):
          ax[i, j] = fig.add_axes([(LEFT + j*(tile+GAP)) / ps.FULL_W,
                                   1 - (E + (i+1)*tile + i*RGAP) / H7, tile / ps.FULL_W, tile / H7])
  for i, (lab, rule) in enumerate(rowspec):
    if rule is None:
        W = rng.uniform(0, 0.3, size=(784, 400))        # the init distribution
    else:
        W = load_w(f"experiments/runs/{RF_RUN[rule]}/weights_final.npz")
    # diverse sample: random 12 from the 120 most-structured filters (avoids over-picking one digit)
    top = np.argsort(W.var(0))[::-1][:120]
    idx = sorted(rng.choice(top, ncol, replace=False))
    for j, nj in enumerate(idx):
        f = W[:, nj].reshape(28, 28)
        a = ax[i, j]
        a.imshow(f, cmap="gray_r", vmin=f.min(), vmax=f.max(), interpolation="none")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#555555")
    ax[i, 0].set_ylabel(lab, fontsize=ps.FS, rotation=0, ha="right", va="center", labelpad=4)
  ps.save(fig, f"{OUT}/fig7_receptive_fields"); plt.close(fig)

print("wrote:", ", ".join(sorted(os.listdir(OUT))))
