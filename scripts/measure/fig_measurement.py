"""Figure 3: the protocol on one network.
(a) MNIST, pair-STDP, 10,000 training images, 5 seeds: the untrained network and the
    trained network, each read by neuron labelling and by the linear probe; dashed line: the
    same probe on the encoded input.
(b, c) Rule gain (trained - untrained, paired by seed) on the eight datasets, measured with
    each readout, for pair- and triplet-STDP.
Sources: results/probefix_matrix_n5.json, results/input_baselines_probefix.json.
Run: python scripts/measure/fig_measurement.py  (writes figures/fig_measurement.pdf, .png)"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
import paper_style as ps
ps.apply()
OUT = os.environ.get("FIG_OUT_DIR", "figures")
d = json.load(open("results/probefix_matrix_n5.json"))
R = {(r["rule"], r["dataset"], r["seed"], r["mode"]): r for r in d["runs"]}
inp = json.load(open("results/input_baselines_probefix.json"))
S = range(5)
DS = ["mnist", "nmnist", "emnist_letters", "fmnist", "organamnist", "bloodmnist", "tissuemnist", "pathmnist"]
DL = ["MNIST", "N-MNIST", "EMNIST", "Fashion", "OrganA", "Blood", "Tissue", "Path$^{\\dagger}$"]
LIGHT = "#D4D4D4"

fig = plt.figure(figsize=(ps.FULL_W, 3.0), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 1.6, 1.6])
ax = fig.add_subplot(gs[0])
vals = {(m, k): [R[("pair_exp", "mnist", s, m)][k] for s in S] for m in ("frozen", "trained") for k in ("posthoc", "linear")}
order = [("frozen", "posthoc"), ("frozen", "linear"), ("trained", "posthoc"), ("trained", "linear")]
xs = [0, 0.9, 2.2, 3.1]
for x, key in zip(xs, order):
    v = np.array(vals[key]); col = ps.GREY if key[1] == "posthoc" else ps.RULE_C["pair_exp"]
    if key[0] == "frozen": col = LIGHT if key[1] == "posthoc" else "#9CC3E0"
    ax.bar(x, v.mean(), 0.8, color=col, yerr=v.std(ddof=1), error_kw=ps.ERR)
    ax.text(x, v.mean() - 2.5, f"{v.mean():.1f}", ha="center", va="top", fontsize=ps.FS - 1, rotation=90, color="white" if key == ("trained", "linear") else ps.INK)
e = inp["mnist"]["encoded"][0]
ax.axhline(e, ls="--", color=ps.INK, lw=0.7)
ax.text(-0.5, e + 1.2, f"input {e:.1f}", ha="left", va="bottom", fontsize=ps.FS)
ax.set_xticks([np.mean(xs[:2]), np.mean(xs[2:])], ["untrained", "trained"])
ax.set_ylim(0, 100); ax.set_ylabel("MNIST accuracy (%)", fontsize=ps.FS + 0.5, labelpad=0)
ax.set_xlim(-0.6, 3.7)
ps.panel_label(ax, "a")

for j, rule in enumerate(["pair_exp", "triplet"]):
    a = fig.add_subplot(gs[j + 1], sharey=None if j == 0 else prev)
    prev = a
    x = np.arange(len(DS)); w = 0.38
    for k, (key, col) in enumerate([("posthoc", ps.GREY), ("linear", ps.RULE_C[rule])]):
        g = np.array([[R[(rule, ds, s, "trained")][key] - R[(rule, ds, s, "frozen")][key] for s in S] for ds in DS])
        a.bar(x + (k - 0.5) * w, g.mean(1), w, yerr=g.std(1, ddof=1), color=col, error_kw=ps.ERR,
              label="neuron labelling" if key == "posthoc" else "linear probe")
    ps.zero_line(a)
    a.set_xticks(x, DL, fontsize=ps.FS, rotation=45, ha="right", rotation_mode="anchor")
    a.set_ylim(-7, 25)
    a.set_title(ps.RULE_L[rule], fontsize=ps.FS)
    if j == 0: a.set_ylabel("gain over untrained network (points)", fontsize=ps.FS + 0.5, labelpad=0)
    else: a.tick_params(labelleft=False)
    a.legend(frameon=False, fontsize=ps.FS, loc="lower left")
    ps.panel_label(a, "bc"[j])
os.makedirs(OUT, exist_ok=True)
ps.save(fig, f"{OUT}/fig_measurement")
print("wrote", OUT)
