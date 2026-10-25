#!/usr/bin/env python3
"""Figure 4 (Section 4.4): when the two terms of Clopath's rule can act.

(a) Voltage ladder. Membrane levels of Clopath's neuron next to ours, on one mV axis
    (broken between -27 and +17 mV so the spike plateau fits):
      Clopath (ModelDB 144566), read from the MATLAB files and their line numbers:
        aEIF.m      th = 20 (spike registered), E_L = -70.6 (rest), VT_rest = -50.4
                    (threshold V_T at rest), VT_jump = 20 (V_T after a spike), tau_VT,
                    the 2-ms spike held at 29.4 then 29.4+3.462, reset E_L+15+6.0984
        RFdevelop.m theta- = E_L + 0 (u_sig_md), theta+ = E_L + 25.3 (uy)
        VoTri.m     tetam = -70.6, tetap = -45.3 (cross-check of the two above)
      ours, read from the tuned network's saved configuration
        run_configs/voltage_S_full_N400_vb3_trained.yaml (the seed-0 run's config.yaml)
        (v_rest, v_reset, v_thresh, refrac_time, theta_ltd, theta_ltp, inh_weight),
        checked against the gate-probe config and Methods 3.6 (paper_drafts/methods.md).
(b) Against the depression threshold theta- (tuned network, MNIST):
      top    potentiation : depression total mass over the first 1,000 training images,
             seed 0 (results/gate_probes_2026-10/p{0,2,3,4,5}_*.json, the probes behind
             Table (b) of results/section44_table.md; scripts/score/make_section44_table.py)
      bottom linear-probe accuracy after 10,000 images, n = 5, mean +- sample SD (single
             probe: fit on 10,000 training images, full test set; results/
             probefix_voltage_thresholds_n5.json), with the STDP-off network as the reference band.
             The potentiation : depression ratios are checked against Table A.7(b) in
             results/section44_table.md.

Style: scripts/figures/paper_style.py (Elsevier sizing, Okabe-Ito). Green and the triangle mean
voltage-STDP and grey the random initialisation, as in the other figures; the two
accents (reddish purple, orange) mark theta- / depression and theta+ / potentiation.
Writes $FIG_OUT_DIR/fig_section44_mechanism.{pdf,png} (default figures/) and prints every plotted number.
Needs CLOPATH_MODELDB_DIR (Clopath's ModelDB 144566 .m files, not redistributed); the tuned
network's configuration is shipped in run_configs/, so no run data is needed.
Run from the repo root: PYTHONPATH=. python scripts/figures/make_section44_mechanism_figure.py
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullLocator

from scripts.figures import paper_style as ps
ps.apply()

ap = argparse.ArgumentParser()
args = ap.parse_args()

# Clopath et al.'s MATLAB code (ModelDB 144566) is not redistributed: download it and set CLOPATH_MODELDB_DIR.
CLO = os.environ.get("CLOPATH_MODELDB_DIR", "")
RUN_CFG = "run_configs/voltage_S_full_N400_vb3_trained.yaml"   # seed-0 config.yaml, device line removed
METHODS = "paper_drafts/methods.md"   # manuscript source, not shipped: its text check runs only if present
APPENDIX = "paper_drafts/appendix.md"
PF = "results/probefix_voltage_thresholds_n5.json"
S44 = "results/section44_table.md"     # gate ratios (no probe involved)
GATES = "results/gate_probes_2026-10/{}.json"
OUT = os.path.join(os.environ.get("FIG_OUT_DIR", "figures"), "fig_section44_mechanism")
if not CLO or not all(os.path.isfile(os.path.join(CLO, m)) for m in ("aEIF.m", "RFdevelop.m", "VoTri.m")):
    sys.exit("ERROR: CLOPATH_MODELDB_DIR is unset or lacks aEIF.m / RFdevelop.m / VoTri.m.\n"
             "Clopath et al.'s MATLAB code (ModelDB accession 144566: aEIF.m, RFdevelop.m, VoTri.m) is not\nredistributed here. Download it from https://modeldb.science/144566 and set\nCLOPATH_MODELDB_DIR to the folder that holds the three .m files.")
if not os.path.isfile(RUN_CFG):
    sys.exit(f"ERROR: {RUN_CFG} not found (run the script from the repository root).")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def fmt(v, nd=1):
    """Number with a typographic minus (U+2212), as matplotlib prints tick labels."""
    s = f"{v:+.{nd}f}" if v > 0 else f"{v:.{nd}f}"
    return s.replace("-", "−")


# =====================================================================================
# 1. Clopath's neuron, from the ModelDB MATLAB code
# =====================================================================================
def grep(path, pattern):
    """First line of `path` matching `pattern`: (match, 1-based line number)."""
    with open(path, encoding="latin-1") as fh:
        for i, line in enumerate(fh, 1):
            m = re.search(pattern, line)
            if m:
                return m, i
    raise SystemExit(f"{path}: no line matches {pattern!r}")


NUM = r"(-?\d+(?:\.\d*)?)"
aeif, rfd, votri = f"{CLO}/aEIF.m", f"{CLO}/RFdevelop.m", f"{CLO}/VoTri.m"
src = {}                                   # name -> (value, "file:line")


def take(name, path, pattern, fn=lambda m: float(m.group(1))):
    m, ln = grep(path, pattern)
    src[name] = (fn(m), f"{os.path.basename(path)}:{ln}")
    return src[name][0]


th_reg = take("spike registered (th)", aeif, rf"^\s*th\s*=\s*{NUM};")
E_L = take("rest E_L", aeif, rf"^\s*E_L\s*=\s*{NUM};")
VT_rest = take("threshold V_T at rest", aeif, rf"^\s*VT_rest\s*=\s*{NUM};")
VT_jump = take("V_T jump after a spike", aeif, rf"^\s*VT_jump\s*=\s*{NUM};")
tau_VT = take("tau_VT (ms)", aeif, rf"^\s*tau_VT\s*=\s*{NUM};")
reset_c = take("reset", aeif, rf"u\s*=\s*E_L\s*\+\s*{NUM}\s*\+\s*{NUM};",
               lambda m: E_L + float(m.group(1)) + float(m.group(2)))
plat2 = take("spike, 2nd ms", aeif, rf"^\s*u\s*=\s*{NUM}\s*\+\s*{NUM};",
             lambda m: float(m.group(1)) + float(m.group(2)))
plat1 = take("spike, 1st ms", aeif, rf"^\s*u\s*=\s*{NUM};")
E_L_rf = take("rest E_L (RFdevelop)", rfd, rf"^\s*E_L\s*=\s*{NUM};")
thm_off = take("theta- offset from E_L", rfd, rf"u_sig_md\s*=\s*\(\(u_md-E_L\)\s*>\s*{NUM}\)")
thp_off = take("theta+ offset from E_L", rfd, rf"uy\s*=\s*\(\(u-E_L\)\s*>\s*{NUM}\)")
tetam = take("theta- (VoTri)", votri, rf"tetam\s*=\s*{NUM};")
tetap = take("theta+ (VoTri)", votri, rf"tetap\s*=\s*{NUM};")
VT_after = VT_rest + VT_jump
th_minus_c, th_plus_c = E_L_rf + thm_off, E_L_rf + thp_off
assert E_L == E_L_rf
assert abs(th_minus_c - tetam) < 1e-9 and abs(th_plus_c - tetap) < 1e-9, \
    "RFdevelop.m and VoTri.m disagree on Clopath's thresholds"

# =====================================================================================
# 2. Our neuron, from the tuned run's config (checked against the probe and Methods)
# =====================================================================================
cfg = yaml.safe_load(open(RUN_CFG))
N, S = cfg["neuron"], cfg["stdp"]
ours = dict(v_rest=float(N["v_rest"]), v_reset=float(N["v_reset"]),
            v_thresh=float(N["v_thresh"]), refrac=int(N["refrac_time"]),
            th_minus=float(S["theta_ltd"]), th_plus=float(S["theta_ltp"]),
            inh=float(cfg["scaffold"]["inh_weight"]))
g_tuned = json.load(open(GATES.format("p0_tuned_repro")))
gc = g_tuned["config"]
for k_probe, k_ours in (("v_rest", "v_rest"), ("v_thresh", "v_thresh"),
                        ("theta_ltd", "th_minus"), ("theta_ltp", "th_plus"),
                        ("inh_weight", "inh")):
    assert float(gc[k_probe]) == ours[k_ours], f"probe config differs from run: {k_probe}"
assert float(gc["A_plus"]) == float(S["A_plus"]) and float(gc["A_minus"]) == float(S["A_minus"])
assert th_minus_c == ours["th_minus"] and th_plus_c == ours["th_plus"], \
    "the tuned run does not use Clopath's published thresholds"
gap_rest = ours["v_rest"] - ours["th_minus"]          # theta- below our rest
gap_thr = ours["th_plus"] - ours["v_thresh"]          # theta+ above our threshold
have_ms = os.path.isfile(METHODS) and os.path.isfile(APPENDIX)
methods = (open(METHODS).read() + open(APPENDIX).read()) if have_ms else ""   # values moved to Appendix E on 2026-10-18
if not have_ms:
    print("note: manuscript sources (paper_drafts/) not present; skipping the Methods/Appendix text check")
for s in () if not have_ms else (f"${gap_rest:.1f}$ mV", f"${gap_thr:.1f}$ mV", f"-52$ mV", f"$-65$ mV",
          f"+{plat1}"):   # "5-ms refractory" check dropped 2026-10-20 (now only in Table A.1)
    assert s in methods, f"Methods / appendix no longer contain {s!r}"
gate_open = 100 * g_tuned["ltd_u_gate_active_frac"]
above_thp = 100 * g_tuned["ltp_v_gate_active_frac"]

# =====================================================================================
# 3. Panel (b) data
# =====================================================================================
# 2026-10-20: accuracies from the single linear probe (fit on 10,000 training images, full test
# set), five seeds: results/probefix_voltage_thresholds_n5.json
# (scripts/score/probefix_voltage_thresholds_aggregate.py). Replaced the three-seed voltage-STDP probe.
SWEEP = [(-70.6, "trained"), (-68.0, "thltd_m68"), (-65.0, "thltd_m65"), (-62.0, "thltd_m62"),
         (-60.0, "thltd_m60"), (-58.0, "thltd_m58"), (-55.0, "thltd_m55"), (-50.0, "thltd_m50")]
_pf = json.load(open(PF))["settings"]
acc = []
for th, key in SWEEP:
    assert _pf[key]["n"] == 5, key
    acc.append((th, *_pf[key]["accuracy"]))
rnd = tuple(_pf["frozen"]["accuracy"])

PROBES = [(-70.6, "p0_tuned_repro"), (-65.0, "p2_thltd_m65"), (-60.0, "p3_thltd_m60"),
          (-58.0, "p4_thltd_m58"), (-50.0, "p5_thltd_m50")]
gates = []
for th, f in PROBES:
    g = json.load(open(GATES.format(f)))
    assert float(g["config"]["theta_ltd"]) == th and int(g["config"]["seed"]) == 0
    assert int(g["config"]["n_train"]) == 1000
    gates.append((th, g))


def ratio_label(g):          # the formatting of scripts/score/make_section44_table.py
    lp, ld = g["ltp_mass_total"], g["ltd_mass_total"]
    if ld == 0:
        return "none"
    r = lp / ld
    return f"1 : {1 / r:,.0f}" if r < 1 else f"{r:.1f} : 1"


tab_b = {}
for line in open(S44):
    m = re.match(r"\| (tuned|\$\\theta_\{-\}=(-?[\d.]+)\$ mV) \|.*\| ([^|]+) \|\s*$", line)
    if m:
        tab_b[-70.6 if m.group(1) == "tuned" else float(m.group(2))] = m.group(3).strip()
for th, g in gates:
    assert tab_b[th] == ratio_label(g), f"ratio at {th}: {tab_b[th]} vs {ratio_label(g)}"

# =====================================================================================
# 4. Print every number the figure shows
# =====================================================================================
print("(a) Clopath's neuron (ModelDB 144566)")
for k, (v, where) in src.items():
    print(f"  {k:28s} {v:9.4f}   {where}")
print(f"  {'V_T after a spike':28s} {VT_after:9.4f}   VT_rest + VT_jump")
print(f"  theta- - rest = {th_minus_c - E_L:+.1f} mV   theta+ - V_T(rest) = {th_plus_c - VT_rest:+.1f} mV")
print(f"(a) our neuron ({RUN_CFG})")
for k, v in ours.items():
    print(f"  {k:28s} {v}")
print(f"  rest - theta- = {gap_rest:.1f} mV   theta+ - threshold = {gap_thr:.1f} mV")
print(f"  tuned probe (seed 0, 1,000 images, {g_tuned['n_events']:,} neuron-steps): "
      f"depression gate open {gate_open:.1f}% (closed {100 - gate_open:.1f}%), "
      f"membrane above theta+ {above_thp:.4f}%, max pre-reset voltage "
      f"{g_tuned['v_max_over_all_events_mV']:.2f} mV")
print("(b) linear-probe accuracy, n=5, mean ± sample SD   [results/probefix_voltage_thresholds_n5.json]")
for th, mu, sd in acc:
    print(f"  theta- {th:6.1f}  {mu:6.2f} ± {sd:5.2f}   vs random {mu - rnd[0]:+6.2f} points")
print(f"  random (STDP off)  {rnd[0]:6.2f} ± {rnd[1]:5.2f}")
print("(b) potentiation : depression, seed 0, first 1,000 training images   [gate probes]")
for th, g in gates:
    print(f"  theta- {th:6.1f}  pot {g['ltp_mass_total']:12.4f}  dep {g['ltd_mass_total']:12.4f}  "
          f"-> {ratio_label(g)}")

# =====================================================================================
# 5. Figure
# =====================================================================================
INK, GREY, VC = ps.INK, ps.GREY, ps.RULE_C["voltage"]
C_DEP, C_POT = ps.ACC1, ps.ACC2          # theta- / depression, theta+ / potentiation
MUTED = "#555555"
FS = ps.FS
# 2026-10-13 redesign for print legibility: (a) two horizontal voltage scales showing only
# the levels the argument uses; (b) potentiation:depression ratio and (c) accuracy vs theta-,
# side by side. Values are those loaded and checked above.
W, H = ps.FULL_W, 3.55
fig = plt.figure(figsize=(W, H))
axa = fig.add_axes([0.125, 0.665, 0.865, 0.25])
axb = fig.add_axes([0.095, 0.115, 0.36, 0.40])
axc = fig.add_axes([0.615, 0.115, 0.37, 0.40])

# ---------------------------------- (a) voltage scales ------------------------------
XL_A = (-73.5, -29.5)
axa.set_xlim(*XL_A); axa.set_ylim(-1.25, 2.0)
for sp in ("left", "right", "top"):
    axa.spines[sp].set_visible(False)
axa.set_yticks([]); axa.set_xticks([-70, -65, -60, -55, -50, -45])
axa.spines["bottom"].set_bounds(-73.5, -43.0)
axa.set_xlabel("membrane potential (mV)", labelpad=1)
rows = {"Clopath's neuron": 1.1, "our neuron": 0.0}
for lab, y in rows.items():
    axa.plot(XL_A, [y, y], color="#CCCCCC", lw=0.8, zorder=1)
    axa.text(XL_A[0] - 0.6, y, lab.replace(" ", "\n", 1), ha="right", va="center", fontsize=FS,
             color=INK, linespacing=1.0)


def mark(x, y, label, color, above=True, dx=0.0, marker="|", ms=9):
    axa.plot([x], [y], marker=marker, ms=ms, mew=1.4, color=color, zorder=3)
    axa.text(x + dx, y + (0.16 if above else -0.17), label, ha="center",
             va="bottom" if above else "top", fontsize=FS, color=color, linespacing=1.0)


yc, yo = rows["Clopath's neuron"], rows["our neuron"]
# Clopath: rest = theta-, firing threshold V_T, theta+
mark(E_L, yc, f"rest = $\\theta_{{-}}$\n{fmt(E_L)}", C_DEP)
mark(VT_rest, yc, f"firing\nthreshold\n{fmt(VT_rest)}", INK, dx=-0.9)
mark(th_plus_c, yc, f"$\\theta_{{+}}$\n{fmt(th_plus_c)}", C_POT, dx=0.9)
axa.annotate("", xy=(-42.0, yc), xytext=(th_plus_c + 0.4, yc),
             arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C_POT, mutation_scale=6))
axa.text(-41.6, yc, f"held at +30 mV for 2 ms,\nthen reset to {fmt(reset_c)} mV", ha="left", va="center",
         fontsize=FS, color=C_POT, linespacing=1.0)
# ours: theta- below rest, rest, firing threshold, theta+
_y0, _y1 = axa.get_ylim()
axa.axvspan(ours["th_minus"], ours["v_rest"], ymin=(yo - 0.10 - _y0) / (_y1 - _y0),
            ymax=(yo + 0.10 - _y0) / (_y1 - _y0), color=C_DEP, alpha=0.22, lw=0)
mark(ours["th_minus"], yo, f"$\\theta_{{-}}$\n{fmt(ours['th_minus'])}", C_DEP, above=False)
mark(ours["v_rest"], yo, f"rest\n{fmt(ours['v_rest'], 0)}", INK, above=False)
mark(ours["v_thresh"], yo, f"firing\nthreshold\n{fmt(ours['v_thresh'], 0)}", INK, above=False)
mark(th_plus_c, yo, f"$\\theta_{{+}}$\n{fmt(ours['th_plus'])}", C_POT, above=False)
axa.text((ours["th_minus"] + ours["v_rest"]) / 2, yo + 0.14, f"{gap_rest:.1f} mV",
         ha="center", va="bottom", fontsize=FS, color=C_DEP, linespacing=1.0)
axa.text(-41.6, yo, f"reset to {fmt(ours['v_reset'], 0)} mV at\nthreshold, no plateau", ha="left", va="center",
         fontsize=FS, color=INK, linespacing=1.0)

# ---------------------------------- (b) potentiation : depression -------------------
XL = (-72.4, -48.6)
for th, g in gates:
    lp, ld = g["ltp_mass_total"], g["ltd_mass_total"]
    if ld == 0:
        axb.text(th, 1.5, "no\ndepression", ha="center", va="bottom", fontsize=FS,
                 color=MUTED, linespacing=1.0)
        continue
    r = lp / ld
    lo, hi = min(r, 1.0), max(r, 1.0)
    axb.bar(th, hi - lo, bottom=lo, width=1.4, color=C_DEP if r < 1 else C_POT, ec="none", zorder=2)
    lab = ratio_label(g).replace(" : ", ":")
    if r < 1 and th > -61:      # -60 mV: below the bar (its right neighbour is the -58 bar)
        axb.text(th, r / 1.6, lab, ha="center", va="top", fontsize=FS, color=INK)
    elif r < 1:
        axb.text(th + 0.9, r * 1.3, lab, ha="left", va="bottom", fontsize=FS, color=INK)
    else:
        axb.text(th, r * 1.6, lab, ha="center", va="bottom", fontsize=FS, color=INK)
axb.set_yscale("log"); axb.set_ylim(10 ** -4.9, 10 ** 1.6); axb.set_xlim(*XL)
axb.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 10]))
axb.yaxis.set_minor_locator(NullLocator())
axb.set_yticklabels(["1:10$^{4}$", "1:100", "1:1", "10:1"])
axb.axhline(1.0, color=INK, lw=0.6, zorder=1.5)
axb.set_ylabel("potentiation : depression")
axb.set_xlabel(r"depression threshold $\theta_{-}$ (mV)")
axb.set_xticks([-70, -65, -60, -55, -50])

# ---------------------------------- (c) accuracy -----------------------------------
axc.axhspan(rnd[0] - rnd[1], rnd[0] + rnd[1], color=GREY, alpha=0.35, lw=0, zorder=1)
axc.axhline(rnd[0], color=GREY, lw=0.8, zorder=1.2)
axc.text(XL[0] + 0.4, rnd[0] + rnd[1] + 1.5, "STDP disabled", ha="left", va="bottom",
         fontsize=FS, color=MUTED)
xs = [a[0] for a in acc]; mu = [a[1] for a in acc]; sds = [a[2] for a in acc]
axc.errorbar(xs, mu, yerr=sds, fmt="none", ecolor=VC, elinewidth=0.6, capthick=0.6,
             capsize=1.6, zorder=3)
axc.plot(xs, mu, color=VC, lw=1.0, marker=ps.RULE_M["voltage"], ms=3.6, mfc=VC,
         mec="white", mew=0.4, zorder=4, label="trained")
axc.set_xlim(*XL); axc.set_ylim(0, 85); axc.set_yticks([0, 20, 40, 60, 80])
axc.set_ylabel("accuracy (%)")
axc.set_xlabel(r"depression threshold $\theta_{-}$ (mV)")
axc.set_xticks([-70, -65, -60, -55, -50])
for ax in (axb, axc):
    ps.guide_line(ax, ours["v_rest"])
axb.text(ours["v_rest"], 1.02, "our rest", transform=axb.get_xaxis_transform(), ha="center",
         va="bottom", fontsize=FS, color=MUTED)
axc.text(ours["v_rest"], 1.02, "our rest", transform=axc.get_xaxis_transform(), ha="center",
         va="bottom", fontsize=FS, color=MUTED)

for x, y, lab in ((0.005, 0.965, "(a)"), (0.005, 0.56, "(b)"), (0.525, 0.56, "(c)")):
    fig.text(x, y, lab, fontsize=ps.FS_PANEL, fontweight="bold", color=INK, ha="left", va="top")
ps.save(fig, OUT)
plt.close(fig)
print(f"wrote {OUT}.pdf and {OUT}.png")
