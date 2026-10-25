#!/usr/bin/env python3
"""Table A.7(b) (Section 4.4): how often each term of the voltage-STDP rule can act.

Gates: scripts/score/voltage_gate_saturation_probe.py on the tuned configuration,
seed 0, first 1,000 training images from initialisation (results/gate_probes_2026-10/).
"Depression gate open" = (u_minus_bar - theta_ltd) > 0; "membrane above theta+" =
(v - theta_ltp) > 0; mass = summed potentiation / depression over the 1,000 images.

Table A.7(a) (accuracy of the theta- sweep with the single linear probe, five seeds) is
written by scripts/score/probefix_voltage_thresholds_aggregate.py. The part (a) this script
used to compute (the retired 70/30 probe on the three-seed thltd_* runs) is not in the release.

Writes results/section44_table.md (paper markdown; read by
scripts/figures/make_section44_mechanism_figure.py) and prints a plain version.
Usage (repo root): PYTHONPATH=. python scripts/score/make_section44_table.py
"""
import json

GATES = "results/gate_probes_2026-10/{}.json"
OUT = "results/section44_table.md"


def ratio(g):
    lp, ld = g["ltp_mass_total"], g["ltd_mass_total"]
    if ld == 0:
        return "none"
    r = lp / ld
    return f"1 : {1 / r:,.0f}" if r < 1 else f"{r:.1f} : 1"


GATE_ROWS = [("tuned", "p0_tuned_repro"),
             ("inhibition off", "p1_no_inhibition"),
             ("theta- = -65 mV", "p2_thltd_m65"), ("theta- = -60 mV", "p3_thltd_m60"),
             ("theta- = -58 mV", "p4_thltd_m58"), ("theta- = -50 mV", "p5_thltd_m50"),
             ("theta+ = -52 mV", "p6_thltp_m52")]
gates = [(lab, json.load(open(GATES.format(f)))) for lab, f in GATE_ROWS]

# ---- plain print ------------------------------------------------------------------
print("(b) gates, seed 0, first 1,000 training images")
for lab, g in gates:
    print(f"  {lab:24s} spikes/img {g['n_spikes'] / 1000:6.0f}  dep gate {100 * g['ltd_u_gate_active_frac']:5.1f}%  "
          f"v>theta+ {100 * g['ltp_v_gate_active_frac']:7.4f}%  pot:dep {ratio(g)}")

# ---- paper markdown -------------------------------------------------------------------
L = []
L.append("**Table A.7(b).** Gate activity in the tuned network ($\\theta_{-}=-70.6$, "
         "$\\theta_{+}=-45.3$ mV) and in variants that change one setting, over the first "
         "$1{,}000$ training images (seed $0$): the share of neuron-steps on which the depression gate is "
         "open ($\\bar u_{-}>\\theta_{-}$) and on which the membrane exceeds $\\theta_{+}$, and "
         "the ratio of total potentiation to total depression. At $\\theta_{-}=-50$ mV there is "
         f"no depression and potentiation is negligible (total ${gates[5][1]['ltp_mass_total']:.3f}$, "
         f"against ${gates[0][1]['ltp_mass_total']:.0f}$ in the tuned network), because "
         "potentiation also needs $\\bar u_{+}>\\theta_{-}$. Part (a), accuracy, is in "
         "results/probefix_voltage_thresholds_n5.md.")
L.append("")
L.append("| **(b)** setting | depression gate open (%) | membrane above $\\theta_{+}$ (%) | potentiation : depression |")
L.append("|---|--:|--:|--:|")
for lab, g in gates:
    lab_t = (lab.replace("theta- = ", "$\\theta_{-}=").replace("theta+ = ", "$\\theta_{+}=")
             .replace(", $\\theta_{+}=", "$, $\\theta_{+}=").replace(" mV", "$ mV"))
    L.append(f"| {lab_t} | {100 * g['ltd_u_gate_active_frac']:.1f} | "
             f"{100 * g['ltp_v_gate_active_frac']:.4f} | {ratio(g)} |")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}")
