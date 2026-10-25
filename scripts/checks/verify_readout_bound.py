#!/usr/bin/env python3
"""Empirical check of Proposition 1 (post-hoc labelling is a constrained linear
readout, so best-linear >= post-hoc in classifier capacity).

We test the *raw test-set* inequality  post-hoc <= linear  across every
rule x scaffold x seed cell of the MNIST scaffold matrix, and characterise any
exceptions.

Sources (two files, joined by run):
  * results/scaffold_readout_table.csv (60 cells) for the linear probe, written
    by scripts/score/scaffold_readout_table.py. Linear probe = LogisticRegression fit
    on the TRAIN features, evaluated on the full 10k test set -- the matched
    protocol Section 4.1 reports.
  * results/posthoc_both_conventions.csv (`posthoc_dc`) for the post-hoc
    accuracy. This is the Diehl-Cook convention with both divisions, which is
    what Table A.3 prints and Section 4.1 reports. The `posthoc` column of
    scaffold_readout_table.csv is the SUPERSEDED convention and is deliberately
    not used here -- it shifts the gap bounds to +5.3 / +22.8 pp.

Expected outcome (Section 4.1):
  * 0 violations in all 60 cells, and 0 among the cells whose linear readout
    clears the majority floor;
  * gap over all 60 cells: min +4.5, mean +9.6, max +23.3 pp -- the three
    values Section 4.1 quotes;
  * pair/triplet gap median +7.7, max +23.3 pp (the largest gap of all 60 cells is a
    pair/triplet cell).
"""
import csv, sys
import numpy as np

CSV = "results/scaffold_readout_table.csv"
PH_CSV = "results/posthoc_both_conventions.csv"
MAJORITY = 0.1135  # MNIST class 1: 1135/10000

rows = list(csv.DictReader(open(CSV)))

# Post-hoc accuracy comes from the Diehl-Cook convention (both divisions), as
# printed in Table A.3 -- not from the superseded `posthoc` column of CSV.
POSTHOC_DC = {r["run"]: float(r["posthoc_dc"]) / 100.0
              for r in csv.DictReader(open(PH_CSV))
              if r["group"] == "scaffold" and r["dataset"] == "mnist"}
missing = [r["run"] for r in rows if r["run"] not in POSTHOC_DC]
if missing:
    sys.exit(f"{PH_CSV} has no Diehl-Cook post-hoc value for {len(missing)} of "
             f"{len(rows)} cells, e.g. {missing[:3]} -- regenerate it with "
             f"scripts/score/posthoc_both_conventions.py before verifying the bound.")

ph = lambda r: POSTHOC_DC[r["run"]]
lg = lambda r: float(r["linear"])
viol  = [r for r in rows if ph(r) > lg(r) + 1e-9]
pt    = [r for r in rows if r["rule"] in ("pair_exp", "triplet")]
above = [r for r in rows if lg(r) > MAJORITY]
n = len(rows)
g_pt  = np.array([lg(r) - ph(r) for r in pt]) * 100
g_all = np.array([lg(r) - ph(r) for r in rows]) * 100

print(f"cells                              : {n}")
print(f"raw test-set bound holds           : {n - len(viol)}/{n}")
print(f"  among pair/triplet               : {len(pt) - sum(ph(r) > lg(r) + 1e-9 for r in pt)}/{len(pt)}")
print(f"  among linear > majority floor    : {len(above) - sum(ph(r) > lg(r) + 1e-9 for r in above)}/{len(above)}")
print(f"gap, all 60 cells  min/mean/max    : {g_all.min():+.2f} / {g_all.mean():+.2f} / {g_all.max():+.2f} pp   <- quoted in 4.1")
print(f"gap, pair/triplet  median / max    : {np.median(g_pt):+.2f} / {g_pt.max():+.2f} pp   <- Proposition 1 check")
print(f"exceptions                         : {len(viol)}")
if viol:
    rules = sorted({r["rule"] for r in viol})
    phs = sorted({round(ph(r), 4) for r in viol})
    short = [ph(r) - lg(r) for r in viol]
    print(f"  rules={rules}  post_hoc values={phs}  (=majority floor {MAJORITY})")
    print(f"  max shortfall below floor        : {max(short)*100:.2f} pp")
    print(f"  all exceptions at maj. floor     : {all(abs(ph(r) - MAJORITY) < 1e-9 for r in viol)}")

# The capacity claim is falsified only if a rule that actually decodes structure
# (linear readout above the majority floor) is still beaten by post-hoc.
capacity_ok = all(ph(r) <= lg(r) + 1e-9 for r in above)
print(f"\nCAPACITY BOUND (above-floor cells): {'HOLDS' if capacity_ok else 'VIOLATED'}")
sys.exit(0 if capacity_ok else 1)
