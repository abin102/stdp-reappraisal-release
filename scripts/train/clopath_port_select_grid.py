#!/usr/bin/env python3
"""Pre-registered choice of the 3 x 3 (input gain g, inhibition k_inh) grid from the
STDP-off calibration (results/voltage_clopath_neuron_port_PREREG.md, Section 5).

Input: calibration.json from `scripts/train/clopath_port_mnist.py calibrate` (seed 0, first 1,000
training images, STDP off, no labels). For every product grid {g1<g2<g3} x {k1<k2<k3} of
calibration values with k1 > 0 (every cell has inhibition), score
    sum over the six marginal targets of min over the 9 cells of |log10(value / target)|,
targets: rate 2, sqrt(40) = 6.32, 20 Hz; active fraction 0.05, sqrt(0.025) = 0.158, 0.5.
Cells with zero rate count as infinitely far. Lowest score wins; ties go to the
lexicographically smallest (g1, g2, g3, k1, k2, k3). Deterministic, no free choice.
Output: grid.json (cells c11..c33, row = g, column = k_inh) + the calibration values.
"""
import argparse
import itertools
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = json.load(open(a.calibration))
    R = {(c["g"], c["k_inh"]): c["rate_hz"] for c in cal["cells"]}
    F = {(c["g"], c["k_inh"]): c["active_fraction"] for c in cal["cells"]}
    gs = sorted(cal["g"]); ks = sorted(k for k in cal["k"] if k > 0)
    tr = (2.0, math.sqrt(40.0), 20.0); tf = (0.05, math.sqrt(0.025), 0.5)

    def d(v, t):
        return abs(math.log10(v / t)) if v > 0 else math.inf

    best = None
    for G in itertools.combinations(gs, 3):
        for K in itertools.combinations(ks, 3):
            cells = [(g, k) for g in G for k in K]
            score = sum(min(d(R[c], t) for c in cells) for t in tr) + \
                sum(min(d(F[c], t) for c in cells) for t in tf)
            key = (score, G + K)
            if best is None or key < best[0]:
                best = (key, G, K)
    (score, _), G, K = best
    cells = [dict(id=f"c{i + 1}{j + 1}", g=g, k_inh=k, calib_rate_hz=R[(g, k)],
                  calib_active_fraction=F[(g, k)]) for i, g in enumerate(G) for j, k in enumerate(K)]
    out = dict(rule="scripts/train/clopath_port_select_grid.py (PREREG Section 5)", calibration=a.calibration,
               score=score, g=list(G), k_inh=list(K), cells=cells, amplitudes=["10x", "1x"],
               nearest=dict(rate={str(t): min(cells, key=lambda c: d(c["calib_rate_hz"], t))["id"] for t in tr},
                            active={str(t): min(cells, key=lambda c: d(c["calib_active_fraction"], t))["id"] for t in tf}))
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"score {score:.3f}  g={G}  k_inh={K}")
    for c in cells:
        print(f"  {c['id']}: g={c['g']}, k={c['k_inh']}: rate {c['calib_rate_hz']:.2f} Hz, "
              f"active {100 * c['calib_active_fraction']:.1f} %")


if __name__ == "__main__":
    main()
