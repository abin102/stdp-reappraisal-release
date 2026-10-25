#!/usr/bin/env python3
"""Why the port potentiates at low-frequency pre-post pairing while VoTri.m does not.

RFdevelop.m reads the voltage averages and the homeostatic average from u1ss, which (after
`u1s = u; u1ss = u1s;`) is u one step late; VoTri.m reads u_md(t-3), whose newest input is
u(t-4), i.e. 4 steps late ("I wait 3ms ... before I read the filtered version"). Here the NumPy
reference (RFDevelopRef, unchanged) is subclassed so that u1ss is u d steps late; d = 1 must
reproduce the reference exactly. The Sjostrom protocol of VoTri.m (published amplitudes,
homeostatic factor 1, pre spikes do not drive the neuron) is run for d = 1, 2, 3, 4.
Out: results/clopath_port_validation/delay_check.json
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
from models import clopath_port as cp  # noqa: E402


class DelayedRef(cp.RFDevelopRef):
    def __init__(self, *a, delay=1, **k):
        super().__init__(*a, **k)
        self.delay = delay
        self.hist = collections.deque([cp.E_L] * delay, maxlen=delay)

    def step(self, inp, I_extra=0.0, u_clamp=None):
        self.u1ss = self.hist[0]                 # u, `delay` steps before this step
        out = super().step(inp, I_extra, u_clamp)
        self.hist.append(self.u)                 # super() set u1ss = u (delay 1); overwritten next step
        return out


def run(rho, Dt, delay):
    x, y = cp.votri_trains(rho, Dt)
    ref = DelayedRef(np.array([0.5]), A3p=cp.A3p_PUB, A2d=cp.A2d_PUB, g=0.0, theta_fixed=cp.theta_tar,
                     delay=delay) if delay else cp.RFDevelopRef(np.array([0.5]), A3p=cp.A3p_PUB,
                                                                  A2d=cp.A2d_PUB, g=0.0, theta_fixed=cp.theta_tar)
    for t in range(4, len(x)):
        ref.step(np.array([x[t]]), I_extra=y[t] * 1000000)
    return float(ref.w[0] - 0.5)


def main():
    rhos = [0.1, 1, 5, 10, 20, 40, 50]
    out = {}
    for d in (0, 1, 2, 3, 4):                    # 0 = the unmodified reference class
        out[str(d)] = {f"{r}|{dt}": run(r, dt, d) for r in rhos for dt in (10, -10)}
        print(f"delay {d if d else '1 (reference class)'}: " + "  ".join(
            f"{r}Hz {100 * out[str(d)][f'{r}|10'] / 0.5:+.0f}/{100 * out[str(d)][f'{r}|-10'] / 0.5:+.0f}%" for r in rhos),
            flush=True)
    assert out["0"] == out["1"], "delay-1 subclass must reproduce the reference exactly"
    eq = json.load(open("results/clopath_port_validation/equivalence.json"))
    R = np.array(eq["votri_octave_R"])
    out["votri_octave"] = {f"{r}|{dt}": float(R[i, j] - 0.5) for i, r in enumerate(rhos) for j, dt in enumerate((10, -10))}
    print("VoTri.m (Octave):      " + "  ".join(
        f"{r}Hz {100 * out['votri_octave'][f'{r}|10'] / 0.5:+.0f}/{100 * out['votri_octave'][f'{r}|-10'] / 0.5:+.0f}%" for r in rhos))
    json.dump(dict(note="weight change (absolute, w0 = 0.5) for the Sjostrom protocol; keys 'rho|dt'; "
                        "'0' = unmodified RFDevelopRef, '1'-'4' = newest voltage in the averages is u(t-d)",
                   results=out), open("results/clopath_port_validation/delay_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
