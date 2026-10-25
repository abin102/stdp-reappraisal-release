#!/usr/bin/env python3
"""Validate the Clopath port where Clopath says the model works.

(a) RFdevelop.m's own setup: 1 neuron, 500 inputs, 10 Gaussian patterns (one pattern per
    100 ms), 100 s, RFdevelop's 10x amplitudes; 10 seeds. Expected: the weights become
    selective, concentrating on one pattern's bump. Also the published 1x amplitudes over
    1,000 s (10x longer), 10 seeds.
(b) Pairing protocols through the full port (aEIF.m neuron + RFdevelop.m rule), as in
    Clopath's VoTri.m: postsynaptic spikes induced by a 1e6-pA current step, presynaptic
    spikes do not drive the neuron (g = 0), homeostatic factor held at 1 (theta = theta_tar),
    published amplitudes, w0 = 0.5:
      - frequency dependence: 5 pairings at rho Hz, dt = +10 / -10 ms, repeated 15 times at
        0.1 Hz (10 times at 0.1 Hz), rho in {0.1, 1, 5, 10, 20, 40, 50};
      - timing window: the same protocol at 10 and 20 Hz, dt from -40 to +40 ms;
      - voltage dependence: postsynaptic voltage clamped at u, 50 presynaptic spikes at 2 Hz.
    Compared with Clopath's own VoTri.m run in Octave (results/clopath_port_validation/
    equivalence.json, filled by scripts/checks/clopath_port_equivalence.py) and with the shapes
    of Clopath et al. 2010 Figs 1-2.

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/checks/clopath_port_validate.py
Out: results/clopath_port_validation/{rf_development.png, protocols.png, validation.json}
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
from models import clopath_port as cp  # noqa: E402

OUT = "results/clopath_port_validation"
torch.set_grad_enabled(False)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# (a) receptive-field development
# ----------------------------------------------------------------------
def patterns(Ni=500, P=10, sigma=10, nu_in_max=0.015, nu_in_min=0.0001):
    ind = np.arange(1, Ni + 1, dtype=float)
    gau = nu_in_min + nu_in_max * np.exp(-(ind - Ni / 2) ** 2 / (2 * sigma ** 2))
    gau = np.concatenate([gau, gau])
    pat = np.zeros((Ni, P))
    for jj in range(1, P + 1):
        mup = int(1 + (jj - 1) * Ni / P)
        pat[:, jj - 1] = gau[mup - 1:mup - 1 + Ni]
    return pat


def bump_fraction(w, centres, half=20):
    """Fraction of total weight within +-half inputs (2 sigma) of each bump centre (circular)."""
    Ni = w.shape[-1]
    fr = []
    for c in centres:
        idx = (np.arange(c - half, c + half + 1)) % Ni
        fr.append(w[..., idx].sum(-1) / np.maximum(w.sum(-1), 1e-12))
    return np.stack(fr, -1)


def rf_development(seeds, NEP, A3p, A2d, label):
    Ni, P, EP = 500, 10, 100
    pat = torch.as_tensor(patterns(), device=DEV)                     # (Ni, P)
    centres = [int(np.argmax(patterns()[:, j])) for j in range(P)]
    B = len(seeds)
    W0 = np.stack([np.random.default_rng(s).random(Ni) * 1.5 + 0.5 for s in seeds])[:, :, None]
    net = cp.ClopathNet(W0, g=cp.G_RF, k_inh=0.0, A3p=A3p, A2d=A2d, dtype=torch.float64, device=DEV)
    pseq = np.stack([np.random.default_rng(1000 + s).integers(0, P, NEP) for s in seeds])  # (B, NEP)
    gen = torch.Generator(device=DEV).manual_seed(int(seeds[0]) * 7919 + NEP)
    T = EP * NEP
    save_every = max(T // 1000, 100)
    w_save = np.zeros((B, Ni, T // save_every + 1))
    n_spk = torch.zeros(B, device=DEV)
    t0 = time.time()
    rates = pat.T.unsqueeze(0).expand(B, P, Ni)                        # (B, P, Ni)
    for ep in range(NEP):
        pr = rates[torch.arange(B), torch.as_tensor(pseq[:, ep], device=DEV)]   # (B, Ni)
        for k in range(EP):
            t = ep * EP + k + 1
            if t == 1:
                continue                                               # RFdevelop loops t = 2:T
            inp = torch.rand(B, Ni, generator=gen, device=DEV, dtype=torch.float64) < pr
            n_spk += net.step(inp)[:, 0].double()
            if t % save_every == 0:
                w_save[:, :, t // save_every] = net.W[:, :, 0].cpu().numpy()
    w_final = net.W[:, :, 0].cpu().numpy()
    bf = bump_fraction(w_final, centres)
    best = bf.argmax(1)
    out = dict(label=label, seeds=list(map(int, seeds)), seconds_simulated=T / 1000, A3p=A3p, A2d=A2d,
               wall_s=time.time() - t0, spikes=n_spk.cpu().numpy().tolist(),
               best_pattern=best.tolist(), best_bump_weight_fraction=bf.max(1).tolist(),
               uniform_expectation=41 / 500,
               weights_at_max=(w_final >= cp.wmax - 1e-9).sum(1).tolist(),
               weights_at_zero=(w_final <= 1e-12).sum(1).tolist(),
               weights_at_max_inside_best_bump=[int(((w_final[i] >= cp.wmax - 1e-9)[
                   (np.arange(centres[best[i]] - 20, centres[best[i]] + 21)) % Ni]).sum()) for i in range(B)])
    print(f"  {label}: best-bump weight fraction per seed "
          f"{np.round(bf.max(1), 2).tolist()} (uniform 0.08); {out['wall_s']:.0f} s", flush=True)
    return out, w_save, w_final, centres, save_every


# ----------------------------------------------------------------------
# (b) protocols
# ----------------------------------------------------------------------
def run_trains(trains, A3p=cp.A3p_PUB, A2d=cp.A2d_PUB, w0=0.5):
    """trains: list of (x, y) 1-based arrays. Steps t = 4..L as in VoTri.m. Returns final w."""
    B = len(trains)
    L = max(len(x) for x, _ in trains) - 1
    X = np.zeros((L + 1, B)); Y = np.zeros((L + 1, B))
    for i, (x, y) in enumerate(trains):
        X[:len(x), i] = x; Y[:len(y), i] = y
    net = cp.ClopathNet(np.full((B, 1, 1), w0), g=0.0, k_inh=0.0, A3p=A3p, A2d=A2d,
                        theta_fixed=cp.theta_tar, dtype=torch.float64, device=DEV)
    Xt = torch.as_tensor(X, device=DEV); It = torch.as_tensor(Y * 1000000, device=DEV)
    nsp = torch.zeros(B, device=DEV)
    for t in range(4, L + 1):
        nsp += net.step(Xt[t].unsqueeze(1), I_ext=It[t].unsqueeze(1))[:, 0].double()
    return net.W[:, 0, 0].cpu().numpy() - w0, nsp.cpu().numpy()


def voltage_clamp(us, A3p=cp.A3p_PUB, A2d=cp.A2d_PUB, n_pre=50, isi=500, warm=500, w0=0.5):
    B = len(us)
    T = warm + n_pre * isi
    net = cp.ClopathNet(np.full((B, 1, 1), w0), g=0.0, k_inh=0.0, A3p=A3p, A2d=A2d,
                        theta_fixed=cp.theta_tar, dtype=torch.float64, device=DEV)
    uc = torch.as_tensor(np.asarray(us, float).reshape(B, 1), device=DEV)
    one = torch.ones(B, 1, device=DEV, dtype=torch.float64); zero = torch.zeros_like(one)
    for t in range(1, T + 1):
        pre = one if (t > warm and (t - warm) % isi == 1) else zero
        net.step(pre, u_clamp=uc)
    dw = net.W[:, 0, 0].cpu().numpy() - w0
    # analytic expectation per presynaptic spike once the averages equal the clamp:
    # (u - theta-)+ * ( -A2d + A3p * (u - theta+)+ * sum_t r ), with sum_t r = 1 per spike
    an = np.array([max(u - cp.E_L, 0) * (-A2d + A3p * max(u - cp.E_L - cp.UY_TH, 0)) for u in us]) * n_pre
    return dw, an


def main():
    os.makedirs(OUT, exist_ok=True)
    res = dict(date=time.strftime("%Y-%m-%d %H:%M"), device=DEV)
    print("(a) receptive-field development", flush=True)
    seeds = list(range(1, 11))
    a10, ws10, wf10, centres, se10 = rf_development(seeds, 1000, cp.A3p_RF, cp.A2d_RF,
                                                    "RFdevelop amplitudes (10x), 100 s")
    a1, ws1, wf1, _, se1 = rf_development(seeds, 10000, cp.A3p_PUB, cp.A2d_PUB,
                                          "published amplitudes (1x), 1000 s")
    res["rf_10x_100s"] = a10; res["rf_1x_1000s"] = a1; res["bump_centres_0based"] = centres

    print("(b) protocols through the port", flush=True)
    rhos = [0.1, 1, 5, 10, 20, 40, 50]
    trains = [cp.votri_trains(r, d) for r in rhos for d in (10, -10)]
    dw, nsp = run_trains(trains)
    freq = {f"{r}|{d}": float(dw[i * 2 + j]) for i, r in enumerate(rhos) for j, d in enumerate((10, -10))}
    res["frequency_port"] = freq
    res["frequency_port_post_spikes"] = {f"{r}|{d}": int(nsp[i * 2 + j]) for i, r in enumerate(rhos)
                                         for j, d in enumerate((10, -10))}
    dts = list(range(-40, 41, 5))
    win = {}
    for rho in (10, 20):
        ok = [d for d in dts if abs(d) < 1000 / rho - 1]
        dwv, _ = run_trains([cp.votri_trains(rho, d) for d in ok])
        win[str(rho)] = {str(d): float(v) for d, v in zip(ok, dwv)}
    res["window_port"] = win
    us = list(np.arange(-80, -19.9, 2.5))
    dwc, an = voltage_clamp(us)
    res["voltage_clamp_port"] = dict(u=[float(u) for u in us], dw=dwc.tolist(), analytic=an.tolist())
    # VoTri numpy transcription for the same protocols (Octave values come from equivalence.json)
    par = (cp.A2d_PUB, cp.A3p_PUB, cp.tau_p, cp.tau_r, cp.tau_d)
    res["frequency_votri_numpy"] = {f"{r}|{d}": float(cp.votri_ref(r, d, par) - 0.5) for r in rhos for d in (10, -10)}
    res["window_votri_numpy"] = {str(rho): {str(d): float(cp.votri_ref(rho, d, par) - 0.5)
                                            for d in dts if abs(d) < 1000 / rho - 1} for rho in (10, 20)}
    eq = os.path.join(OUT, "equivalence.json")
    if os.path.isfile(eq):
        E = json.load(open(eq))
        if "votri_octave_R" in E:
            R = np.array(E["votri_octave_R"])
            res["frequency_votri_octave"] = {f"{r}|{d}": float(R[i, j] - 0.5) for i, r in enumerate(rhos)
                                             for j, d in enumerate((10, -10))}
    for k in ("frequency_port", "frequency_votri_numpy"):
        print(f"  {k}: " + "  ".join(f"{r}Hz:{res[k][f'{r}|10']:+.3f}/{res[k][f'{r}|-10']:+.3f}" for r in rhos), flush=True)
    print("  clamp: " + " ".join(f"{u:.0f}:{d:+.1e}" for u, d in zip(us, dwc)), flush=True)
    json.dump(res, open(os.path.join(OUT, "validation.json"), "w"), indent=2)
    print(f"wrote {OUT}/validation.json", flush=True)

    # ------------------------------------------------------------------ figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    im = ax[0, 0].imshow(ws10[0], aspect="auto", origin="lower", vmin=0, vmax=3, cmap="viridis",
                         extent=[0, ws10.shape[2] * se10 / 1000, 1, 500])
    ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("input index")
    ax[0, 0].set_title("RFdevelop.m setup through the port, seed 1 (10x amplitudes)\n"
                       "weights over time (Clopath's w_save plot, colour 0-3)", fontsize=9)
    fig.colorbar(im, ax=ax[0, 0], fraction=0.04)
    for i in range(len(seeds)):
        ax[0, 1].plot(np.arange(1, 501), wf10[i] + 3.3 * i, lw=0.8)
    for c in centres:
        ax[0, 1].axvline(c + 1, color="0.8", lw=0.5, zorder=0)
    ax[0, 1].set_yticks([3.3 * i + 1.5 for i in range(len(seeds))], [f"seed {s}" for s in seeds], fontsize=7)
    ax[0, 1].set_xlabel("input index (grey lines: centres of the 10 patterns)")
    ax[0, 1].set_title("Final weights after 100 s, 10 seeds (each offset by 3.3; range 0-3)", fontsize=9)
    im2 = ax[1, 0].imshow(ws1[0], aspect="auto", origin="lower", vmin=0, vmax=3, cmap="viridis",
                          extent=[0, ws1.shape[2] * se1 / 1000, 1, 500])
    ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("input index")
    ax[1, 0].set_title("Published amplitudes (1x), 1,000 s, seed 1", fontsize=9)
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.04)
    ax[1, 1].bar(np.arange(len(seeds)) - 0.2, a10["best_bump_weight_fraction"], 0.4, label="10x, 100 s")
    ax[1, 1].bar(np.arange(len(seeds)) + 0.2, a1["best_bump_weight_fraction"], 0.4, label="1x, 1,000 s")
    ax[1, 1].axhline(41 / 500, color="k", ls="--", lw=0.8, label="uniform weights (41/500)")
    ax[1, 1].set_xticks(range(len(seeds)), [str(s) for s in seeds]); ax[1, 1].set_xlabel("seed")
    ax[1, 1].set_ylabel("fraction of total weight\nwithin 2 sigma of the best bump")
    ax[1, 1].legend(fontsize=7); ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_title("Selectivity at the end of the run", fontsize=9)
    fig.savefig(os.path.join(OUT, "rf_development.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    ax[0].plot(us, dwc, "o-", ms=3, label="port (clamped u, 50 pre spikes at 2 Hz)")
    ax[0].plot(us, an, "k--", lw=0.8, label="analytic: (u-theta-)+ (-A_LTD + A_LTP (u-theta+)+) x 50")
    ax[0].axvline(-70.6, color="0.7", lw=0.6); ax[0].axvline(-45.3, color="0.7", lw=0.6)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].set_xlabel("clamped postsynaptic voltage u (mV)"); ax[0].set_ylabel("weight change")
    ax[0].set_title("Voltage dependence (cf. Clopath 2010 Fig. 1)", fontsize=9); ax[0].legend(fontsize=6)
    for j, d in enumerate((10, -10)):
        ax[1].plot(rhos, [res["frequency_port"][f"{r}|{d}"] / 0.5 * 100 for r in rhos], "o-", color=f"C{j}",
                   label=f"port, dt = {d:+d} ms")
        if "frequency_votri_octave" in res:
            ax[1].plot(rhos, [res["frequency_votri_octave"][f"{r}|{d}"] / 0.5 * 100 for r in rhos], "s--",
                       color=f"C{j}", mfc="white", label=f"VoTri.m in Octave, dt = {d:+d} ms")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xscale("symlog", linthresh=1)
    ax[1].set_xlabel("pairing frequency (Hz)"); ax[1].set_ylabel("weight change (% of w0 = 0.5)")
    ax[1].set_title("Frequency dependence, Sjostrom protocol (cf. Clopath 2010 Fig. 2)", fontsize=9)
    ax[1].legend(fontsize=6)
    for j, rho in enumerate(("10", "20")):
        d = sorted(win[rho], key=int)
        ax[2].plot([int(x) for x in d], [win[rho][x] / 0.5 * 100 for x in d], "o-", ms=3, color=f"C{j + 2}",
                   label=f"port, {rho} Hz")
        dv = res["window_votri_numpy"][rho]
        ax[2].plot([int(x) for x in d], [dv[x] / 0.5 * 100 for x in d], "s--", ms=3, mfc="white",
                   color=f"C{j + 2}", label=f"VoTri.m (NumPy transcription), {rho} Hz")
    ax[2].axhline(0, color="k", lw=0.5); ax[2].axvline(0, color="k", lw=0.5)
    ax[2].set_xlabel("dt = t_post - t_pre (ms)"); ax[2].set_ylabel("weight change (% of w0)")
    ax[2].set_title("Timing window (5 pairings x 15)", fontsize=9); ax[2].legend(fontsize=6)
    fig.savefig(os.path.join(OUT, "protocols.png"), dpi=130); plt.close(fig)
    print(f"wrote {OUT}/rf_development.png, protocols.png")


if __name__ == "__main__":
    main()
