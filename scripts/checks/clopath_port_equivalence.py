#!/usr/bin/env python3
"""Equivalence checks for the Clopath port (models/clopath_port.py).

1. RFdevelop.m's own setup (1 neuron, 500 inputs, 10 Gaussian patterns): the same input
   spike sequence and initial weights are run through
     (a) the NumPy float64 reference (RFDevelopRef, line by line from the .m files),
     (b) the batched torch version (ClopathNet) in float64 on CPU and on GPU,
     (c) the ORIGINAL aEIF.m and a copy of RFdevelop.m in GNU Octave, in which only the
         random-number lines are replaced by the same inputs/weights (diff saved),
     (d) the torch version in float32 on GPU (information only: where it drifts).
   Compared: membrane voltage at every step, spike times, weights every 100 steps.
2. Network with lateral inhibition (the MNIST extension): network_ref (NumPy, neuron by
   neuron) vs ClopathNet, several networks in one batch, plastic and STDP-off.
3. VoTri.m (Clopath's pairing-protocol code): votri_ref (NumPy transcription) vs the
   original VoTri.m in Octave, for the protocols used in the validation.

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/checks/clopath_port_equivalence.py [--nep 1000]
Octave: put octave-cli on PATH or set OCTAVE_CLI (one install without root:
micromamba create -p ENV -c conda-forge octave, then OCTAVE_CLI=ENV/bin/octave-cli). The Octave
comparisons are skipped if Octave is absent; they also need CLOPATH_MODELDB_DIR (see below).
Out: results/clopath_port_validation/equivalence.{json,png}, octave/ (modified .m + diff)
"""
import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
from models import clopath_port as cp  # noqa: E402

# Clopath et al.'s MATLAB code (ModelDB accession 144566: aEIF.m, RFdevelop.m, VoTri.m) is not redistributed here: download it and point
# CLOPATH_MODELDB_DIR at the folder. Only the Octave comparisons read it.
REF_DIR = os.environ.get("CLOPATH_MODELDB_DIR", "")
OUT = "results/clopath_port_validation"
SCR = os.environ.get("CLOPATH_SCRATCH", os.path.join(tempfile.gettempdir(), "clopath_port_scratch"))
OCTAVE = (os.environ.get("OCTAVE_CLI") or shutil.which("octave-cli")
          or os.path.join(SCR, "mamba/octave_env/bin/octave-cli"))
# conda's Octave cannot relocate its built-in paths under a long prefix: set OCTAVE_HOME
OCT_ENV = dict(os.environ, OCTAVE_HOME=os.path.dirname(os.path.dirname(OCTAVE)))
torch.set_grad_enabled(False)


# ----------------------------------------------------------------------
def rfdevelop_setup(seed, NEP=1000, Ni=500, EP=100, P=10, sigma=10, nu_in_max=0.015, nu_in_min=0.0001):
    """RFdevelop.m lines 20-59 with numpy's RNG in place of MATLAB's rand. Index 0 unused
    (MATLAB is 1-based); inputs[t] is `inp` at loop step t (t = 1..T; t = 1 is the
    pre-loop draw, overwritten before use as in the .m file)."""
    rng = np.random.default_rng(seed)
    T = EP * NEP
    p = np.zeros(T + 1, dtype=int)
    for n in range(1, NEP + 1):
        p[(n - 1) * EP + 1:n * EP + 1] = int(np.floor(P * rng.random())) + 1
    ind = np.arange(1, Ni + 1, dtype=float)
    gau = nu_in_min + nu_in_max * np.exp(-(ind - Ni / 2) ** 2 / (2 * sigma ** 2))
    gau = np.concatenate([gau, gau])
    pat = np.zeros((Ni, P))
    for jj in range(1, P + 1):
        mup = int(1 + (jj - 1) * Ni / P)
        pat[:, jj - 1] = gau[mup - 1:mup - 1 + Ni]
    w0 = rng.random(Ni) * 1.5 + 0.5
    inputs = np.zeros((T + 1, Ni), dtype=bool)
    for t in range(1, T + 1):
        inputs[t] = rng.random(Ni) < pat[:, p[t] - 1]
    centres = [int(np.argmax(pat[:, j])) for j in range(P)]   # 0-based input index of each bump
    return dict(T=T, Ni=Ni, p=p, pat=pat, w0=w0, inputs=inputs, centres=centres)


def run_numpy_ref(S, A3p=cp.A3p_RF, A2d=cp.A2d_RF):
    ref = cp.RFDevelopRef(S["w0"], A3p=A3p, A2d=A2d)
    T = S["T"]
    u = np.full(T + 1, np.nan); th = np.full(T + 1, np.nan)
    w_save = np.zeros((S["Ni"], T // 100 + 1))
    spikes = []
    for t in range(2, T + 1):
        if ref.step(S["inputs"][t]):
            spikes.append(t)
        u[t] = ref.u; th[t] = ref.theta
        w_save[:, t // 100] = ref.w          # w_save(:,floor(t/100)+1) = w
    return dict(u=u, theta=th, w_save=w_save, spikes=np.array(spikes), w=ref.w.copy())


def run_torch(S, device, dtype, A3p=cp.A3p_RF, A2d=cp.A2d_RF, fused=False):
    net = cp.ClopathNet(S["w0"].reshape(1, -1, 1), g=cp.G_RF, k_inh=0.0, A3p=A3p, A2d=A2d,
                        dtype=dtype, device=device, fused=fused)
    T = S["T"]
    X = torch.as_tensor(S["inputs"], device=device)
    u = torch.full((T + 1,), float("nan"), dtype=torch.float64, device=device)
    th = torch.full((T + 1,), float("nan"), dtype=torch.float64, device=device)
    sp = torch.zeros(T + 1, dtype=torch.bool, device=device)
    w_save = torch.zeros((S["Ni"], T // 100 + 1), dtype=torch.float64, device=device)
    for t in range(2, T + 1):
        s = net.step(X[t])
        u[t] = net.u[0, 0]; th[t] = net.theta[0, 0]; sp[t] = s[0, 0]
        w_save[:, t // 100] = net.W[0, :, 0]
    return dict(u=u.cpu().numpy(), theta=th.cpu().numpy(), w_save=w_save.cpu().numpy(),
                spikes=np.nonzero(sp.cpu().numpy())[0], w=net.W[0, :, 0].double().cpu().numpy())


def compare(A, B, label):
    sa, sb = set(A["spikes"].tolist()), set(B["spikes"].tolist())
    same_spikes = sa == sb
    first_diff = None
    if not same_spikes:
        first_diff = int(min(sa ^ sb))
    du = np.nanmax(np.abs(A["u"] - B["u"]))
    # voltage difference before the first spike-time difference (if any)
    lim = first_diff if first_diff is not None else len(A["u"])
    du_pre = float(np.nanmax(np.abs(A["u"][:lim] - B["u"][:lim]))) if lim > 2 else float("nan")
    dw = float(np.max(np.abs(A["w_save"] - B["w_save"])))
    dth = float(np.nanmax(np.abs(A["theta"] - B["theta"])))
    out = dict(label=label, n_spikes_a=len(sa), n_spikes_b=len(sb), identical_spike_times=same_spikes,
               first_spike_time_difference_ms=first_diff, max_abs_du_mV=float(du),
               max_abs_du_before_first_difference_mV=du_pre, max_abs_dw_wsave=dw,
               max_abs_dtheta=dth, bitwise_identical_u=bool(np.array_equal(A["u"][2:], B["u"][2:])),
               bitwise_identical_w=bool(np.array_equal(A["w_save"], B["w_save"])))
    print(f"  {label}: spikes {len(sa)} vs {len(sb)}, identical={same_spikes}, "
          f"max|du|={du:.3e} mV, max|dw|={dw:.3e}, max|dtheta|={dth:.3e}, "
          f"bitwise u={out['bitwise_identical_u']} w={out['bitwise_identical_w']}", flush=True)
    return out


# ----------------------------------------------------------------------
def octave_available():
    if not (os.path.isfile(OCTAVE) and os.access(OCTAVE, os.X_OK)):
        return False
    gone = [m for m in ("aEIF.m", "RFdevelop.m", "VoTri.m") if not os.path.isfile(os.path.join(REF_DIR, m))]
    if gone:
        sys.exit(f"ERROR: Octave is installed but {', '.join(gone)} not found in CLOPATH_MODELDB_DIR={REF_DIR!r}.\n"
                 "Clopath et al.'s MATLAB code (ModelDB accession 144566: aEIF.m, RFdevelop.m, VoTri.m) is not\nredistributed here. Download it from https://modeldb.science/144566 and set\nCLOPATH_MODELDB_DIR to the folder that holds the three .m files.")
    return True


def read_m(name):
    return open(os.path.join(REF_DIR, name), "rb").read().decode("latin-1").replace("\r\n", "\n")


def make_rfdevelop_fixed():
    """RFdevelop.m with only the random-number lines replaced by loaded data, the plot
    section replaced by a save, and three trace-recording lines added."""
    src = read_m("RFdevelop.m")
    rep = [
        ("NEP       = 1000;", "NEP       = NEP_SET;  % [port check] was 1000"),
        ("w       = rand(Ni,n_neur)*1.5+0.5;", "w       = W0;  % [port check] was rand(Ni,n_neur)*1.5+0.5"),
        ("inp      = (rand(Ni,1)<pat(:,p(1)));", "inp      = INP(:,1);  % [port check] was (rand(Ni,1)<pat(:,p(1)))"),
        ("    inp = (rand(Ni,1)<pat(:,p(t))); % input pattern",
         "    inp = INP(:,t); % [port check] was (rand(Ni,1)<pat(:,p(t)))"),
        ("    u1ss = u1s; % save the voltage 2ms before for the update order\n",
         "    u1ss = u1s; % save the voltage 2ms before for the update order\n"
         "    U_TRACE(t) = u; TH_TRACE(t) = theta; C_TRACE(t) = counter;  % [port check] recording\n"),
    ]
    out = src
    for a, b_ in rep:
        if out.count(a) != 1:
            raise RuntimeError(f"pattern not found exactly once in RFdevelop.m: {a!r}")
        out = out.replace(a, b_)
    head = ("% [port check] inputs and initial weights from Python (scripts/checks/clopath_port_equivalence.py)\n"
            "load('rf_inputs.mat'); INP = logical(INP); NEP_SET = double(NEP_SET);\n"
            "U_TRACE = nan(1, EP_T); TH_TRACE = nan(1, EP_T); C_TRACE = nan(1, EP_T);\n")
    k = out.index("% Plot")
    out = head + out[:k] + "% [port check] plot replaced by save\nsave('-mat7-binary', 'rf_out.mat', 'U_TRACE', 'TH_TRACE', 'C_TRACE', 'w_save', 'w');\n"
    return src, out


def run_octave_rfdevelop(S, work):
    import scipy.io as sio
    os.makedirs(work, exist_ok=True)
    shutil.copy(os.path.join(REF_DIR, "aEIF.m"), os.path.join(work, "aEIF.m"))   # original, unmodified
    src, mod = make_rfdevelop_fixed()
    open(os.path.join(work, "RFdevelop_portcheck.m"), "w").write(mod)
    sio.savemat(os.path.join(work, "rf_inputs.mat"),
                dict(INP=S["inputs"][1:].T.astype(np.uint8), W0=S["w0"].reshape(-1, 1),
                     NEP_SET=float(S["T"] // 100), EP_T=float(S["T"])), do_compression=False)
    t0 = time.time()
    r = subprocess.run([OCTAVE, "--no-gui", "--quiet", "--eval",
                        "RFdevelop_portcheck"],
                       cwd=work, capture_output=True, text=True, env=OCT_ENV)
    if not os.path.isfile(os.path.join(work, "rf_out.mat")):
        raise RuntimeError("Octave run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    o = sio.loadmat(os.path.join(work, "rf_out.mat"))
    T = S["T"]
    u = np.concatenate([[np.nan], o["U_TRACE"].ravel()])
    th = np.concatenate([[np.nan], o["TH_TRACE"].ravel()])
    c = np.concatenate([[np.nan], o["C_TRACE"].ravel()])
    spikes = np.nonzero(c == 1)[0]
    w_save = o["w_save"][:, :T // 100 + 1]
    print(f"  Octave RFdevelop run: {time.time() - t0:.0f} s", flush=True)
    return dict(u=u, theta=th, w_save=w_save, spikes=spikes, w=o["w"].ravel()), src, mod


VOTRI_DRIVER = """
par = [14e-5, 8e-5, 7, 15, 10];
rhos = [0.1 1 5 10 20 40 50]; dts = [10 -10];
R = zeros(numel(rhos), numel(dts));
for i = 1:numel(rhos), for j = 1:numel(dts), R(i,j) = VoTri(rhos(i), dts(j), par); end, end
dtw = -40:5:40; RW = zeros(2, numel(dtw)); rw = [10 20];
for i = 1:2, for j = 1:numel(dtw)
  if abs(dtw(j)) < 1000/rw(i) - 1, RW(i,j) = VoTri(rw(i), dtw(j), par); else RW(i,j) = NaN; end
end, end
save('-mat7-binary', 'votri_out.mat', 'R', 'RW', 'rhos', 'dts', 'dtw', 'rw');
"""


def run_octave_votri(work):
    import scipy.io as sio
    os.makedirs(work, exist_ok=True)
    shutil.copy(os.path.join(REF_DIR, "aEIF.m"), os.path.join(work, "aEIF.m"))
    shutil.copy(os.path.join(REF_DIR, "VoTri.m"), os.path.join(work, "VoTri.m"))   # original
    open(os.path.join(work, "votri_driver.m"), "w").write(VOTRI_DRIVER)
    t0 = time.time()
    r = subprocess.run([OCTAVE, "--no-gui", "--quiet", "votri_driver.m"], cwd=work,
                       capture_output=True, text=True, env=OCT_ENV)
    if not os.path.isfile(os.path.join(work, "votri_out.mat")):
        raise RuntimeError("Octave VoTri failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    print(f"  Octave VoTri runs: {time.time() - t0:.0f} s", flush=True)
    return sio.loadmat(os.path.join(work, "votri_out.mat"))


# ----------------------------------------------------------------------
def network_test(device):
    """Network with inhibition and the -100 mV floor: network_ref vs ClopathNet (batches of
    networks; unfused and, on GPU, the fused production update)."""
    rng = np.random.default_rng(7)
    Ni, N, T = 784, 12, 1200
    # MNIST-like Poisson input: two 350-ms images at up to 63.75 Hz + 150 ms silence each
    imgs = rng.random((2, Ni)) * (rng.random((2, Ni)) < 0.2)
    inputs = np.zeros((T, Ni), dtype=bool)
    for k in range(2):
        for t in range(350):
            inputs[k * 500 + t] = rng.random(Ni) < imgs[k] * 63.75 / 1000
    W0 = rng.random((Ni, N)) * 1.5 + 0.5
    cells = [dict(g=0.4, k=0.0, A3p=cp.A3p_RF, A2d=cp.A2d_RF, plastic=True),
             dict(g=0.4, k=8.0, A3p=cp.A3p_RF, A2d=cp.A2d_RF, plastic=True),
             dict(g=0.8, k=20.0, A3p=cp.A3p_PUB, A2d=cp.A2d_PUB, plastic=True),
             dict(g=1.2, k=60.0, A3p=cp.A3p_RF, A2d=cp.A2d_RF, plastic=True),
             dict(g=0.8, k=20.0, A3p=cp.A3p_RF, A2d=cp.A2d_RF, plastic=False)]
    res = []
    X = torch.as_tensor(inputs, device=device)
    refs = {i: cp.network_ref(W0, inputs, c["g"], c["k"], c["A3p"], c["A2d"], plastic=c["plastic"],
                              record_every=0, u_floor=cp.U_FLOOR_INH) for i, c in enumerate(cells)}
    variants = [(p, False) for p in (True, False)]
    if str(device).startswith("cuda"):
        variants.insert(1, (True, True))
    for plastic, fused in variants:
        idx = [i for i, c in enumerate(cells) if c["plastic"] == plastic]
        net = cp.ClopathNet(np.stack([W0] * len(idx)), g=[cells[i]["g"] for i in idx],
                            k_inh=[cells[i]["k"] for i in idx], A3p=[cells[i]["A3p"] for i in idx],
                            A2d=[cells[i]["A2d"] for i in idx], plastic=plastic,
                            dtype=torch.float64, device=device, fused=fused, u_floor=cp.U_FLOOR_INH)
        U = []; SP = []
        for t in range(T):
            SP.append(net.step(X[t]).cpu().numpy()); U.append(net.u.cpu().numpy())
        U = np.array(U); SP = np.array(SP)
        for bi, i in enumerate(idx):
            c = cells[i]
            ref = refs[i]
            same = bool(np.array_equal(ref["spikes"], SP[:, bi, :]))
            du = float(np.max(np.abs(ref["u"] - U[:, bi, :])))
            dW = float(np.max(np.abs(ref["W"] - net.W[bi].cpu().numpy())))
            nsp = int(ref["spikes"].sum())
            umin = float(ref["u"].min())
            print(f"  network cell {c} (batch {bi}, fused={fused}): spikes {nsp}, identical={same}, "
                  f"max|du|={du:.2e}, max|dW|={dW:.2e}, min u {umin:.1f} mV", flush=True)
            res.append(dict(cell={k: (float(v) if not isinstance(v, bool) else v) for k, v in c.items()},
                            fused_update=fused, n_spikes=nsp, identical_spike_trains=same, max_abs_du_mV=du,
                            max_abs_dW=dW, min_u_mV=umin, floor_reached=bool(umin <= cp.U_FLOOR_INH),
                            W_changed=float(np.max(np.abs(ref["W"] - W0)))))
    return res


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nep", type=int, default=1000, help="epochs of 100 ms (RFdevelop: 1000 = 100 s)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    report = dict(date=time.strftime("%Y-%m-%d %H:%M"), nep=args.nep, seed=args.seed,
                  torch=torch.__version__, numpy=np.__version__)
    dev = "cuda:0" if torch.cuda.is_available() else None

    print(f"[1] RFdevelop setup, {args.nep * 100 / 1000:.0f} s, seed {args.seed}", flush=True)
    S = rfdevelop_setup(args.seed, NEP=args.nep)
    t0 = time.time(); A = run_numpy_ref(S); report["numpy_ref_seconds"] = time.time() - t0
    print(f"  numpy reference: {len(A['spikes'])} spikes, {report['numpy_ref_seconds']:.0f} s", flush=True)
    comps = []
    t0 = time.time(); Bc = run_torch(S, "cpu", torch.float64)
    print(f"  torch cpu float64: {time.time() - t0:.0f} s", flush=True)
    comps.append(compare(A, Bc, "numpy reference vs torch CPU float64"))
    if dev:
        t0 = time.time(); Bg = run_torch(S, dev, torch.float64)
        print(f"  torch gpu float64: {time.time() - t0:.0f} s", flush=True)
        comps.append(compare(A, Bg, "numpy reference vs torch GPU float64"))
        t0 = time.time(); Bfu = run_torch(S, dev, torch.float64, fused=True)
        print(f"  torch gpu float64 fused update: {time.time() - t0:.0f} s", flush=True)
        comps.append(compare(A, Bfu, "numpy reference vs torch GPU float64, fused weight update (MNIST production path)"))
        Bf = run_torch(S, dev, torch.float32)
        comps.append(compare(A, Bf, "numpy reference vs torch GPU float32 (information)"))
    O = None
    if octave_available():
        try:
            O, src, mod = run_octave_rfdevelop(S, os.path.join(SCR, "octave_rf_check"))
            comps.append(compare(O, A, "Octave (original aEIF.m + RFdevelop.m, inputs fixed) vs numpy reference"))
            if dev:
                comps.append(compare(O, Bg, "Octave vs torch GPU float64"))
            od = os.path.join(OUT, "octave"); os.makedirs(od, exist_ok=True)
            open(os.path.join(od, "RFdevelop_portcheck.m"), "w").write(mod)
            open(os.path.join(od, "RFdevelop_portcheck.diff"), "w").write("".join(difflib.unified_diff(
                src.splitlines(True), mod.splitlines(True), "RFdevelop.m (ModelDB 144566)", "RFdevelop_portcheck.m")))
            report["octave"] = dict(binary=OCTAVE, version=subprocess.run(
                [OCTAVE, "--version"], capture_output=True, text=True, env=OCT_ENV).stdout.splitlines()[0])
        except Exception as e:  # report, do not hide
            report["octave_error"] = f"{type(e).__name__}: {e}"
            print("  Octave check failed:", report["octave_error"][:500], flush=True)
    else:
        report["octave"] = "not available (set OCTAVE_CLI)"
    report["rfdevelop_comparisons"] = comps
    report["rfdevelop_n_spikes"] = int(len(A["spikes"]))

    print("[2] network with lateral inhibition and -100 mV floor: network_ref vs ClopathNet (float64)", flush=True)
    report["network_comparisons"] = network_test(dev or "cpu")

    print("[3] VoTri.m: numpy transcription vs original in Octave", flush=True)
    par = (14e-5, 8e-5, 7, 15, 10)
    rhos = [0.1, 1, 5, 10, 20, 40, 50]
    vt = {f"{r}|{d}": cp.votri_ref(r, d, par) for r in rhos for d in (10, -10)}
    report["votri_numpy"] = vt
    if octave_available():
        try:
            V = run_octave_votri(os.path.join(SCR, "octave_votri_check"))
            diffs = []
            for i, r in enumerate(V["rhos"].ravel()):
                for j, d in enumerate(V["dts"].ravel()):
                    diffs.append(abs(V["R"][i, j] - vt[f"{r:g}|{int(d)}"]))
            report["votri_octave_R"] = V["R"].tolist()
            report["votri_octave_window"] = dict(rw=V["rw"].ravel().tolist(), dtw=V["dtw"].ravel().tolist(),
                                                 RW=np.where(np.isnan(V["RW"]), None, V["RW"]).tolist())
            report["votri_max_abs_diff_numpy_vs_octave"] = float(max(diffs))
            print(f"  VoTri numpy vs Octave: max |dw| = {max(diffs):.3e}", flush=True)
        except Exception as e:
            report["votri_octave_error"] = f"{type(e).__name__}: {e}"
            print("  Octave VoTri failed:", report["votri_octave_error"][:500], flush=True)

    json.dump(report, open(os.path.join(OUT, "equivalence.json"), "w"), indent=2, default=float)
    print(f"wrote {OUT}/equivalence.json", flush=True)

    # figure: voltage and weight traces, reference vs torch vs Octave
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), layout="constrained")
    t = np.arange(len(A["u"]))
    sl = slice(2, 1500)
    ax[0].plot(t[sl], A["u"][sl], lw=1.6, color="0.6", label="NumPy reference")
    if dev:
        ax[0].plot(t[sl], Bg["u"][sl], lw=0.7, color="C0", label="torch GPU float64")
    if O is not None:
        ax[0].plot(t[sl], O["u"][sl], lw=0.6, ls="--", color="C3", label="Octave (original .m)")
    ax[0].set_ylabel("u (mV)"); ax[0].set_xlabel("time (ms)"); ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("Membrane voltage, first 1.5 s of RFdevelop.m's setup (same inputs)", fontsize=9)
    syn = S["centres"][:3]
    for k, i in enumerate(syn):
        ax[1].plot(A["w_save"][i], color=f"C{k}", lw=1.6, alpha=0.5, label=f"input {i + 1}: reference")
        if dev:
            ax[1].plot(Bg["w_save"][i], color=f"C{k}", lw=0.6, ls="--", label=f"input {i + 1}: torch GPU")
    ax[1].set_ylabel("w"); ax[1].set_xlabel("time (100 ms)"); ax[1].legend(fontsize=6, ncol=3)
    ax[1].set_title("Weights at three pattern centres, every 100 ms (w_save)", fontsize=9)
    if dev:
        ax[2].semilogy(t[2:], np.abs(Bg["u"][2:] - A["u"][2:]) + 1e-18, lw=0.5, label="|u torch GPU f64 - u reference|")
        ax[2].semilogy(t[2:], np.abs(Bf["u"][2:] - A["u"][2:]) + 1e-18, lw=0.5, alpha=0.6, label="|u torch GPU f32 - u reference|")
    if O is not None:
        ax[2].semilogy(t[2:], np.abs(O["u"][2:] - A["u"][2:]) + 1e-18, lw=0.5, label="|u Octave - u reference|")
    ax[2].set_ylabel("|difference| (mV)"); ax[2].set_xlabel("time (ms)"); ax[2].legend(fontsize=7)
    ax[2].set_title("Voltage differences over the whole run (1e-18 added for the log scale)", fontsize=9)
    fig.savefig(os.path.join(OUT, "equivalence.png"), dpi=130)
    print(f"wrote {OUT}/equivalence.png")


if __name__ == "__main__":
    main()
