"""POSITIVE CONTROL — does our voltage-STDP code reproduce canonical Clopath 2010
behaviour? (The paper's positive control, Appendix D.)

We drive the **exact** plasticity step used by the failing FC runs —
`models.fc_stdp_layer._make_adex_voltage_step` — with classic spike-pairing
protocols and check for two canonical signatures:

  (A) STDP timing window:  pre-before-post -> LTP, post-before-pre -> LTD.
  (C) Sjostrom/Clopath frequency dependence: at fixed (pre->post) timing, low
      pairing frequency -> net depression, high frequency -> net potentiation
      (the BCM-like LTD->LTP crossover that voltage/triplet rules capture and
      plain pair-STDP cannot).

The postsynaptic spike is evoked by a strong "teacher" synapse (mimicking the
experimental depolarising current pulse); Delta-w is measured on a separate weak
"plastic" synapse. Parameters are the repo's actual voltage config (theta_ltp=
-45.3, theta_ltd=-70.6 = Clopath 2010's published thresholds; tau_+=7, tau_-=10).

We run each protocol at TWO amplitude ratios:
  - "Clopath" LTP-dominant (A_LTP:A_LTD = 14:8, his published ratio)
  - "B3" the task-tuned config (A+:A- = 1:3, LTD-dominant — INVERTED)
to show (a) the code reproduces Clopath when given Clopath's ratio [implementation
is correct], and (b) the B3 task config sits in an LTD-dominant regime [why FC fails].

Run:  PYTHONPATH=. python scripts/checks/clopath_positive_control.py
Out:  results/clopath_positive_control.{md,png}
"""
import math, json, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from models.fc_stdp_layer import _make_adex_voltage_step

torch.set_grad_enabled(False)
DT = 1.0  # ms — matches the FC sim

# ---- repo voltage config (from a saved medical-matrix voltage cell config.yaml) ----
NEU = dict(v_thresh=-52.0, tau_v=100.0, refrac_time=5,
           delta_T=2.0, v_T_offset=-2.0, a_adapt=0.0, b_adapt=0.1, tau_w_adapt=144.0,
           I_sp=0.2, tau_z=40.0)
STD = dict(tau_pre=16.8, tau_plus=7.0, tau_minus=10.0, theta_ltp=-45.3, theta_ltd=-70.6)
# Conditions to compare. v_rest is decisive: Clopath sets v_rest = θ₋ = -70.6 so
# there is NO depression at rest; the repo's FC task config uses v_rest = -65
# (> θ₋), which injects tonic LTD on every pre-spike — a mechanistic contributor
# to the voltage collapse. Amplitude ratio: Clopath's 14:8 (LTP-dominant) vs the
# B3 task config's 1:3 (LTD-dominant, inverted).
CONDITIONS = {
    "Clopath 2010 (v_rest=θ₋, A 14:8)": dict(v_rest=-70.6, A_plus=0.0021, A_minus=0.0012),
    "B3 amps, Clopath v_rest (A 1:3)":  dict(v_rest=-70.6, A_plus=0.0010, A_minus=0.0030),
    "B3 task config (v_rest=−65, A 1:3)": dict(v_rest=-65.0, A_plus=0.0010, A_minus=0.0030),
}

def build_step(v_rest, A_plus, A_minus):
    """The repo's real Clopath step, no adaptive threshold / no homeostasis."""
    return _make_adex_voltage_step(
        decay_v=math.exp(-DT/NEU["tau_v"]), v_rest=v_rest, v_reset=v_rest,
        v_thresh=NEU["v_thresh"], refrac_time=NEU["refrac_time"], inh_weight=0.0,
        theta_plus=0.0, decay_theta=1.0,
        delta_T=NEU["delta_T"], v_T=NEU["v_thresh"]+NEU["v_T_offset"],
        a_adapt=NEU["a_adapt"], b_adapt=NEU["b_adapt"],
        decay_w_adapt=math.exp(-DT/NEU["tau_w_adapt"]), I_sp=NEU["I_sp"],
        decay_z=math.exp(-DT/NEU["tau_z"]),
        decay_x_pre_bar=math.exp(-DT/STD["tau_pre"]),
        decay_u_plus=math.exp(-DT/STD["tau_plus"]),
        decay_u_minus=math.exp(-DT/STD["tau_minus"]),
        A_plus=A_plus, A_minus=A_minus,
        theta_ltp=STD["theta_ltp"], theta_ltd=STD["theta_ltd"],
    )

def simulate(step, v_rest, A_minus, pre_times, teacher_times, T, w_plastic=1.0, w_teacher=30.0):
    """1 post-neuron, 2 pre (plastic idx0, teacher idx1). Returns Δw on plastic,
    and number of post-spikes (to confirm the teacher actually evoked spikes)."""
    f = lambda x: torch.tensor(x, dtype=torch.float32)
    v = f([v_rest]); refrac = f([0.0]); theta = f([0.0])
    w_adapt = f([0.0]); z = f([0.0])
    x_pre_bar = f([0.0, 0.0]); u_plus_bar = f([v_rest]); u_minus_bar = f([v_rest])
    u_homeo = f([0.0]); dw = torch.zeros(2, 1); spike_count = f([0.0])
    W = f([[w_plastic], [w_teacher]]); a_minus_vec = f([A_minus])
    pre_set, teach_set = set(pre_times), set(teacher_times)
    for t in range(T):
        pre = torch.zeros(2)
        if t in pre_set:  pre[0] = 1.0
        if t in teach_set: pre[1] = 1.0
        (v, refrac, theta, w_adapt, z, x_pre_bar, u_plus_bar, u_minus_bar,
         u_homeo, dw, spike_count) = step(v, refrac, theta, w_adapt, z,
            x_pre_bar, u_plus_bar, u_minus_bar, u_homeo, dw, spike_count,
            W, pre, True, a_minus_vec)
    return float(dw[0, 0]), int(spike_count.item())

# ---------- Protocol A: STDP timing window ----------
def stdp_window(cond, dts, n_pairs=15, period=200, warmup=100):
    step = build_step(cond["v_rest"], cond["A_plus"], cond["A_minus"])
    out = []
    for dt in dts:
        base = warmup + max(0, -dt)            # keep all events >= 0
        pre_t = [base + k*period for k in range(n_pairs)]
        teach_t = [p + dt for p in pre_t]      # dt>0: pre before post (LTP expected)
        T = pre_t[-1] + max(period, abs(dt)+50)
        dw, nsp = simulate(step, cond["v_rest"], cond["A_minus"], pre_t, teach_t, T)
        out.append((dt, dw/n_pairs, nsp))
    return out

# ---------- Protocol C: Sjostrom frequency dependence ----------
def freq_dependence(cond, freqs, dt=8, n_pairs=10, warmup=100):
    step = build_step(cond["v_rest"], cond["A_plus"], cond["A_minus"])
    out = []
    for fhz in freqs:
        period = int(round(1000.0/fhz))
        pre_t = [warmup + k*period for k in range(n_pairs)]
        teach_t = [p + dt for p in pre_t]
        T = pre_t[-1] + period + 50
        dw, nsp = simulate(step, cond["v_rest"], cond["A_minus"], pre_t, teach_t, T)
        out.append((fhz, dw/n_pairs, nsp))
    return out

def main():
    dts = list(range(-40, 41, 4))
    freqs = [0.5, 1, 2, 5, 10, 20, 40, 50]
    res = {"window": {}, "freq": {}}
    print("=== Protocol A: STDP timing window (Δw per pairing) ===")
    for name, cond in CONDITIONS.items():
        w = stdp_window(cond, dts)
        res["window"][name] = w
        ltp = [d for (x, d, n) in w if x > 0]; ltd = [d for (x, d, n) in w if x < 0]
        sign = "LTP✓" if np.mean(ltp) > 0 else "no-LTP"
        print(f"  {name:34s}  Δw(pre→post)={np.mean(ltp):+.2e} [{sign}]  "
              f"Δw(post→pre)={np.mean(ltd):+.2e}  spk/run≈{w[len(w)//2][2]}")
    print("=== Protocol C: frequency dependence (Δw per pairing, Δt=+8ms) ===")
    for name, cond in CONDITIONS.items():
        fr = freq_dependence(cond, freqs)
        res["freq"][name] = fr
        lo = fr[0][1]; hi = fr[-1][1]
        print(f"  {name:34s}  Δw@{freqs[0]}Hz={lo:+.2e} → Δw@{freqs[-1]}Hz={hi:+.2e}  "
              f"(↑ with freq: {'YES' if hi>lo else 'no'})")

    # ---- voltage-gating curves (the code's exact normalised gates vs post voltage) ----
    v_scale = abs(NEU["v_thresh"] - (-70.6))   # |v_thresh - v_rest| as in the step
    us = np.linspace(-75, -40, 200)
    ltd_gate = np.clip(us - STD["theta_ltd"], 0, v_scale) / v_scale
    ltp_gate = (np.clip(us - STD["theta_ltp"], 0, v_scale) / v_scale) * ltd_gate
    res["gates"] = dict(u=us.tolist(), ltd=ltd_gate.tolist(), ltp=ltp_gate.tolist())
    print("=== Voltage gates (mechanism) ===")
    for vr in (-70.6, -65.0):
        g = float(np.clip(vr - STD["theta_ltd"], 0, v_scale) / v_scale)
        print(f"  resting LTD gate at v_rest={vr:6.1f}: {g:.2f}  "
              f"({'no tonic LTD' if g < 1e-6 else 'TONIC LTD on every pre-spike'})")

    # ---- figure ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))
        # (A) canonical window — the two Clopath-v_rest conditions (fine scale)
        canon = ["Clopath 2010 (v_rest=θ₋, A 14:8)", "B3 amps, Clopath v_rest (A 1:3)"]
        for name in canon:
            w = res["window"][name]
            xs = [x for x, d, n in w]; ys = [d for x, d, n in w]
            ax[0].plot(xs, ys, "-o", ms=3, label=name)
        ax[0].axhline(0, color="k", lw=.6); ax[0].axvline(0, color="k", lw=.6, ls=":")
        ax[0].annotate("LTP", (22, max(res["window"][canon[0]][i][1] for i in range(12, 21))*0.8),
                       color="tab:blue", fontsize=9)
        ax[0].set_xlabel("Δt = t_post − t_pre  (ms)"); ax[0].set_ylabel("Δw per pairing")
        ax[0].set_title("(A) Implementation reproduces Clopath STDP window\n"
                        "pre→post = LTP, post→pre = LTD  ✓")
        ax[0].legend(fontsize=8)
        # (B) B3-task config window — pure depression (coarse scale)
        w = res["window"]["B3 task config (v_rest=−65, A 1:3)"]
        xs = [x for x, d, n in w]; ys = [d for x, d, n in w]
        ax[1].plot(xs, ys, "-o", ms=3, color="tab:green")
        ax[1].axhline(0, color="k", lw=.6); ax[1].axvline(0, color="k", lw=.6, ls=":")
        ax[1].set_xlabel("Δt = t_post − t_pre  (ms)"); ax[1].set_ylabel("Δw per pairing")
        ax[1].set_title("(B) Same code, B3 TASK config (v_rest=−65, A 1:3)\n"
                        "→ pure depression, no LTP lobe (the FC collapse)")
        # (C) gates
        ax = ax  # keep
        g = res["gates"]
        ax[2].plot(g["u"], g["ltd"], "-", color="tab:red", label="LTD gate  [ū₋−θ₋]₊")
        ax[2].plot(g["u"], g["ltp"], "-", color="tab:green", label="LTP gate  [v−θ₊]₊·[ū₊−θ₋]₊")
        ax[2].axvline(STD["theta_ltd"], color="tab:red", ls=":", lw=.8)
        ax[2].axvline(STD["theta_ltp"], color="tab:green", ls=":", lw=.8)
        ax[2].annotate("θ₋=−70.6", (STD["theta_ltd"], .92), fontsize=7, color="tab:red")
        ax[2].annotate("θ₊=−45.3", (STD["theta_ltp"], .92), fontsize=7, color="tab:green")
        ax[2].axvline(-70.6, color="k", ls="--", lw=.8); ax[2].annotate("v_rest Clopath", (-70.6, .05), fontsize=7, rotation=90)
        ax[2].axvline(-65.0, color="gray", ls="--", lw=.8); ax[2].annotate("v_rest B3-task → tonic LTD", (-65.0, .25), fontsize=7, rotation=90)
        ax[2].set_xlabel("post membrane voltage u (mV)"); ax[2].set_ylabel("normalised plasticity gate")
        ax[2].set_title("(C) Clopath voltage gates (mechanism)\nB3-task v_rest sits ABOVE θ₋ → depression at rest")
        ax[2].legend(fontsize=7, loc="center left")
        fig.suptitle("Clopath positive control — repo's _make_adex_voltage_step "
                     "(θ₊=−45.3, θ₋=−70.6 = Clopath 2010)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig("results/clopath_positive_control.png", dpi=130)
        print("wrote results/clopath_positive_control.png")
    except Exception as e:
        print("figure skipped:", type(e).__name__, str(e)[:80])

    out_json = {"gates": res["gates"]}
    for k in ("window", "freq"):
        out_json[k] = {n: [[float(a), float(b), int(c)] for (a, b, c) in v]
                       for n, v in res[k].items()}
    json.dump(out_json, open("results/clopath_positive_control.json", "w"), indent=2)
    print("wrote results/clopath_positive_control.json")
    return res

if __name__ == "__main__":
    main()
