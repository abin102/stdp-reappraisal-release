"""Full Clopath-2010 curve reproduction — independent + primary implementation overlay.

Goal: implement Clopath's voltage-STDP equations independently (this file) and
check we reproduce his published curves, then show the repo's PRODUCTION plasticity
(`models.fc_stdp_layer._make_adex_voltage_step`) reproduces the SAME curves at
matched parameters. Together: "our equations match the paper" + "our production
code matches our equations" ⇒ the FC failure is not an implementation bug.

Three canonical curves (Clopath 2010 / its Brian2 reference implementation):
  (A) voltage dependence  — Δw vs clamped post voltage         (Fig 1)
  (B) STDP timing window  — Δw vs Δt                           (Fig 2a)
  (C) frequency dependence— Δw vs pairing rate                 (Fig 2b)

Standalone equations (full two-threshold Clopath form, = the repo's form):
  LTD (on pre-spike):  Δw −= A_LTD · [ū_d − θ₋]₊
  LTP (continuous):    Δw += A_LTP · x̄_pre · [u − θ₊]₊ · [ū_p − θ₋]₊
where ū_d, ū_p are low-pass post voltages (τ_ltd, τ_ltp), x̄_pre the pre trace.
Fig-2b protocol from Brian2 `Clopath_et_al_2010_no_homeostasis.py`: rates
1–50 Hz, 15 reps, Δt=+10 ms. Run: PYTHONPATH=. python scripts/figures/clopath_curve_reproduction.py
Out: results/clopath_curve_reproduction.{json,pdf,png} + $FIG_OUT_DIR/fig6_clopath_curves.{pdf,png} (default figures/)
(Figure A.1 of the paper) + console summary. --plot-only redraws the figure from the saved json.
"""
import math, json, numpy as np
import torch
torch.set_grad_enabled(False)
from models.fc_stdp_layer import _make_adex_voltage_step

DT = 1.0  # ms
# Clopath params (Brian2 Fig-2b reference set; θ₊ is Clopath's published LTP thresh)
PAR = dict(v_rest=-70.0, v_thresh=-50.0, theta_low=-70.0, theta_high=-45.3,
           tau_ltd=40.0, tau_ltp=30.0, tau_x=15.0, A_LTD=1.5e-4, A_LTP=1.5e-2,
           spike_peak=0.0, spike_dur=2)

# ----------------------- STANDALONE Clopath rule -----------------------
def _run_standalone(pre_times, u_trace, par, full_form=True):
    du_d = math.exp(-DT/par["tau_ltd"]); du_p = math.exp(-DT/par["tau_ltp"])
    dx = math.exp(-DT/par["tau_x"])
    ub_d = par["v_rest"]; ub_p = par["v_rest"]; xbar = 0.0
    pre = set(pre_times); dw = 0.0
    for t in range(len(u_trace)):
        ut = u_trace[t]
        ub_d = ub_d*du_d + (1-du_d)*ut
        ub_p = ub_p*du_p + (1-du_p)*ut
        xbar = xbar*dx + (1.0 if t in pre else 0.0)
        gate = max(ub_p - par["theta_low"], 0.0)
        if full_form:
            gate *= max(ut - par["theta_high"], 0.0)
        dw += par["A_LTP"] * xbar * gate
        if t in pre:
            dw -= par["A_LTD"] * max(ub_d - par["theta_low"], 0.0)
    return dw

def _u_with_spikes(T, spike_times, par):
    u = np.full(T, par["v_rest"]);
    for ts in spike_times:
        for k in range(par["spike_dur"]):
            if 0 <= ts+k < T: u[ts+k] = par["spike_peak"]
    return u

def sa_voltage(par, us=np.linspace(-72, -40, 17), n_pre=20, isi=20, warm=50):
    """Clamp post to fixed u, present pre-spikes, measure Δw — Fig 1."""
    out = []
    for uc in us:
        T = warm + n_pre*isi + warm
        u = np.full(T, uc); pre = [warm + k*isi for k in range(n_pre)]
        out.append((float(uc), _run_standalone(pre, u, par)))
    return out

def sa_window(par, dts=range(-40, 41, 4), n=15, period=200, warm=80):
    out = []
    for dt in dts:
        base = warm + max(0, -dt)
        pre = [base + k*period for k in range(n)]
        post = [p + dt for p in pre]
        T = pre[-1] + period
        u = _u_with_spikes(T, post, par)
        out.append((int(dt), _run_standalone(pre, u, par)/n))
    return out

def sa_freq(par, rates=(1,5,10,15,20,30,50), reps=15, delta=10):
    out = []
    for r in rates:
        per = int(round(1000.0/r))
        pre = [50 + k*per for k in range(reps)]; post = [p+delta for p in pre]
        T = pre[-1] + per + 60; u = _u_with_spikes(T, post, par)
        out.append((float(r), _run_standalone(pre, u, par)/reps))
    return out

# ----------------------- PRODUCTION-code driver -----------------------
def _build_prod(par):
    return _make_adex_voltage_step(
        decay_v=math.exp(-DT/100.0), v_rest=par["v_rest"], v_reset=par["v_rest"],
        v_thresh=par["v_thresh"], refrac_time=2, inh_weight=0.0,
        theta_plus=0.0, decay_theta=1.0, delta_T=2.0, v_T=par["v_thresh"]-2.0,
        a_adapt=0.0, b_adapt=0.0, decay_w_adapt=math.exp(-DT/144.0), I_sp=0.0,
        decay_z=math.exp(-DT/40.0),
        decay_x_pre_bar=math.exp(-DT/par["tau_x"]),
        decay_u_plus=math.exp(-DT/par["tau_ltp"]),
        decay_u_minus=math.exp(-DT/par["tau_ltd"]),
        A_plus=par["A_LTP"], A_minus=par["A_LTD"],
        theta_ltp=par["theta_high"], theta_ltd=par["theta_low"])

def _prod_sim(step, par, pre_t, teach_t, T, w_teacher=45.0):
    f = lambda x: torch.tensor(x, dtype=torch.float32)
    v=f([par["v_rest"]]); refrac=f([0.]); theta=f([0.]); wa=f([0.]); z=f([0.])
    xb=f([0.,0.]); up=f([par["v_rest"]]); um=f([par["v_rest"]]); uh=f([0.])
    dw=torch.zeros(2,1); sc=f([0.]); W=f([[1.0],[w_teacher]]); am=f([par["A_LTD"]])
    P,Tc=set(pre_t),set(teach_t)
    for t in range(T):
        pre=torch.zeros(2)
        if t in P: pre[0]=1.
        if t in Tc: pre[1]=1.
        (v,refrac,theta,wa,z,xb,up,um,uh,dw,sc)=step(v,refrac,theta,wa,z,xb,up,um,uh,dw,sc,W,pre,True,am)
    return float(dw[0,0])

def prod_window(par, dts=range(-40,41,4), n=15, period=200, warm=80):
    step=_build_prod(par); out=[]
    for dt in dts:
        base=warm+max(0,-dt); pre=[base+k*period for k in range(n)]; teach=[p+dt for p in pre]
        out.append((int(dt), _prod_sim(step,par,pre,teach,pre[-1]+period)/n))
    return out

def prod_freq(par, rates=(1,5,10,15,20,30,50), reps=15, delta=10):
    step=_build_prod(par); out=[]
    for r in rates:
        per=int(round(1000.0/r)); pre=[50+k*per for k in range(reps)]; teach=[p+delta for p in pre]
        out.append((float(r), _prod_sim(step,par,pre,teach,pre[-1]+per+60)/reps))
    return out

import os as _os
FIG_STEMS = ("results/clopath_curve_reproduction",
             _os.path.join(_os.environ.get("FIG_OUT_DIR", "figures"), "fig6_clopath_curves"))

def plot(res):
    """Figure A.1 of the paper, drawn at its printed size in the shared journal style
    (scripts/figures/paper_style.py). No titles: the caption describes the panels (a)-(c).
    Panel (a) has only the independent implementation; the shared legend says so by colour."""
    import os, sys
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator
    from scripts.figures import paper_style as ps
    ps.apply()
    def norm(ys):
        m=max(abs(y) for y in ys) or 1.0; return [y/m for y in ys]
    plain=FuncFormatter(lambda v,_: f"{v:g}".replace("-","\u2212"))
    # primary = the paper's voltage-STDP code, in the voltage-STDP colour; independent in ink
    # with open circles drawn on top, so both stay visible where the curves coincide
    IND=dict(color=ps.INK, marker="o", ms=3.0, mfc="white", mew=0.8, zorder=3,
             label="independent implementation")
    PRI=dict(color=ps.RULE_C["voltage"], marker=ps.RULE_M["voltage"], ms=3.0, zorder=2,
             label="primary implementation")
    fig,ax=plt.subplots(1,3,figsize=(ps.FULL_W,2.15),layout="constrained")
    # (a) voltage dependence: independent implementation only
    xs=[u for u,_ in res["sa_volt"]]; ys=[d for _,d in res["sa_volt"]]
    ax[0].plot(xs,ys,**IND)
    ps.zero_line(ax[0])
    ax[0].set_yscale("symlog", linthresh=0.05)  # LTD ~-0.07 and LTP ~+700 both visible
    ax[0].set_ylim(-0.5, 3000)
    ax[0].set_yticks([-0.1,0,1,100]); ax[0].yaxis.set_major_formatter(plain)
    ax[0].yaxis.set_minor_locator(NullLocator())
    ax[0].set_xticks([-70,-60,-50,-40])
    # threshold labels on the empty side of their lines (theta+ would touch the LTP points)
    for th,lab,dx,ha in ((PAR["theta_low"],r"$\theta_{-}$",0.6,"left"),
                         (PAR["theta_high"],r"$\theta_{+}$",-0.6,"right")):
        ps.guide_line(ax[0], th)
        ax[0].text(th+dx,0.97,lab,transform=ax[0].get_xaxis_transform(),ha=ha,va="top",
                   color=ps.INK,fontsize=ps.FS_SUB)
    ax[0].set_xlabel("clamped post voltage $u$ (mV)"); ax[0].set_ylabel(r"$\Delta w$ (symmetric log scale)")
    # (b) timing window (overlay, normalised)
    ax[1].plot([x for x,_ in res["sa_window"]],norm([d for _,d in res["sa_window"]]),**IND)
    ax[1].plot([x for x,_ in res["pr_window"]],norm([d for _,d in res["pr_window"]]),**PRI)
    ps.zero_line(ax[1]); ps.guide_line(ax[1], 0)
    ax[1].set_xticks([-40,-20,0,20,40])
    ax[1].set_xlabel(r"$\Delta t = t_{\mathrm{post}} - t_{\mathrm{pre}}$ (ms)")
    ax[1].set_ylabel(r"$\Delta w$ (normalised)")
    # (c) frequency dependence (overlay, normalised)
    ax[2].plot([x for x,_ in res["sa_freq"]],norm([d for _,d in res["sa_freq"]]),**IND)
    ax[2].plot([x for x,_ in res["pr_freq"]],norm([d for _,d in res["pr_freq"]]),**PRI)
    ps.zero_line(ax[2]); ax[2].set_xscale("log")
    ax[2].set_xticks([1,10,50]); ax[2].xaxis.set_major_formatter(plain)
    ax[2].xaxis.set_minor_locator(NullLocator())
    ax[2].set_xlabel("pairing frequency (Hz)"); ax[2].set_ylabel(r"$\Delta w$ per pair (normalised)")
    for a,l in zip(ax,("(a)","(b)","(c)")): ps.panel_label(a,l)
    fig.legend(*ax[1].get_legend_handles_labels(),loc="outside upper center",ncol=2)
    for stem in FIG_STEMS:
        os.makedirs(os.path.dirname(stem), exist_ok=True); ps.save(fig, stem)
        print(f"wrote {stem}.pdf/.png")
    plt.close(fig)

def main():
    res={}
    res["sa_volt"]=sa_voltage(PAR)
    res["sa_window"]=sa_window(PAR); res["pr_window"]=prod_window(PAR)
    res["sa_freq"]=sa_freq(PAR);     res["pr_freq"]=prod_freq(PAR)

    def rep(name, w):
        ltp=np.mean([d for x,d in w if x>0]); ltd=np.mean([d for x,d in w if x<0])
        print(f"  {name:16s} Δw(pre→post)={ltp:+.2e}  Δw(post→pre)={ltd:+.2e}")
    print("=== (B) STDP window ===");  rep("standalone",res["sa_window"]); rep("production",res["pr_window"])
    print("=== (C) frequency (Δw/pair, low→high Hz) ===")
    print("  standalone:", " ".join(f"{d:+.1e}" for _,d in res["sa_freq"]))
    print("  production:", " ".join(f"{d:+.1e}" for _,d in res["pr_freq"]))
    print("=== (A) voltage dep (Δw vs clamp u) ===")
    print("  ", " ".join(f"{u:.0f}:{d:+.1e}" for u,d in res["sa_volt"]))

    try:
        plot(res)
    except Exception as e:
        print("figure skipped:",type(e).__name__,str(e)[:90])
    json.dump({k:[[float(a),float(b)] for a,b in v] for k,v in res.items()},
              open("results/clopath_curve_reproduction.json","w"),indent=2)
    print("wrote results/clopath_curve_reproduction.json")

if __name__=="__main__":
    import sys
    if "--plot-only" in sys.argv:     # redraw from the saved curves, no simulation
        plot({k:[tuple(p) for p in v] for k,v in
              json.load(open("results/clopath_curve_reproduction.json")).items()})
    else:
        main()
