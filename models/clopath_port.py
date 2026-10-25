"""Exact port of Clopath et al. (2010)'s neuron and voltage-STDP update loop.

Source: ModelDB 144566, aEIF.m (neuron) and RFdevelop.m (update loop); not redistributed
here (download from ModelDB and set CLOPATH_MODELDB_DIR; the files are CRLF, latin-1).
Written for results/voltage_clopath_neuron_port.md (2026-10-13). This module does not
touch models/fc_stdp_layer.py.

Two implementations of the same equations:

* NumPy float64 reference, written line by line from the .m files:
    aeif_step_ref()   aEIF.m, one neuron (MATLAB `if` on a scalar)
    RFDevelopRef      RFdevelop.m loop body, n_neur = 1
    network_ref()     RFdevelop.m with n_neur = N, aEIF.m applied neuron by neuron, plus
                      the one extension used on MNIST (lateral inhibition, see below)
    votri_ref()       VoTri.m (Clopath's pairing-protocol code), for the protocol checks
* ClopathNet: batched torch version, B independent networks x N neurons x Ni inputs.

Facts of Clopath's code that are kept as written (they are not "fixed" here):
  - `u1s = u; u1ss = u1s;` both take the current u, so the voltage averages u_md, u_mp
    and the homeostatic average theta read u one step late (the comment says 2 ms;
    VoTri.m reads u_md(t-3), whose newest input is u(t-4)).
  - b = 0.0805 (commented "nA") is added to the adaptation variable, which is in pA
    (w_tail = 400 pA, I in pA, C in pF), so spike-triggered adaptation is 0.08 pA.
  - a spike is registered when u > 20 mV; u is then clamped at 29.4 and 32.862 mV for
    two steps; at the start of the following step u is reset to E_L + 15 + 6.0984 =
    -49.5016 mV, V_T jumps to -30.4 mV and w_tail to 400 pA. In the second clamped step
    the adaptation update is undone (w = w - wdot); w_tail and V_T keep relaxing.
  - input current I = w' * inp * C * 4: each input spike gives a 4*w mV step.
  - weights are updated every 1-ms step and clipped to [0, wmax = 3].

The only extension (MNIST setting, k_inh > 0): lateral inhibition as a current. A spike
registered in step t-1 (counter == 1 at the start of step t) injects -k_inh * C pA into
every OTHER neuron of the same network during step t, i.e. a k_inh-mV step, the same form
as the excitatory input (I = g * C * W' * spikes; Clopath: g = 4). k_inh = 0 and g = 4
give RFdevelop.m exactly. With inhibition the voltage is floored at u_floor (MNIST runs:
-100 mV, the inhibitory reversal potential of Diehl & Cook 2015): many neurons can fire in
the same step, and an unbounded current pulse then drives u to -10^4 mV and Clopath's
subthreshold adaptation (a = 4 nS) to a huge depolarising rebound. Without inhibition the
neuron never approaches -100 mV, so the floor does not change RFdevelop.m.

Homeostatic target (added 2026-10-13 for results/voltage_clopath_port_thetatar_PREREG.md):
theta_tar (RFdevelop.m: 60 mV^2) is an argument of ClopathNet and network_ref, default 60.
It enters only the depression term. When all networks of a ClopathNet share one value the
update is `theta / theta_tar` with a Python float, the expression used before the option
existed; several values per object use a per-network column (true division; on CUDA torch
divides by a Python scalar by multiplying with its reciprocal, so the two can differ in the
last bit).
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ----------------------------------------------------------------------
# aEIF.m model parameters (verbatim)
# ----------------------------------------------------------------------
th = 20.0          # [mV] spike threshold
C = 281.0          # [pF] membrane capacitance
g_L = 30.0         # [nS] membrane conductance
E_L = -70.6        # [mV] resting voltage
VT_rest = -50.4    # [mV] resetting voltage
Delta_T = 2.0      # [mV] exponential parameters
tau_w = 144.0      # [ms] time constant for adaptation variable w
a = 4.0            # [nS] adaptation coupling constant
b = 0.0805         # [nA] spike triggered adaptation (added to a pA variable, see above)
w_jump = 400.0     # spike after depolarisation
tau_wtail = 40.0   # [ms] time constant for spike after depolarisation
tau_VT = 50.0      # [ms] time constant for VT
VT_jump = 20.0     # adaptive threshold

U_FLOOR_INH = -100.0             # MNIST runs with inhibition only (see module docstring)
U_RESET = E_L + 15 + 6.0984      # aEIF.m line 45 (evaluated left to right, as MATLAB)
U_CLAMP1 = 29.4                  # aEIF.m line 67
U_CLAMP2 = 29.4 + 3.462          # aEIF.m line 62
VT_AFTER = VT_jump + VT_rest     # aEIF.m line 49

# ----------------------------------------------------------------------
# RFdevelop.m parameters (verbatim)
# ----------------------------------------------------------------------
dt = 1.0                          # duration of time step [ms]
tau_th = 1.2 * 1000.0             # time constant for sliding threshold
theta_tar = 60.0
A2d_RF = 10 * 0.00014             # depression amplitude, 10x the published value
A3p_RF = 10 * 0.00008             # potentiation amplitude, 10x the published value
A2d_PUB = 0.00014                 # published (Clopath 2010 Table 1, visual cortex)
A3p_PUB = 0.00008
tau_p = 7.0                       # voltage low-pass for the potentiation term
tau_r = 15.0                      # presynaptic spike low-pass
tau_d = 10.0                      # voltage low-pass for the depression term
wmax = 3.0                        # max synaptic weight
G_RF = 4.0                        # I = w'*inp*C*4
UY_TH = 25.3                      # uy = ((u-E_L) > 25.3).*(u-E_L-25.3): theta+ = E_L+25.3


# ======================================================================
# NumPy float64 reference
# ======================================================================

def aeif_step_ref(u, w, w_tail, I, counter, V_T):
    """aEIF.m, line by line, for one neuron (scalars)."""
    if counter == 2:                       # trick to force the spike to be 2ms long
        u = E_L + 15 + 6.0984
        w = w + b
        w_tail = w_jump
        counter = 0
        V_T = VT_jump + VT_rest
    # Updates of the variables for the aEIF
    udot = 1 / C * (-g_L * (u - E_L) + g_L * Delta_T * math.exp((u - V_T) / Delta_T) - w + w_tail + I)
    wdot = 1 / tau_w * (a * (u - E_L) - w)
    u = u + udot
    w = w + wdot
    w_tail = w_tail - w_tail / tau_wtail
    V_T = VT_rest / tau_VT + (1 - 1 / tau_VT) * V_T
    if counter == 1:
        counter = 2
        u = 29.4 + 3.462
        w = w - wdot
    if u > th and counter == 0:            # threshold condition for the spike
        u = 29.4
        counter = 1
    return u, w, w_tail, counter, V_T


class RFDevelopRef:
    """RFdevelop.m variables and loop body for n_neur = 1 (float64).

    step(inp) runs one iteration of `for t=2:T` given the input vector inp (Ni, 0/1).
    Options used only by the validation protocols (defaults = RFdevelop.m):
      g            input gain (RFdevelop: 4)
      I_extra      current added to I (pA), e.g. a spike-inducing pulse
      u_clamp      voltage clamp: u is set to this value after the neuron step
      theta_fixed  homeostasis held at this value (theta_tar gives factor 1)
    """

    def __init__(self, w0, A3p=A3p_RF, A2d=A2d_RF, g=G_RF, theta_fixed=None):
        self.w = np.array(w0, dtype=np.float64).reshape(-1).copy()
        Ni = self.w.shape[0]
        self.A3p, self.A2d, self.g, self.theta_fixed = A3p, A2d, g, theta_fixed
        self.u = E_L                 # membrane potential
        self.u_md = E_L              # low pass of u for the depression term
        self.u_mp = E_L              # low pass of u for the potentiation term
        self.uy = 0.0                # voltage thresholded
        self.u_sig_md = 0.0
        self.u_sig_mp = 0.0
        self.r = np.zeros(Ni)        # presyn low pass
        self.theta = 0.0             # sliding threshold
        self.u1s = E_L
        self.u1ss = E_L
        self.counter = 0
        self.w_tail = 0.0
        self.wad = 0.0
        self.V_T = -50.4
        self.spiked = False

    def step(self, inp, I_extra=0.0, u_clamp=None):
        inp = np.asarray(inp, dtype=np.float64)
        I = (self.w @ inp) * C * self.g                      # I = w'*inp*C*4; % currents
        if I_extra != 0.0:
            I = I + I_extra
        self.u, self.wad, self.w_tail, self.counter, self.V_T = aeif_step_ref(
            self.u, self.wad, self.w_tail, I, self.counter, self.V_T)
        self.spiked = self.counter == 1          # counter is 1 only in the step a spike is registered
        if u_clamp is not None:
            self.u = float(u_clamp)
        u = self.u
        self.uy = float((u - E_L) > 25.3) * (u - E_L - 25.3)
        self.r = (1 - dt / tau_r) * self.r + dt / tau_r * inp
        self.u_mp = self.u1ss * dt / tau_p + (1 - dt / tau_p) * self.u_mp
        self.u_md = self.u1ss * dt / tau_d + (1 - dt / tau_d) * self.u_md
        self.u_sig_md = float((self.u_md - E_L) > 0.) * (self.u_md - E_L - 0.)
        self.u_sig_mp = float((self.u_mp - E_L) > 0.) * (self.u_mp - E_L - 0.)
        self.theta = (1 - dt / tau_th) * self.theta + dt / tau_th * ((self.u1ss - E_L) ** 2)
        if self.theta_fixed is not None:
            self.theta = float(self.theta_fixed)
        self.w = self.w + self.A3p * self.r * (self.u_sig_mp * self.uy) \
            - self.A2d * inp * (self.u_sig_md * (self.theta / theta_tar))
        self.w[self.w < 0] = 0
        self.w[self.w > wmax] = wmax
        self.u1s = self.u
        self.u1ss = self.u1s
        return self.spiked


def network_ref(W0, inputs, g, k_inh, A3p, A2d, plastic=True, record_every=1, u_floor=None,
                theta_tar=theta_tar):
    """RFdevelop.m with n_neur = N (vectorised lines as in the .m file), aEIF.m applied to
    each neuron separately, plus lateral inhibition (k_inh mV per spike of another neuron,
    delivered in the step after the spike is registered). Reference for ClopathNet.

    W0 (Ni, N); inputs (T, Ni) of 0/1. Returns dict of traces (u, spikes) and final W.
    """
    W = np.array(W0, dtype=np.float64).copy()
    Ni, N = W.shape
    u = np.full(N, E_L); u_md = np.full(N, E_L); u_mp = np.full(N, E_L)
    r = np.zeros(Ni); theta = np.zeros(N); u1ss = np.full(N, E_L)
    counter = np.zeros(N, dtype=int); w_tail = np.zeros(N); wad = np.zeros(N)
    V_T = np.full(N, -50.4)
    T = inputs.shape[0]
    us, spikes, Ws = [], [], []
    for t in range(T):
        inp = inputs[t].astype(np.float64)
        I = (W.T @ inp) * C * g
        if k_inh != 0.0:
            s_prev = (counter == 1).astype(np.float64)
            I = I - (s_prev.sum() - s_prev) * C * k_inh
        spk = np.zeros(N, dtype=bool)
        for j in range(N):
            u[j], wad[j], w_tail[j], counter[j], V_T[j] = aeif_step_ref(
                u[j], wad[j], w_tail[j], I[j], counter[j], V_T[j])
            if u_floor is not None and u[j] < u_floor:
                u[j] = u_floor
            spk[j] = counter[j] == 1
        uy = ((u - E_L) > 25.3) * (u - E_L - 25.3)
        r = (1 - dt / tau_r) * r + dt / tau_r * inp
        u_mp = u1ss * dt / tau_p + (1 - dt / tau_p) * u_mp
        u_md = u1ss * dt / tau_d + (1 - dt / tau_d) * u_md
        u_sig_md = ((u_md - E_L) > 0.) * (u_md - E_L - 0.)
        u_sig_mp = ((u_mp - E_L) > 0.) * (u_mp - E_L - 0.)
        theta = (1 - dt / tau_th) * theta + dt / tau_th * ((u1ss - E_L) ** 2)
        if plastic:
            W = W + np.outer(A3p * r, u_sig_mp * uy) - np.outer(A2d * inp, u_sig_md * (theta / theta_tar))
            W[W < 0] = 0
            W[W > wmax] = wmax
        u1ss = u.copy()
        us.append(u.copy()); spikes.append(spk)
        if record_every and (t + 1) % record_every == 0:
            Ws.append(W.copy())
    return dict(u=np.array(us), spikes=np.array(spikes), W=W, W_trace=Ws)


def votri_ref(rho, Dt, par):
    """VoTri.m, line by line (1-based MATLAB indices kept via an offset of one)."""
    A_m, A_p, tau_p_, tau_r_, tau_d_ = par
    tetam = -70.6
    tetap = -45.3
    n = 5
    f = 1000 / rho
    if abs(f - round(f)) > 1e-12:
        raise ValueError("VoTri.m needs 1000/rho to be an integer (MATLAB index step)")
    f = int(round(f))
    l = n * f + abs(Dt) + 1
    x = np.zeros(l + 1)                       # index 0 unused
    x[abs(Dt) + 1:l:f] = 1                    # x(abs(Dt)+1:f:end-1) = 1
    y = np.zeros(l + 1)
    y[abs(Dt) + 1 + Dt:l + 1:f] = 1           # y(abs(Dt)+1+Dt:f:end) = 1
    rho_low = 0.1
    n_rep = 15
    if rho == 0.1:
        n_rep = 10
    pad = max(int(round(1000 / rho_low)) - l, 0)
    x = np.concatenate([[0.0], np.tile(np.concatenate([x[1:], np.zeros(pad)]), n_rep)])
    y = np.concatenate([[0.0], np.tile(np.concatenate([y[1:], np.zeros(pad)]), n_rep)])
    L = len(x) - 1
    I_s = y * 1000000
    u = np.full(L + 2, E_L); u_md = np.full(L + 2, E_L); u_mp = np.full(L + 2, E_L)
    wad = np.zeros(L + 2); w_tail_a = np.zeros(L + 2); r = np.zeros(L + 2)
    V_T = -50.4
    w = 0.5
    counter = 0
    for t in range(4, L + 1):
        I_tot = I_s[t] + 0.0
        u[t], wad[t], w_tail_a[t], counter, V_T = aeif_step_ref(
            u[t - 1], wad[t - 1], w_tail_a[t - 1], I_tot, counter, V_T)
        u_md[t + 1] = u[t] / tau_d_ + (1 - (1 / tau_d_)) * u_md[t]
        u_mp[t + 1] = u[t] / tau_p_ + (1 - (1 / tau_p_)) * u_mp[t]
        r[t + 1] = x[t] / tau_r_ + (1 - (1 / tau_r_)) * r[t]
        u_sig = (u[t] > tetap) * (u[t] - tetap)
        u_md_sig = ((u_md[t - 3] - tetam) > 0) * (u_md[t - 3] - tetam)
        u_mp_sig = ((u_mp[t - 3] - tetam) > 0) * (u_mp[t - 3] - tetam)
        w = w - A_m * x[t] * u_md_sig + A_p * u_sig * r[t] * u_mp_sig
    return w


# ======================================================================
# Batched torch version
# ======================================================================

_FUSED = None


def _fused_update_eager(W, pre_p, post_p, pre_d, post_d):
    """Same arithmetic as the unfused path, written as one expression for torch.compile."""
    W = W + pre_p.unsqueeze(2) * post_p.unsqueeze(1)
    W = W - pre_d.unsqueeze(2) * post_d.unsqueeze(1)
    return W.clamp(min=0.0, max=wmax)


def fused_update(W, pre_p, post_p, pre_d, post_d):
    """One fused GPU kernel for w = w + A3p*r*(..)' - A2d*inp*(..)', clipped to [0, wmax]
    (reads and writes W once instead of about five times). May differ from the unfused path
    in the last bit (fused multiply-add); checked in scripts/checks/clopath_port_equivalence.py."""
    global _FUSED
    if _FUSED is None:
        _FUSED = torch.compile(_fused_update_eager, dynamic=False)
    return _FUSED(W, pre_p, post_p, pre_d, post_d)


def _col(v, B, dtype, device):
    t = torch.as_tensor(v, dtype=dtype, device=device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(B)
    if t.numel() != B:
        raise ValueError(f"per-network parameter has {t.numel()} values, expected {B}")
    return t.reshape(B, 1).clone()


class ClopathNet:
    """B independent networks, each N aEIF.m neurons with an Ni x N weight matrix updated
    every step by the RFdevelop.m rule; optional lateral inhibition (k_inh, mV).

    W0: (B, Ni, N). g, k_inh, A3p, A2d: scalars or length-B sequences.
    plastic=False runs the same network with the weight update skipped (STDP disabled).
    Every arithmetic expression mirrors the NumPy reference (same operation order), so in
    float64 the two agree to rounding (tested in scripts/checks/clopath_port_equivalence.py).
    """

    def __init__(self, W0, g=G_RF, k_inh=0.0, A3p=A3p_RF, A2d=A2d_RF, *, plastic=True,
                 theta_fixed=None, dtype=torch.float64, device="cpu", fused=False, u_floor=None,
                 theta_tar=theta_tar):
        self.dtype, self.device = dtype, torch.device(device)
        self.W = torch.as_tensor(W0, dtype=dtype, device=self.device).clone()
        if self.W.dim() == 2:
            self.W = self.W.unsqueeze(0)
        B, Ni, N = self.W.shape
        self.B, self.Ni, self.N = B, Ni, N
        self.g = _col(g, B, dtype, self.device)
        self.k_inh = _col(k_inh, B, dtype, self.device)
        self.use_inh = bool((self.k_inh != 0).any())
        self.A3p = _col(A3p, B, dtype, self.device)
        self.A2d = _col(A2d, B, dtype, self.device)
        tt = torch.as_tensor(theta_tar, dtype=torch.float64).reshape(-1)
        if bool((tt == tt[0]).all()):
            self.theta_tar = float(tt[0])      # one value: theta / theta_tar as in RFdevelop.m
        else:
            self.theta_tar = _col(tt, B, dtype, self.device)   # one value per network
        self.plastic = plastic
        self.fused = fused
        self.u_floor = u_floor
        self.theta_fixed = theta_fixed
        z = lambda v: torch.full((B, N), v, dtype=dtype, device=self.device)
        self.u = z(E_L); self.u_md = z(E_L); self.u_mp = z(E_L)
        self.theta = z(0.0); self.u1ss = z(E_L)
        self.counter = torch.zeros((B, N), dtype=torch.int8, device=self.device)
        self.w_tail = z(0.0); self.wad = z(0.0); self.V_T = z(-50.4)
        self.r = torch.zeros((B, Ni), dtype=dtype, device=self.device)
        # diagnostics of the last step (cheap sums, before clipping)
        self.last_ltp = None
        self.last_ltd = None

    # -- state I/O (checkpoints) --------------------------------------
    STATE = ("W", "u", "u_md", "u_mp", "theta", "u1ss", "counter", "w_tail", "wad", "V_T", "r")

    def state_dict(self):
        return {k: getattr(self, k).clone() for k in self.STATE}

    def load_state_dict(self, sd):
        for k in self.STATE:
            getattr(self, k).copy_(sd[k].to(self.device))

    @classmethod
    def concat(cls, nets, plastic=False):
        """One ClopathNet holding the networks of several (same Ni, N, dtype, device), with
        their current state; used to present the test images to all networks at once."""
        n0 = nets[0]
        out = cls(torch.cat([n.W for n in nets], 0), g=torch.cat([n.g for n in nets]).reshape(-1),
                  k_inh=torch.cat([n.k_inh for n in nets]).reshape(-1),
                  A3p=torch.cat([n.A3p for n in nets]).reshape(-1),
                  A2d=torch.cat([n.A2d for n in nets]).reshape(-1), plastic=plastic,
                  theta_fixed=n0.theta_fixed, dtype=n0.dtype, device=n0.device, fused=n0.fused,
                  u_floor=n0.u_floor,
                  theta_tar=torch.cat([torch.as_tensor(n.theta_tar, dtype=torch.float64).reshape(-1).cpu()
                                       .expand(n.B) for n in nets]))
        out.use_inh = any(n.use_inh for n in nets)
        for k in cls.STATE:
            setattr(out, k, torch.cat([getattr(n, k) for n in nets], 0).clone())
        return out

    # -- one 1-ms step --------------------------------------------------
    @torch.no_grad()
    def step(self, inp, I_ext=None, u_clamp=None, diag=False):
        """inp: (B, Ni) or (Ni,) of 0/1 (shared by all B networks). Returns spikes (B, N) bool
        (spike registered in this step: counter set to 1)."""
        dtype = self.dtype
        inp = inp.to(dtype)
        if inp.dim() == 1:
            inp = inp.unsqueeze(0).expand(self.B, self.Ni)
        # I = w'*inp*C*4 (g = 4 in RFdevelop.m)
        I = torch.bmm(inp.unsqueeze(1), self.W).squeeze(1) * C * self.g
        if self.use_inh:
            s_prev = (self.counter == 1).to(dtype)
            I = I - (s_prev.sum(1, keepdim=True) - s_prev) * C * self.k_inh
        if I_ext is not None:
            I = I + I_ext
        # ---- aEIF.m -------------------------------------------------------
        u, w, w_tail, counter, V_T = self.u, self.wad, self.w_tail, self.counter, self.V_T
        c2 = counter == 2
        u = torch.where(c2, U_RESET, u)
        w = torch.where(c2, w + b, w)
        w_tail = torch.where(c2, w_jump, w_tail)
        counter = torch.where(c2, torch.zeros_like(counter), counter)
        V_T = torch.where(c2, VT_AFTER, V_T)
        udot = 1 / C * (-g_L * (u - E_L) + g_L * Delta_T * torch.exp((u - V_T) / Delta_T) - w + w_tail + I)
        wdot = 1 / tau_w * (a * (u - E_L) - w)
        u = u + udot
        w = w + wdot
        w_tail = w_tail - w_tail / tau_wtail
        V_T = VT_rest / tau_VT + (1 - 1 / tau_VT) * V_T
        c1 = counter == 1
        counter = torch.where(c1, torch.full_like(counter, 2), counter)
        u = torch.where(c1, U_CLAMP2, u)
        w = torch.where(c1, w - wdot, w)
        spk = (u > th) & (counter == 0)
        u = torch.where(spk, U_CLAMP1, u)
        counter = torch.where(spk, torch.ones_like(counter), counter)
        if self.u_floor is not None:
            u = u.clamp(min=self.u_floor)
        if u_clamp is not None:
            u = torch.as_tensor(u_clamp, dtype=dtype, device=self.device).expand_as(u).clone()
        self.u, self.wad, self.w_tail, self.counter, self.V_T = u, w, w_tail, counter, V_T
        # ---- RFdevelop.m loop body ------------------------------------------
        d = u - E_L
        uy = torch.where(d > 25.3, u - E_L - 25.3, torch.zeros_like(u))
        self.r = (1 - dt / tau_r) * self.r + dt / tau_r * inp
        self.u_mp = self.u1ss * dt / tau_p + (1 - dt / tau_p) * self.u_mp
        self.u_md = self.u1ss * dt / tau_d + (1 - dt / tau_d) * self.u_md
        dmd = self.u_md - E_L
        u_sig_md = torch.where(dmd > 0., self.u_md - E_L - 0., torch.zeros_like(dmd))
        dmp = self.u_mp - E_L
        u_sig_mp = torch.where(dmp > 0., self.u_mp - E_L - 0., torch.zeros_like(dmp))
        self.theta = (1 - dt / tau_th) * self.theta + dt / tau_th * ((self.u1ss - E_L) ** 2)
        if self.theta_fixed is not None:
            self.theta = torch.full_like(self.theta, float(self.theta_fixed))
        if self.plastic:
            pre_p = self.A3p * self.r                         # A3p*r
            post_p = u_sig_mp * uy                            # (u_sig_mp.*uy)'
            pre_d = self.A2d * inp                            # A2d*inp
            post_d = u_sig_md * (self.theta / self.theta_tar)  # (u_sig_md.*(theta./theta_tar))'
            if diag:
                self.last_ltp = pre_p.sum(1) * post_p.sum(1)  # total potentiation this step
                self.last_ltd = pre_d.sum(1) * post_d.sum(1)  # total depression this step
            if self.fused:
                self.W = fused_update(self.W, pre_p, post_p, pre_d, post_d)
            else:
                W = self.W + torch.bmm(pre_p.unsqueeze(2), post_p.unsqueeze(1))
                W = W - torch.bmm(pre_d.unsqueeze(2), post_d.unsqueeze(1))
                self.W = W.clamp_(min=0.0, max=wmax)
        self.u1ss = self.u                                    # u1s = u; u1ss = u1s;
        return spk


def votri_trains(rho, Dt):
    """The pre (x) and post (y) spike trains that VoTri.m builds (same lines as votri_ref),
    1-based (index 0 unused). Used to run the same protocols through the port."""
    n = 5
    f = int(round(1000 / rho))
    l = n * f + abs(Dt) + 1
    x = np.zeros(l + 1)
    x[abs(Dt) + 1:l:f] = 1
    y = np.zeros(l + 1)
    y[abs(Dt) + 1 + Dt:l + 1:f] = 1
    n_rep = 10 if rho == 0.1 else 15
    pad = max(int(round(1000 / 0.1)) - l, 0)
    x = np.concatenate([[0.0], np.tile(np.concatenate([x[1:], np.zeros(pad)]), n_rep)])
    y = np.concatenate([[0.0], np.tile(np.concatenate([y[1:], np.zeros(pad)]), n_rep)])
    return x, y
