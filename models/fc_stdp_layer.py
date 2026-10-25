"""Unified fully-connected STDP layer for the scaffolding-ablation paper.

Architecture:  784 Poisson inputs  ->  N excitatory neurons (default N=400).

One class, three learning-rule branches:
    rule.name == "pair_exp"  :  D&C 2015 Rule-2 pair-based STDP
    rule.name == "triplet"   :  Pfister & Gerstner 2006 triplet STDP
    rule.name == "voltage"   :  Clopath 2010 voltage-based STDP (AdEx neuron)

Performance: each (rule, neuron) pair has its own timestep function built
in `_build_timestep()` and wrapped in torch.compile. This gets FC per-sample
time down to ~10-20 ms on GPU, vs ~200 ms for the naive unfused path.

Scaffolding toggles (already flattened into numerical params by
experiments/scripts/config.py:apply_scaffold_toggles):
    scaffold.lateral_inhibition=False -> inh_weight == 0
    scaffold.adaptive_threshold=False -> theta_plus == 0, decay_theta == 1.0

Firing policy: All three rules use refractory multi-fire (Q3); for pair_exp
this is a DEVIATION from the original SpikingConv single-fire — explicit in
paper Methods.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# ----------------------------------------------------------------------
# Rule-specific timestep factories. Each returns a function with signature
#     step(state_tuple, pre, W) -> new_state_tuple
# The state tuple contents depend on (rule, neuron_model); caller matches.
# ----------------------------------------------------------------------

def _make_lif_pair_step(
    *, decay_v, v_rest, v_reset, v_thresh, refrac_time, inh_weight,
    theta_plus, decay_theta,
    decay_pre_pair, decay_post_pair,
    eta_post, eta_pre, x_target, mu, w_min, w_max,
):
    def step(v, refrac, theta, x_pre, x_post, dw, spike_count, W, pre, train):
        exc = torch.mv(W.t(), pre)
        not_refrac = refrac == 0
        v = torch.where(not_refrac, v * decay_v + (1 - decay_v) * v_rest + exc, v)
        v_th = v_thresh + theta
        spikes = (v >= v_th) & not_refrac
        post = spikes.to(v.dtype)
        spike_count = spike_count + post
        v = torch.where(spikes, torch.full_like(v, v_reset), v)
        refrac = torch.where(spikes, torch.full_like(refrac, refrac_time), refrac)
        theta = torch.where(spikes, theta + theta_plus, theta) * decay_theta
        n_sp = post.sum()
        v = v - n_sp * inh_weight
        v = torch.where(spikes, torch.full_like(v, v_reset), v)
        refrac = torch.clamp(refrac - 1, min=0)

        x_pre = x_pre * decay_pre_pair
        x_post = x_post * decay_post_pair

        if train:
            w_dep_ltp = (w_max - W).clamp(min=0).pow(mu)
            dlt_post = post * eta_post
            dw_ltp = torch.outer(x_pre - x_target, dlt_post) * w_dep_ltp
            w_dep_ltd = (W - w_min).clamp(min=0).pow(mu)
            dlt_pre = pre * eta_pre
            dw_ltd = -torch.outer(dlt_pre, x_post) * w_dep_ltd
            dw = dw + dw_ltp + dw_ltd

        x_pre = x_pre + pre
        x_post = x_post + post
        return v, refrac, theta, x_pre, x_post, dw, spike_count
    return step


def _make_lif_triplet_step(
    *, decay_v, v_rest, v_reset, v_thresh, refrac_time, inh_weight,
    theta_plus, decay_theta,
    decay_r1, decay_r2, decay_o1, decay_o2,
    A2p, A2m, A3p, A3m,
):
    def step(v, refrac, theta, r1, r2, o1, o2, dw, spike_count, W, pre, train):
        exc = torch.mv(W.t(), pre)
        not_refrac = refrac == 0
        v = torch.where(not_refrac, v * decay_v + (1 - decay_v) * v_rest + exc, v)
        v_th = v_thresh + theta
        spikes = (v >= v_th) & not_refrac
        post = spikes.to(v.dtype)
        spike_count = spike_count + post
        v = torch.where(spikes, torch.full_like(v, v_reset), v)
        refrac = torch.where(spikes, torch.full_like(refrac, refrac_time), refrac)
        theta = torch.where(spikes, theta + theta_plus, theta) * decay_theta
        n_sp = post.sum()
        v = v - n_sp * inh_weight
        v = torch.where(spikes, torch.full_like(v, v_reset), v)
        refrac = torch.clamp(refrac - 1, min=0)

        r2_prev = r2
        o2_prev = o2
        r1 = r1 * decay_r1
        r2 = r2 * decay_r2
        o1 = o1 * decay_o1
        o2 = o2 * decay_o2

        if train:
            trip_ltp = A2p + A3p * o2_prev
            dw = dw + torch.outer(r1, post * trip_ltp)
            trip_ltd = A2m + A3m * r2_prev
            dw = dw - torch.outer(pre * trip_ltd, o1)

        r1 = r1 + pre; r2 = r2 + pre
        o1 = o1 + post; o2 = o2 + post
        return v, refrac, theta, r1, r2, o1, o2, dw, spike_count
    return step


def _make_adex_voltage_step(
    *, decay_v, v_rest, v_reset, v_thresh, refrac_time, inh_weight,
    theta_plus, decay_theta,
    delta_T, v_T, a_adapt, b_adapt, decay_w_adapt, I_sp, decay_z,
    decay_x_pre_bar, decay_u_plus, decay_u_minus,
    A_plus, A_minus, theta_ltp, theta_ltd,
    decay_u_homeo=1.0, homeo_voltage_on=False,
    spike_plateau=False, plateau_v=(29.4, 32.862), gate_cap=None,
):
    # Clopath 2010 normalisation (matches conv/conv2layer_voltage_stdp.py:813-816).
    # v_scale caps each rectified gate to [0, v_scale] before dividing by
    # v_scale → gates in [0, 1]. Without this, the AdEx exponential term can
    # push v to ~v_thresh+50 transiently, giving (v - theta_ltp) ~ 100 and a
    # per-spike LTP coefficient orders of magnitude above L1 renorm's reach,
    # causing catastrophic LTP runaway.
    v_scale = abs(v_thresh - v_rest)
    inv_vscale = 1.0 / max(v_scale, 1e-6)
    v_upper = v_thresh + 50.0
    v_lower = v_rest - 100.0
    # Optional Clopath spike plateau, seen by the plasticity rule only
    # (stdp.spike_plateau). Clopath's neuron holds each spike at +29.4 mV and
    # then +32.862 mV for 2 ms (ModelDB 144566, aEIF.m); this neuron resets in
    # the step it crosses threshold, so its spikes rarely reach theta_ltp. With
    # the option on, the voltage the rule and its traces see is replaced by
    # those two values on the spike step and the step after; the neuron's own
    # voltage, reset and refractory period are unchanged. "Spiked on the
    # previous step" is read from the refractory counter, which equals
    # refrac_time - 1 exactly then. gate_cap (mV) replaces the v_scale cap on
    # the three rectified gates (stdp.gate_cap_mv); Clopath's gates are
    # unbounded. Both default to the original behaviour.
    if gate_cap is None:
        gate_cap = v_scale
    if spike_plateau and refrac_time < 2:
        raise ValueError("stdp.spike_plateau needs neuron.refrac_time >= 2")
    plateau_v1, plateau_v2 = float(plateau_v[0]), float(plateau_v[1])

    def step(v, refrac, theta, w_adapt, z_after,
             x_pre_bar, u_plus_bar, u_minus_bar, u_homeo_bar,
             dw, spike_count, W, pre, train, a_minus_vec):
        exc = torch.mv(W.t(), pre)
        not_refrac = refrac == 0
        if spike_plateau:
            prev_spike = refrac == (refrac_time - 1)
        exp_arg = ((v - v_T) / delta_T).clamp(max=3.0)
        exp_term = delta_T * torch.exp(exp_arg)
        v_new = v * decay_v + (1 - decay_v) * v_rest \
                + exp_term + exc - w_adapt + z_after
        # Cap v to a safe range — AdEx exp-term positive feedback would
        # otherwise pump v to NaN under heavy drive.
        v_new = v_new.clamp(min=v_lower, max=v_upper)
        v = torch.where(not_refrac, v_new, v)

        w_adapt = w_adapt * decay_w_adapt \
                  + (1 - decay_w_adapt) * a_adapt * (v - v_rest)
        z_after = z_after * decay_z

        v_th = v_thresh + theta
        spikes = (v >= v_th) & not_refrac
        post = spikes.to(v.dtype)
        spike_count = spike_count + post

        # Voltage traces and plasticity are computed on the pre-reset v so the
        # spike depolarisation (exp_term pushing v above v_thresh) is visible
        # to the LTP rectification. Matches conv/conv2layer_voltage_stdp.py:857-886.
        # v_pl is that voltage, with the optional Clopath plateau substituted.
        if spike_plateau:
            v_pl = torch.where(spikes, torch.full_like(v, plateau_v1), v)
            v_pl = torch.where(prev_spike, torch.full_like(v, plateau_v2), v_pl)
        else:
            v_pl = v
        u_plus_bar = u_plus_bar * decay_u_plus + (1 - decay_u_plus) * v_pl
        u_minus_bar = u_minus_bar * decay_u_minus + (1 - decay_u_minus) * v_pl
        x_pre_bar = x_pre_bar * decay_x_pre_bar + pre

        # Clopath slow homeostatic average of the SQUARED depolarisation,
        # updated every timestep (ms-scale) on the PRE-reset v — this is the
        # variable Clopath's adaptive A_LTD depends on (running ⟨(u-u_rest)²⟩),
        # as opposed to the per-image spike-count EMA used in the 'activity'
        # homeostasis mode. homeo_voltage_on is a compile-time constant, so when
        # False this line is specialised away and the path is byte-identical to
        # the non-homeostatic voltage rule.
        if homeo_voltage_on:
            u_homeo_bar = u_homeo_bar * decay_u_homeo \
                          + (1.0 - decay_u_homeo) * (v_pl - v_rest) ** 2

        if train:
            v_rect = (v_pl - theta_ltp).clamp(min=0.0, max=gate_cap) * inv_vscale
            u_plus_rect = (u_plus_bar - theta_ltd).clamp(min=0.0, max=gate_cap) * inv_vscale
            ltp_post_coeff = v_rect * u_plus_rect * A_plus
            dw = dw + torch.outer(x_pre_bar, ltp_post_coeff)
            u_minus_rect = (u_minus_bar - theta_ltd).clamp(min=0.0, max=gate_cap) * inv_vscale
            # Per-neuron LTD amplitude. a_minus_vec already folds in the base
            # A_minus and (when homeostasis is enabled) the Clopath slow
            # homeostatic factor A_LTD·(ū/u_ref)^2. With homeostasis disabled
            # a_minus_vec is a constant = A_minus, reproducing the prior rule.
            dw = dw - torch.outer(pre, a_minus_vec * u_minus_rect)

        # Post-spike state updates (reset, refractory, adaptation,
        # afterpotential, adaptive threshold).
        v = torch.where(spikes, torch.full_like(v, v_reset), v)
        refrac = torch.where(spikes, torch.full_like(refrac, refrac_time), refrac)
        w_adapt = w_adapt + post * b_adapt
        z_after = z_after + post * I_sp
        theta = torch.where(spikes, theta + theta_plus, theta) * decay_theta

        # Subtractive lateral inhibition on co-active population.
        n_sp = post.sum()
        v = v - n_sp * inh_weight
        v = torch.where(spikes, torch.full_like(v, v_reset), v)
        refrac = torch.clamp(refrac - 1, min=0)

        return v, refrac, theta, w_adapt, z_after, \
               x_pre_bar, u_plus_bar, u_minus_bar, u_homeo_bar, \
               dw, spike_count
    return step


# ----------------------------------------------------------------------

class FCStdpLayer:
    """Single-layer FC STDP network. See module docstring for design notes."""

    def __init__(self, cfg: dict, device: Optional[torch.device] = None,
                 use_compile: bool = True):
        self.cfg = cfg
        self.device = device or torch.device(cfg["experiment"]["device"]
                                             if torch.cuda.is_available() else "cpu")
        dt = float(cfg["data"]["dt"])
        self.dt = dt
        self.n_steps_image = int(cfg["data"]["sim_time"] / dt)
        self.n_steps_quiet = int(cfg["data"]["quiet_time"] / dt)
        self.n_steps_total = self.n_steps_image + self.n_steps_quiet
        self.intensity = float(cfg["data"]["intensity"])
        self.n_input = int(cfg["model"]["n_input"])
        self.n_neurons = int(cfg["model"]["n_neurons"])
        self.rule_name = cfg["rule"]["name"]

        # Scaffold parameters (already flattened by apply_scaffold_toggles)
        scaf = cfg["scaffold"]
        inh_weight = float(scaf["inh_weight"])
        theta_plus = float(scaf["theta_plus"])
        tc_theta = scaf["tc_theta_decay"]
        decay_theta = 1.0 if (tc_theta == math.inf or tc_theta == float("inf")) \
                      else math.exp(-dt / float(tc_theta))
        self.theta_init = float(scaf["theta_init"])
        self.inh_weight = inh_weight
        self.theta_plus = theta_plus
        self.decay_theta = decay_theta

        # Weight-norm infrastructure
        wn = cfg["weight_norm"]
        self.weight_norm_enabled = bool(wn["enabled"])
        self.norm = float(wn["norm"]) if wn["norm"] is not None else None
        self.w_min = float(wn["w_min"])
        self.w_max = float(wn["w_max"])

        # Neuron model
        nm = cfg["neuron"]["model"]
        self.neuron_model = nm
        self._v_rest = float(cfg["neuron"]["v_rest"])
        self._v_reset = float(cfg["neuron"]["v_reset"])
        self._v_thresh = float(cfg["neuron"]["v_thresh"])
        self._decay_v = math.exp(-dt / float(cfg["neuron"]["tau_v"]))
        self._refrac_time = int(cfg["neuron"]["refrac_time"])

        # Weight initialisation (D&C uniform(0, high) + L1 col-norm)
        wi = cfg["weight_init"]
        g = torch.Generator(device="cpu").manual_seed(int(cfg["experiment"]["seed"]))
        w0 = torch.empty(self.n_input, self.n_neurons).uniform_(
            float(wi["low"]), float(wi["high"]), generator=g
        )
        self.w = w0.to(self.device)
        if self.weight_norm_enabled:
            self._normalize_weights()

        # Adaptive threshold (one per post neuron)
        self.theta = torch.full((self.n_neurons,), self.theta_init, device=self.device)

        # STDP hyperparameters - stored as scalars for closure capture
        s = cfg["stdp"]
        self._stdp = s

        # Build rule-specific timestep
        self._build_timestep(cfg, dt, use_compile=use_compile)

        # Post-hoc labelling state — n_classes is auto-set from dataset in
        # run_single.py (10 for MNIST/FMNIST/CIFAR/N-MNIST; 8/9/11 for
        # TissueMNIST/PathMNIST/OrganAMNIST). Default 10 for older configs.
        self.n_classes = int(cfg.get("data", {}).get("n_classes", 10))
        self.assignments = -torch.ones(self.n_neurons, dtype=torch.long, device=self.device)
        self.rates = torch.zeros(self.n_neurons, self.n_classes, device=self.device)

        # Per-run accumulators
        self._fire_total = torch.zeros(self.n_neurons, device=self.device)
        self._n_images_seen = 0

    # ------------------------------------------------------------------
    def _build_timestep(self, cfg: dict, dt: float, use_compile: bool):
        s = cfg["stdp"]
        common = dict(
            decay_v=self._decay_v, v_rest=self._v_rest, v_reset=self._v_reset,
            v_thresh=self._v_thresh, refrac_time=self._refrac_time,
            inh_weight=self.inh_weight,
            theta_plus=self.theta_plus, decay_theta=self.decay_theta,
        )
        if self.rule_name == "pair_exp":
            raw = _make_lif_pair_step(
                **common,
                decay_pre_pair=math.exp(-dt / s["tau_pre"]),
                decay_post_pair=math.exp(-dt / s["tau_post"]),
                eta_post=s["eta_post"], eta_pre=s["eta_pre"],
                x_target=s["x_target"], mu=s["mu"],
                w_min=self.w_min, w_max=self.w_max,
            )
        elif self.rule_name == "triplet":
            raw = _make_lif_triplet_step(
                **common,
                decay_r1=math.exp(-dt / s["tau_plus"]),
                decay_r2=math.exp(-dt / s["tau_x"]),
                decay_o1=math.exp(-dt / s["tau_minus"]),
                decay_o2=math.exp(-dt / s["tau_y"]),
                A2p=s["A2_plus"], A2m=s["A2_minus"],
                A3p=s["A3_plus"], A3m=s["A3_minus"],
            )
        elif self.rule_name == "voltage":
            n = cfg["neuron"]
            # --- Clopath slow homeostatic LTD scaling ---
            # Two homeostatic VARIABLES are supported (voltage_homeo.variable):
            #   "activity" (default, the Tier-0 sweep): A_LTD_eff[j] =
            #       A_minus * (ū_act[j] / u_ref)^2, where ū_act is a per-neuron
            #       EMA of PER-IMAGE spike count, window tau_homeo (in IMAGES).
            #   "voltage" (R1, Clopath as written): A_LTD_eff[j] =
            #       A_minus * (ū²[j] / u_ref²), where ū²[j] is a per-neuron
            #       running average of the SQUARED depolarisation (v-v_rest)²
            #       updated EVERY TIMESTEP (ms-scale), window tau_homeo·n_steps
            #       in ms. This is the membrane-voltage variable Clopath's
            #       adaptive A_LTD actually depends on — not spike activity.
            # Disabled by default so every prior run reproduces exactly
            # (a_minus_vec = const A_minus).
            self._A_minus_base = float(s["A_minus"])
            vh = cfg.get("voltage_homeo", {}) or {}
            self._homeo_enabled = bool(vh.get("enabled", False))
            self._homeo_variable = str(vh.get("variable", "activity")).lower()
            self._tau_homeo = float(vh.get("tau_homeo", 1000.0))   # images
            self._u_ref_cfg = vh.get("u_ref", "auto")
            self._homeo_clip = float(vh.get("clip", 10.0))         # cap factor
            self._u_homeo = None         # activity-mode EMA (lazily init)
            self._u_ref_running = None
            self._u_homeo_bar = None     # voltage-mode ⟨(v-v_rest)²⟩ (lazily init)
            # voltage-mode: window in ms = tau_homeo images × steps/image.
            homeo_voltage_on = self._homeo_enabled and self._homeo_variable == "voltage"
            tau_homeo_ms = max(self._tau_homeo * self.n_steps_total, 1.0)
            decay_u_homeo = math.exp(-dt / tau_homeo_ms) if homeo_voltage_on else 1.0
            raw = _make_adex_voltage_step(
                **common,
                delta_T=n["delta_T"], v_T=self._v_thresh + n["v_T_offset"],
                a_adapt=n["a_adapt"], b_adapt=n["b_adapt"],
                decay_w_adapt=math.exp(-dt / n["tau_w_adapt"]),
                I_sp=n["I_sp"], decay_z=math.exp(-dt / n["tau_z"]),
                decay_x_pre_bar=math.exp(-dt / s["tau_pre"]),
                decay_u_plus=math.exp(-dt / s["tau_plus"]),
                decay_u_minus=math.exp(-dt / s["tau_minus"]),
                A_plus=s["A_plus"], A_minus=s["A_minus"],
                theta_ltp=s["theta_ltp"], theta_ltd=s["theta_ltd"],
                decay_u_homeo=decay_u_homeo, homeo_voltage_on=homeo_voltage_on,
                spike_plateau=bool(s.get("spike_plateau", False)),
                plateau_v=(float(s.get("plateau_v1", 29.4)), float(s.get("plateau_v2", 32.862))),
                gate_cap=(None if s.get("gate_cap_mv") is None else float(s["gate_cap_mv"])),
            )
        else:
            raise ValueError(f"unknown rule: {self.rule_name}")

        if use_compile:
            try:
                # mode="default" gives fusion without cudagraphs — cudagraphs
                # would need per-step mark_step_begin because we reuse buffers
                # across iterations of the Python-level timestep loop.
                self._step_fn = torch.compile(raw, mode="default", dynamic=False)
                self._compiled = True
            except Exception:
                self._step_fn = raw
                self._compiled = False
        else:
            self._step_fn = raw
            self._compiled = False

    # ------------------------------------------------------------------
    def _normalize_weights(self):
        if self.norm is None:
            self.w.clamp_(self.w_min, self.w_max)
            return
        col_sums = self.w.sum(0, keepdim=True).clamp_(min=1e-8)
        self.w.mul_(self.norm / col_sums)
        self.w.clamp_(self.w_min, self.w_max)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_sample(self, image: torch.Tensor, train: bool = True) -> torch.Tensor:
        device = self.device
        dtype = torch.float32
        n_in, n_out = self.n_input, self.n_neurons
        N_t = self.n_steps_total
        N_img = self.n_steps_image

        image = image.to(device=device, dtype=dtype).view(-1)
        rates = (image * self.intensity / 1000.0 * self.dt).clamp_(0.0, 1.0)
        spike_image = torch.bernoulli(rates.unsqueeze(0).expand(N_img, -1))
        zero_pre = torch.zeros(n_in, device=device)

        v = torch.full((n_out,), self._v_rest, device=device)
        refrac = torch.zeros(n_out, dtype=torch.int32, device=device)
        theta = self.theta
        spike_count = torch.zeros(n_out, device=device)
        dw = torch.zeros_like(self.w) if train else torch.zeros(1, 1, device=device)

        if self.rule_name == "pair_exp":
            x_pre = torch.zeros(n_in, device=device)
            x_post = torch.zeros(n_out, device=device)
            for t in range(N_t):
                pre = spike_image[t] if t < N_img else zero_pre
                v, refrac, theta, x_pre, x_post, dw, spike_count = self._step_fn(
                    v, refrac, theta, x_pre, x_post, dw, spike_count, self.w, pre, train
                )
        elif self.rule_name == "triplet":
            r1 = torch.zeros(n_in, device=device)
            r2 = torch.zeros(n_in, device=device)
            o1 = torch.zeros(n_out, device=device)
            o2 = torch.zeros(n_out, device=device)
            for t in range(N_t):
                pre = spike_image[t] if t < N_img else zero_pre
                v, refrac, theta, r1, r2, o1, o2, dw, spike_count = self._step_fn(
                    v, refrac, theta, r1, r2, o1, o2, dw, spike_count, self.w, pre, train
                )
        elif self.rule_name == "voltage":
            w_adapt = torch.zeros(n_out, device=device)
            z_after = torch.zeros(n_out, device=device)
            x_pre_bar = torch.zeros(n_in, device=device)
            u_plus_bar = torch.full((n_out,), self._v_rest, device=device)
            u_minus_bar = torch.full((n_out,), self._v_rest, device=device)

            # Carry the slow voltage-homeostatic trace across images.
            if self._u_homeo_bar is None:
                u_homeo_bar = torch.zeros(n_out, device=device)
            else:
                u_homeo_bar = self._u_homeo_bar

            # Per-neuron LTD amplitude for THIS image (Clopath homeostasis).
            if not self._homeo_enabled:
                a_minus_vec = torch.full((n_out,), self._A_minus_base, device=device)
            elif self._homeo_variable == "voltage":
                # Clopath as written: A_LTD scaled by ⟨(v-v_rest)²⟩ / u_ref²,
                # with the ⟨·⟩ accumulated per-timestep (ms-scale) in the loop.
                if self._u_homeo_bar is None:
                    a_minus_vec = torch.full((n_out,), self._A_minus_base, device=device)
                else:
                    if self._u_ref_cfg == "auto":
                        u_ref_sq = max(float(self._u_ref_running), 1e-6)
                    else:
                        u_ref_sq = max(float(self._u_ref_cfg) ** 2, 1e-6)
                    factor = (self._u_homeo_bar / u_ref_sq).clamp(
                        min=1.0 / self._homeo_clip, max=self._homeo_clip)
                    a_minus_vec = self._A_minus_base * factor
            else:
                # activity mode (per-image spike-count EMA) — the Tier-0 sweep.
                if self._u_homeo is None:
                    a_minus_vec = torch.full((n_out,), self._A_minus_base, device=device)
                else:
                    if self._u_ref_cfg == "auto":
                        u_ref = max(float(self._u_ref_running), 1e-3)
                    else:
                        u_ref = max(float(self._u_ref_cfg), 1e-3)
                    factor = (self._u_homeo / u_ref).pow(2).clamp(
                        min=1.0 / self._homeo_clip, max=self._homeo_clip)
                    a_minus_vec = self._A_minus_base * factor

            for t in range(N_t):
                pre = spike_image[t] if t < N_img else zero_pre
                (v, refrac, theta, w_adapt, z_after,
                 x_pre_bar, u_plus_bar, u_minus_bar, u_homeo_bar,
                 dw, spike_count) = self._step_fn(
                    v, refrac, theta, w_adapt, z_after,
                    x_pre_bar, u_plus_bar, u_minus_bar, u_homeo_bar,
                    dw, spike_count, self.w, pre, train, a_minus_vec
                )

            # Update slow homeostatic trace on train.
            if train and self._homeo_enabled:
                if self._homeo_variable == "voltage":
                    # u_homeo_bar already evolved per-timestep inside the loop;
                    # persist it and refresh the population reference (auto).
                    self._u_homeo_bar = u_homeo_bar
                    if self._u_ref_cfg == "auto":
                        self._u_ref_running = float(u_homeo_bar.mean().item())
                else:
                    alpha = 1.0 / max(self._tau_homeo, 1.0)
                    if self._u_homeo is None:
                        self._u_homeo = spike_count.clone()
                        self._u_ref_running = float(spike_count.mean().item())
                    else:
                        self._u_homeo = (1.0 - alpha) * self._u_homeo + alpha * spike_count
                        if self._u_ref_cfg == "auto":
                            self._u_ref_running = ((1.0 - alpha) * self._u_ref_running
                                                   + alpha * float(spike_count.mean().item()))

        self.theta = theta
        if train:
            self.w = self.w + dw
            self.w.clamp_(self.w_min, self.w_max)
            if self.weight_norm_enabled:
                self._normalize_weights()

        self._fire_total = self._fire_total + spike_count
        self._n_images_seen += 1
        return spike_count

    # ------------------------------------------------------------------
    # Post-hoc labelling (Diehl & Cook)
    # ------------------------------------------------------------------
    def update_rates(self, spike_count: torch.Tensor, label: int):
        self.rates[:, label] = self.rates[:, label] + spike_count

    def compute_assignments(self):
        self.assignments = self.rates.argmax(dim=1)
        never_spiked = self.rates.sum(dim=1) == 0
        self.assignments[never_spiked] = -1

    @torch.no_grad()
    def classify(self, spike_count: torch.Tensor) -> int:
        votes = torch.zeros(self.n_classes, device=spike_count.device)
        for c in range(self.n_classes):
            mask = self.assignments == c
            if mask.any():
                votes[c] = spike_count[mask].sum()
        return int(votes.argmax().item())

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, extra: dict | None = None):
        ckpt = {
            "w": self.w.cpu(),
            "theta": self.theta.cpu(),
            "assignments": self.assignments.cpu(),
            "rates": self.rates.cpu(),
            "fire_total": self._fire_total.cpu(),
            "n_images_seen": self._n_images_seen,
            "rule_name": self.rule_name,
            "neuron_model": self.neuron_model,
            "n_input": self.n_input,
            "n_neurons": self.n_neurons,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.w = ckpt["w"].to(self.device)
        self.theta = ckpt["theta"].to(self.device)
        self.assignments = ckpt["assignments"].to(self.device)
        self.rates = ckpt["rates"].to(self.device)
        self._fire_total = ckpt["fire_total"].to(self.device)
        self._n_images_seen = int(ckpt["n_images_seen"])
