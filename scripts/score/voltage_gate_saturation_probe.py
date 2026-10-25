"""Measurement 1: how often do the voltage rule's implementation guards bind?

Three implementation choices in `models/fc_stdp_layer._make_adex_voltage_step`
depart from Clopath et al. (2010) / textbook AdEx:

  (a) exponent cap        exp_arg = ((v - v_T)/delta_T).clamp(max=3.0)
  (b) membrane clamp      v_new.clamp(v_rest - 100, v_thresh + 50)
  (c) gate normalisation  (u - theta).clamp(0, v_scale) / v_scale,
                          v_scale = |v_thresh - v_rest| = 13 mV

(c) is the one that matters scientifically: Clopath's LTP term is *graded* in
the postsynaptic voltage, and saturating it at 13 mV would flatten exactly the
voltage dependence the rule is about.

This script measures the binding frequency of all three during real training.
It does NOT reimplement the step: it wraps the production `_step_fn` and
recomputes the internal quantities from the same inputs with the same
constants, then self-checks that its reconstruction is exact by comparing the
spike pattern it derives against the spike_count delta the real step returns.
Any mismatch aborts the run — the numbers are only reported if the
reconstruction provably tracks the real code.

Usage:
    python -u scripts/score/voltage_gate_saturation_probe.py \
        --scaffold full --seed 0 --n-train 1000 --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]

from models.fc_stdp_layer import FCStdpLayer          # noqa: E402
from experiments.scripts.config import resolve                            # noqa: E402
from experiments.scripts.run_single import load_dataset                   # noqa: E402


class GateStats:
    """Accumulates binding counts over every (timestep x neuron) event."""

    def __init__(self):
        self.n_events = 0            # timesteps x neurons
        self.n_spikes = 0            # spiking events
        # (a) exponent cap
        self.exp_cap = 0
        # (b) membrane clamp
        self.v_clamp_hi = 0
        self.v_clamp_lo = 0
        # (c) gate saturation, counted over events where the gate is active
        self.ltp_v_active = 0
        self.ltp_v_sat = 0
        self.ltp_u_active = 0
        self.ltp_u_sat = 0
        self.ltd_u_active = 0
        self.ltd_u_sat = 0
        # the headline number: LTP voltage gate at spike times
        self.ltp_v_sat_at_spike = 0
        # raw gate magnitude, to show how far past the ceiling it goes
        # (-inf so an all-negative gate is reported as negative, not as 0)
        self.ltp_v_raw_max = float("-inf")
        self.ltp_v_raw_sum_at_spike = 0.0
        self.v_max_any = float("-inf")
        # total plasticity mass: sum(outer(a,b)) == sum(a)*sum(b), so the
        # whole 784x400 update's mass costs two reductions, not an outer product
        self.ltp_mass = 0.0
        self.ltd_mass = 0.0

    def as_dict(self, v_scale):
        def frac(a, b):
            return (a / b) if b else float("nan")
        return {
            "n_events": self.n_events,
            "n_spikes": self.n_spikes,
            "v_scale_mV": v_scale,
            "exp_cap_frac_of_events": frac(self.exp_cap, self.n_events),
            "v_clamp_hi_frac_of_events": frac(self.v_clamp_hi, self.n_events),
            "v_clamp_lo_frac_of_events": frac(self.v_clamp_lo, self.n_events),
            "ltp_v_gate_active_frac": frac(self.ltp_v_active, self.n_events),
            "ltp_v_gate_sat_frac_of_active": frac(self.ltp_v_sat, self.ltp_v_active),
            "ltp_u_gate_active_frac": frac(self.ltp_u_active, self.n_events),
            "ltp_u_gate_sat_frac_of_active": frac(self.ltp_u_sat, self.ltp_u_active),
            "ltd_u_gate_active_frac": frac(self.ltd_u_active, self.n_events),
            "ltd_u_gate_sat_frac_of_active": frac(self.ltd_u_sat, self.ltd_u_active),
            "ltp_v_gate_sat_frac_at_spike": frac(self.ltp_v_sat_at_spike, self.n_spikes),
            "ltp_v_raw_gate_max_mV": self.ltp_v_raw_max,
            "v_max_over_all_events_mV": self.v_max_any,
            "ltp_v_raw_gate_mean_at_spike_mV": frac(self.ltp_v_raw_sum_at_spike, self.n_spikes),
            "ltp_mass_total": self.ltp_mass,
            "ltd_mass_total": self.ltd_mass,
            "ltp_over_ltd_ratio": frac(self.ltp_mass, self.ltd_mass),
        }


def wrap_step(layer: FCStdpLayer, cfg: dict, dt: float, stats: GateStats):
    """Wrap layer._step_fn with an exact reconstruction of its internals."""
    n = cfg["neuron"]
    s = cfg["stdp"]
    v_rest = layer._v_rest
    v_thresh = layer._v_thresh
    v_T = v_thresh + n["v_T_offset"]
    delta_T = n["delta_T"]
    decay_v = layer._decay_v
    decay_w_adapt = math.exp(-dt / n["tau_w_adapt"])
    decay_z = math.exp(-dt / n["tau_z"])
    decay_u_plus = math.exp(-dt / s["tau_plus"])
    decay_u_minus = math.exp(-dt / s["tau_minus"])
    theta_ltp = s["theta_ltp"]
    theta_ltd = s["theta_ltd"]
    A_plus = s["A_plus"]
    v_scale = abs(v_thresh - v_rest)
    v_upper = v_thresh + 50.0
    v_lower = v_rest - 100.0

    inner = layer._step_fn

    def instrumented(v, refrac, theta, w_adapt, z_after,
                     x_pre_bar, u_plus_bar, u_minus_bar, u_homeo_bar,
                     dw, spike_count, W, pre, train, a_minus_vec):
        # --- reconstruct the internals exactly as the production step does ---
        not_refrac = refrac == 0
        exp_arg_raw = (v - v_T) / delta_T
        exp_term = delta_T * torch.exp(exp_arg_raw.clamp(max=3.0))
        exc = torch.mv(W.t(), pre)
        v_raw = v * decay_v + (1 - decay_v) * v_rest \
            + exp_term + exc - w_adapt + z_after
        v_clamped = v_raw.clamp(min=v_lower, max=v_upper)
        v_pre_reset = torch.where(not_refrac, v_clamped, v)
        v_th = v_thresh + theta
        spikes = (v_pre_reset >= v_th) & not_refrac

        u_plus_new = u_plus_bar * decay_u_plus + (1 - decay_u_plus) * v_pre_reset
        u_minus_new = u_minus_bar * decay_u_minus + (1 - decay_u_minus) * v_pre_reset

        g_ltp_v = v_pre_reset - theta_ltp
        g_ltp_u = u_plus_new - theta_ltd
        g_ltd_u = u_minus_new - theta_ltd

        # --- run the real step ---
        out = inner(v, refrac, theta, w_adapt, z_after,
                    x_pre_bar, u_plus_bar, u_minus_bar, u_homeo_bar,
                    dw, spike_count, W, pre, train, a_minus_vec)
        sc_after = out[10]

        # --- self-check: our derived spikes must match the real ones ---
        real_spikes = (sc_after - spike_count) > 0.5
        if not torch.equal(real_spikes, spikes):
            raise RuntimeError(
                "reconstruction mismatch: derived spike pattern differs from "
                "the production step (%d vs %d spikes)"
                % (int(spikes.sum()), int(real_spikes.sum()))
            )

        if train:
            n_ev = v.numel()
            stats.n_events += n_ev
            stats.n_spikes += int(spikes.sum().item())
            stats.exp_cap += int((exp_arg_raw >= 3.0).sum().item())
            stats.v_clamp_hi += int((v_raw > v_upper).sum().item())
            stats.v_clamp_lo += int((v_raw < v_lower).sum().item())

            a = g_ltp_v > 0
            stats.ltp_v_active += int(a.sum().item())
            stats.ltp_v_sat += int((g_ltp_v >= v_scale).sum().item())
            a = g_ltp_u > 0
            stats.ltp_u_active += int(a.sum().item())
            stats.ltp_u_sat += int((g_ltp_u >= v_scale).sum().item())
            a = g_ltd_u > 0
            stats.ltd_u_active += int(a.sum().item())
            stats.ltd_u_sat += int((g_ltd_u >= v_scale).sum().item())

            inv = 1.0 / v_scale
            v_rect = g_ltp_v.clamp(min=0.0, max=v_scale) * inv
            u_plus_rect = g_ltp_u.clamp(min=0.0, max=v_scale) * inv
            u_minus_rect = g_ltd_u.clamp(min=0.0, max=v_scale) * inv
            stats.ltp_mass += float(x_pre_bar.sum().item()) * \
                float((v_rect * u_plus_rect * A_plus).sum().item())
            stats.ltd_mass += float(pre.sum().item()) * \
                float((a_minus_vec * u_minus_rect).sum().item())
            stats.v_max_any = max(stats.v_max_any, float(v_pre_reset.max().item()))
            if int(spikes.sum().item()) > 0:
                gs = g_ltp_v[spikes]
                stats.ltp_v_sat_at_spike += int((gs >= v_scale).sum().item())
                stats.ltp_v_raw_sum_at_spike += float(gs.sum().item())
                stats.ltp_v_raw_max = max(stats.ltp_v_raw_max, float(gs.max().item()))

        return out

    layer._step_fn = instrumented
    return v_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaffold", default="full")
    ap.add_argument("--config-file", default=None,
                    help="a resolved config.yaml (e.g. a saved run's) to probe "
                         "instead of resolve(rule, scaffold) — used for the "
                         "tuned voltage configuration, which lives only in "
                         "saved runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="results/voltage_gate_saturation.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.config_file:
        import yaml
        cfg = yaml.safe_load(Path(args.config_file).read_text())
        # A saved run's config carries its own smoke override; the probe's
        # --n-train is authoritative here.
        cfg.get("smoke", {}).pop("n_train_override", None)
    else:
        cfg = resolve("voltage", args.scaffold)
    cfg["experiment"]["seed"] = args.seed
    cfg["training"]["n_train"] = args.n_train
    # torch.compile would hide the closure we need to wrap.
    cfg.setdefault("runtime", {})["compile"] = False

    device = torch.device(args.device)
    net = FCStdpLayer(cfg, device=device, use_compile=False)

    dt = float(cfg["data"]["dt"])
    stats = GateStats()
    v_scale = wrap_step(net, cfg, dt, stats)

    # Use the production loader so preprocessing is identical to the runs.
    dataset_name = cfg["data"]["dataset"]
    train_set, _ = load_dataset(cfg["data"]["data_dir"], dataset_name)
    threshold = cfg.get("data", {}).get("threshold", None)

    g = torch.Generator().manual_seed(args.seed + 1_000)
    order = torch.randperm(len(train_set), generator=g).tolist()[: args.n_train]

    print(f"probe: voltage / S_{args.scaffold} / seed {args.seed} / "
          f"{args.n_train} images on {device} [{dataset_name}]", flush=True)
    print(f"v_scale = {v_scale} mV, theta_ltp = {cfg['stdp']['theta_ltp']} mV, "
          f"saturation at v >= {cfg['stdp']['theta_ltp'] + v_scale} mV", flush=True)

    for k, idx in enumerate(order):
        image, _ = train_set[idx]
        if threshold is not None:
            image = (image > threshold).float()
        net.run_sample(image, train=True)
        if (k + 1) % 100 == 0:
            d = stats.as_dict(v_scale)
            print(f"  [{k+1}/{args.n_train}] "
                  f"LTP v-gate saturated at spike: "
                  f"{d['ltp_v_gate_sat_frac_at_spike']*100:.1f}% | "
                  f"exp cap: {d['exp_cap_frac_of_events']*100:.3f}% | "
                  f"v clamp hi: {d['v_clamp_hi_frac_of_events']*100:.3f}%",
                  flush=True)

    out = stats.as_dict(v_scale)
    out["config"] = {
        "rule": "voltage", "scaffold": args.scaffold, "seed": args.seed,
        "dataset": dataset_name,
        "config_file": args.config_file,
        "tc_theta_decay": cfg["scaffold"].get("tc_theta_decay"),
        "n_train": args.n_train, "theta_ltp": cfg["stdp"]["theta_ltp"],
        "theta_ltd": cfg["stdp"]["theta_ltd"], "v_thresh": net._v_thresh,
        "v_rest": net._v_rest, "inh_weight": net.inh_weight,
        "A_plus": cfg["stdp"]["A_plus"], "A_minus": cfg["stdp"]["A_minus"],
    }
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print("\n=== RESULT ===")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
