"""
2-Layer Convolutional Voltage-Modulated STDP with Surrogate Gradient Readout
============================================================================

Architecture:
  Layer 1: Conv (n_filters1 x kernel_size1) — Voltage STDP, unsupervised
  Layer 2: Conv (n_filters2 x kernel_size2) — Voltage STDP, unsupervised
  Readout: FC SNN trained with surrogate gradients (supervised)

Voltage STDP (Clopath et al. 2010):
  Uses low-pass filtered membrane voltage traces to modulate STDP:
    LTP: dW+ = A_plus * x_pre_bar * (u_plus_bar / voltage_scale)   [pre active]
    LTD: dW- = A_minus * (u_minus_bar / voltage_scale)              [pre silent]

  x_pre_bar   : low-pass filtered pre-synaptic spike trace
  u_plus_bar  : fast post-synaptic voltage trace (modulates LTP)
  u_minus_bar : slow post-synaptic voltage trace (modulates LTD)

3-Phase Pipeline:
  Phase A: Layer-by-layer unsupervised voltage STDP training
  Phase B: Feature extraction (concatenated pooled spike counts)
  Phase C: Surrogate gradient readout SNN training

Usage:
    python conv/conv2layer_voltage_stdp.py
    python conv/conv2layer_voltage_stdp.py --config conv/configs/conv_voltage_config.yaml
"""

import os
import sys
import argparse
import glob
import re
import time
import math
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

# Enable TF32 for faster float32 matmuls on Ampere+ GPUs (RTX 3xxx/A-series)
torch.set_float32_matmul_precision('high')
from torchvision import datasets, transforms

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# LOGGING
# =============================================================================

class TeeStream:
    """Duplicates all writes to both terminal and log file."""
    def __init__(self, original_stream, log_file):
        self.original = original_stream
        self.log_file = log_file

    def write(self, message):
        self.original.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def setup_logging(save_dir, experiment_name):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"{experiment_name}_{timestamp}.log")
    log_file = open(log_path, "w")

    log_file.write(f"{'='*70}\n")
    log_file.write(f"  Conv Voltage STDP Training Log\n")
    log_file.write(f"  Experiment : {experiment_name}\n")
    log_file.write(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"  Log file   : {log_path}\n")
    log_file.write(f"{'='*70}\n\n")
    log_file.flush()

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    print(f"Logging to: {log_path}")
    return log_file, log_path


# =============================================================================
# SURROGATE GRADIENT
# =============================================================================

class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient using fast sigmoid."""
    scale = 25.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output / (SurrogateSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


spike_fn = SurrogateSpike.apply


def _new_layer_diagnostics():
    """Create empty interval/epoch diagnostics for an STDP layer.

    `voltage_max` is initialized to -inf (not 0.0) because membrane voltages
    are typically negative (v_rest=-65 mV). A 0.0 sentinel would mask real
    peaks like -2.0 from reporting (since max(0.0, -2.0) == 0.0).
    """
    return {
        "total_spikes": 0.0,
        "total_input_spikes": 0.0,
        "raw_ltp_mag": 0.0,
        "raw_ltd_mag": 0.0,
        "net_dw_pos_mag": 0.0,
        "net_dw_neg_mag": 0.0,
        "u_plus_sum": 0.0,
        "u_minus_sum": 0.0,
        "pre_trace_sum": 0.0,
        "voltage_sum": 0.0,
        "voltage_max": -float("inf"),
        "dw_pos_sum": 0.0,
        "dw_neg_sum": 0.0,
        "n_images": 0,
    }


def _accumulate_layer_diagnostics(targets, values):
    """Accumulate one image worth of metrics into all requested scopes."""
    for diag in targets:
        diag["voltage_max"] = max(diag["voltage_max"],
                                  values.get("voltage_max", 0.0))
        for key, value in values.items():
            if key == "voltage_max":
                continue
            diag[key] += value


def _finalize_layer_diagnostics(layer, source):
    """Add derived metrics and current layer state to a diagnostic snapshot."""
    d = source.copy()
    n = max(d["n_images"], 1)
    total_neurons = layer.n_pos * layer.n_filters

    # Backward-compatible aliases: ltp/ltd now mean raw plasticity terms.
    d["ltp_mag"] = d["raw_ltp_mag"]
    d["ltd_mag"] = d["raw_ltd_mag"]

    d["avg_spikes_per_image"] = d["total_spikes"] / n
    d["avg_input_spikes_per_image"] = d["total_input_spikes"] / n
    d["firing_rate"] = d["total_spikes"] / (n * total_neurons)
    d["avg_voltage"] = d["voltage_sum"] / n
    # Replace -inf sentinel (no images seen yet) with NaN so wandb/plots
    # don't get a giant negative outlier.
    d["peak_voltage"] = (
        d["voltage_max"] if d["voltage_max"] != -float("inf") else float("nan"))
    d["avg_u_plus"] = d["u_plus_sum"] / n
    d["avg_u_minus"] = d["u_minus_sum"] / n
    d["avg_pre_trace"] = d["pre_trace_sum"] / n

    raw_total = d["raw_ltp_mag"] + d["raw_ltd_mag"] + 1e-12
    d["raw_ltp_fraction"] = d["raw_ltp_mag"] / raw_total
    d["raw_ltd_fraction"] = d["raw_ltd_mag"] / raw_total
    d["ltp_fraction"] = d["raw_ltp_fraction"]
    d["ltd_fraction"] = d["raw_ltd_fraction"]
    d["avg_raw_ltp_mag"] = d["raw_ltp_mag"] / n
    d["avg_raw_ltd_mag"] = d["raw_ltd_mag"] / n

    net_total = d["net_dw_pos_mag"] + d["net_dw_neg_mag"] + 1e-12
    d["net_dw_pos_fraction"] = d["net_dw_pos_mag"] / net_total
    d["net_dw_neg_fraction"] = d["net_dw_neg_mag"] / net_total
    d["avg_net_dw_pos"] = d["dw_pos_sum"] / n
    d["avg_net_dw_neg"] = d["dw_neg_sum"] / n
    d["avg_dw_pos"] = d["avg_net_dw_pos"]
    d["avg_dw_neg"] = d["avg_net_dw_neg"]

    w = layer.filters
    d["weight_mean"] = float(w.mean())
    d["weight_std"] = float(w.std())
    d["weight_min"] = float(w.min())
    d["weight_max"] = float(w.max())
    d["weight_near_0_pct"] = float((w < 0.05).sum()) / w.numel() * 100
    d["weight_near_1_pct"] = float((w > 0.95).sum()) / w.numel() * 100
    d["convergence"] = float((w * (1 - w)).mean())

    d["theta_mean"] = float(layer.theta.mean())
    d["theta_max"] = float(layer.theta.max())

    voltage_span = max(abs(layer.v_thresh - layer.v_rest), 1.0)
    has_images = d["n_images"] > 0
    d["zero_ltd_warning"] = float(has_images and d["raw_ltd_mag"] <= 1e-12)
    d["dead_firing_warning"] = float(has_images and d["firing_rate"] <= 1e-6)
    d["theta_saturated_warning"] = float(
        has_images and (
            d["theta_mean"] > 4.0 * voltage_span or
            d["theta_max"] > 6.0 * voltage_span
        )
    )
    # Collapse detection distinguishes two pathologies from healthy bimodal:
    #   (1) weight_uniform_warning: all weights stuck at the same value
    #       (std ~ 0). This is the genuine all-to-same-filter collapse.
    #   (2) weight_saturation_warning: nearly every weight is clamped to 0 or
    #       1 AND std is small (< 0.1). The std guard is critical — healthy
    #       hard-bounded STDP converges to a bimodal {0, 1} distribution with
    #       std up to sqrt(p*(1-p)) ~ 0.49, and that is the CORRECT endpoint,
    #       not a failure. Only flag saturation when the distribution has
    #       collapsed to (near-)all-one-value AT the extremes.
    d["weight_uniform_warning"] = float(has_images and d["weight_std"] < 1e-4)
    d["weight_saturation_warning"] = float(
        has_images
        and (d["weight_near_0_pct"] + d["weight_near_1_pct"]) > 95.0
        and d["weight_std"] < 0.1
    )
    d["weight_collapse_warning"] = float(
        d["weight_uniform_warning"] or d["weight_saturation_warning"]
    )
    # Informational flag for healthy bimodal (no longer an error condition).
    d["weight_bimodal_info"] = float(
        has_images
        and (d["weight_near_0_pct"] + d["weight_near_1_pct"]) > 90.0
        and d["weight_std"] >= 0.1
    )

    return d


# =============================================================================
# READOUT SNN (trained with surrogate gradient)
# =============================================================================

class ReadoutSNN(nn.Module):
    """Fully connected LIF readout SNN trained with surrogate gradients."""

    def __init__(self, n_input, n_hidden, n_output, n_steps=50,
                 tau=10.0, dt=1.0):
        super().__init__()
        self.n_steps = n_steps
        self.decay = math.exp(-dt / tau)
        self.n_output = n_output

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """x: (batch, n_input) spike count features -> (batch, n_output) logits."""
        batch = x.shape[0]
        device = x.device

        x_norm = x / (x.max(dim=1, keepdim=True).values.clamp(min=1.0))

        v1 = torch.zeros(batch, self.fc1.out_features, device=device)
        v2 = torch.zeros(batch, self.n_output, device=device)
        out_spikes = torch.zeros(batch, self.n_output, device=device)

        for t in range(self.n_steps):
            inp_spikes = (torch.rand_like(x_norm) < x_norm).float()

            v1 = self.decay * v1 + self.fc1(inp_spikes)
            s1 = spike_fn(v1 - 1.0)
            v1 = v1 * (1.0 - s1)

            v2 = self.decay * v2 + self.fc2(s1)
            s2 = spike_fn(v2 - 1.0)
            v2 = v2 * (1.0 - s2)
            out_spikes = out_spikes + s2

        return out_spikes


# =============================================================================
# CONV VOLTAGE STDP LAYER (unsupervised, GPU)
# =============================================================================

class ConvVoltageSTDPLayer:
    """Single convolutional layer with voltage-modulated STDP (Clopath et al. 2010).

    LIF neuron model with membrane leak and two voltage traces:
      - u_plus_bar  (fast): low-pass of membrane voltage, modulates LTP
      - u_minus_bar (slow): low-pass of membrane voltage, modulates LTD

    Weight update (accumulated over timesteps, applied once per image):
      LTP: dW += A_plus * x_pre_bar * u_plus_norm     [where pre trace > 0]
      LTD: dW -= A_minus * u_minus_norm                [where pre trace ~ 0]

    Voltage traces are normalized by (v_thresh - v_rest) to keep updates in
    a consistent range regardless of threshold settings.
    """

    def __init__(self, n_input_channels, n_filters, kernel_size, input_h, input_w,
                 dt=1.0, sim_time=350.0, intensity=128.0, norm=10.0,
                 A_plus=6.0e-3, A_minus=3.0e-3,
                 tau_pre=16.8, tau_plus=3.0, tau_minus=8.0,
                 v_rest=-65.0, v_reset=-65.0, v_thresh=-52.0, tau_v=100.0,
                 theta_plus=0.05, tc_theta_decay=1e7,
                 inh_weight=17.5, refrac_time=5, device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_input_ch = n_input_channels
        self.n_filters = n_filters
        self.ks = kernel_size
        self.k_sq = kernel_size * kernel_size * n_input_channels
        self.dt = dt
        self.n_steps = int(sim_time / dt)
        self.intensity = intensity
        self.norm = norm
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh

        self.H_out = input_h - kernel_size + 1
        self.W_out = input_w - kernel_size + 1
        self.n_pos = self.H_out * self.W_out
        self.pool_h = self.H_out // 2
        self.pool_w = self.W_out // 2

        # Precompute decay factors
        self.decay_v = math.exp(-dt / tau_v)
        self.decay_theta = math.exp(-dt / tc_theta_decay)
        self.decay_pre = math.exp(-dt / tau_pre)
        self.decay_plus = math.exp(-dt / tau_plus)
        self.decay_minus = math.exp(-dt / tau_minus)

        # Voltage STDP amplitudes
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.voltage_scale = max(abs(v_thresh - v_rest), 1.0)

        # Inhibition / refractory
        self.inh_weight = inh_weight
        self.theta_plus = theta_plus
        self._v_reset_t = torch.tensor(v_reset, device=self.device)
        self._refrac_time_t = torch.tensor(refrac_time, dtype=torch.int32, device=self.device)

        # Filter weights
        self.filters = torch.rand(n_filters, self.k_sq, device=self.device) * 0.3
        self._normalize_filters()

        # Adaptive threshold
        self.theta = torch.zeros(self.n_pos, n_filters, device=self.device)

        # torch.compile: built lazily on first forward()
        self._compiled = False

        # Diagnostics accumulators: interval resets for frequent logs, epoch
        # resets only at epoch summaries.
        self._reset_diagnostics(scope="all")

    def _reset_diagnostics(self, scope="interval"):
        if scope in ("all", "interval"):
            self._diag = _new_layer_diagnostics()
        if scope in ("all", "epoch"):
            self._epoch_diag = _new_layer_diagnostics()

    def get_diagnostics(self, scope="interval", reset=True):
        """Return accumulated diagnostics for interval or epoch scope."""
        source = self._epoch_diag if scope == "epoch" else self._diag
        d = _finalize_layer_diagnostics(self, source)
        if reset:
            self._reset_diagnostics(scope=scope)
        return d

    def _normalize_filters(self):
        if self.norm is not None:
            row_sums = self.filters.sum(1, keepdim=True).clamp_(min=1e-8)
            self.filters.mul_(self.norm / row_sums)
            self.filters.clamp_(0.0, 1.0)

    def _build_compiled_steps(self):
        """Build torch.compiled timestep functions for train and inference.

        torch.compile fuses the many small GPU kernels inside each timestep
        into fewer, larger fused kernels — dramatically reducing kernel launch
        overhead and increasing GPU utilization.
        """
        # All scalar constants captured by closure
        decay_pre = self.decay_pre
        decay_v = self.decay_v
        decay_plus = self.decay_plus
        decay_minus = self.decay_minus
        decay_theta = self.decay_theta
        v_rest = self.v_rest
        v_thresh = self.v_thresh
        leak_rest = (1 - decay_v) * v_rest
        one_minus_dp = 1 - decay_plus
        one_minus_dm = 1 - decay_minus
        inh_weight = self.inh_weight
        inh_global = inh_weight * 0.1
        theta_plus = self.theta_plus
        v_reset_t = self._v_reset_t
        refrac_time_t = self._refrac_time_t
        inv_vscale = 1.0 / self.voltage_scale
        A_plus = self.A_plus
        A_minus = self.A_minus

        def _step_train(pre_patches, v, refrac, theta, x_pre_bar,
                        u_plus_bar, u_minus_bar, spike_count, dw_accum, filters_t):
            # 1. Pre trace
            x_pre_bar = x_pre_bar * decay_pre + pre_patches
            # 2. LIF
            exc_input = pre_patches @ filters_t
            not_refrac = refrac == 0
            v = torch.where(not_refrac, v * decay_v + leak_rest + exc_input, v)
            # 3. Voltage traces
            v_dev = (v - v_rest).clamp(min=0)
            u_plus_bar = u_plus_bar * decay_plus + v_dev * one_minus_dp
            u_minus_bar = u_minus_bar * decay_minus + v_dev * one_minus_dm
            # 4. Spikes
            spikes = (v >= (v_thresh + theta)) & not_refrac
            post = spikes.float()
            spike_count = spike_count + post
            # 5. Reset & refractory
            v = torch.where(spikes, v_reset_t, v)
            refrac = torch.where(spikes, refrac_time_t, refrac)
            theta = torch.where(spikes, theta + theta_plus, theta) * decay_theta
            # 6. Lateral inhibition
            v = v - post.sum(dim=1, keepdim=True) * inh_weight
            v = v - post.sum(dim=0, keepdim=True) * inh_global
            v = torch.where(spikes, v_reset_t, v)
            refrac = (refrac - 1).clamp(min=0)
            # 7. Voltage STDP
            u_plus_norm = u_plus_bar * inv_vscale
            u_minus_norm = u_minus_bar * inv_vscale
            has_pre = (x_pre_bar > 1e-6).float()
            ltp_coeff = post * u_plus_norm
            ltd_coeff = post * u_minus_norm
            dw_accum = dw_accum + A_plus * (ltp_coeff.t() @ (x_pre_bar * has_pre))
            dw_accum = dw_accum - A_minus * (ltd_coeff.t() @ (1.0 - has_pre))
            return v, refrac, theta, x_pre_bar, u_plus_bar, u_minus_bar, spike_count, dw_accum, post

        def _step_infer(pre_patches, v, refrac, theta, spike_count, filters_t):
            exc_input = pre_patches @ filters_t
            not_refrac = refrac == 0
            v = torch.where(not_refrac, v * decay_v + leak_rest + exc_input, v)
            spikes = (v >= (v_thresh + theta)) & not_refrac
            post = spikes.float()
            spike_count = spike_count + post
            v = torch.where(spikes, v_reset_t, v)
            refrac = torch.where(spikes, refrac_time_t, refrac)
            theta = torch.where(spikes, theta + theta_plus, theta) * decay_theta
            v = v - post.sum(dim=1, keepdim=True) * inh_weight
            v = v - post.sum(dim=0, keepdim=True) * inh_global
            v = torch.where(spikes, v_reset_t, v)
            refrac = (refrac - 1).clamp(min=0)
            return v, refrac, theta, spike_count, post

        def _step_infer_batch(pre, v, refrac, theta, spike_count, filters_t):
            """Same as _step_infer but for batched (B, n_pos, ...) tensors."""
            exc = pre @ filters_t
            not_refrac = refrac == 0
            v = torch.where(not_refrac, v * decay_v + leak_rest + exc, v)
            spikes = (v >= (v_thresh + theta)) & not_refrac
            post = spikes.float()
            spike_count = spike_count + post
            v = torch.where(spikes, v_reset_t, v)
            refrac = torch.where(spikes, refrac_time_t, refrac)
            theta = torch.where(spikes, theta + theta_plus, theta) * decay_theta
            v = v - post.sum(dim=2, keepdim=True) * inh_weight
            v = v - post.sum(dim=1, keepdim=True) * inh_global
            v = torch.where(spikes, v_reset_t, v)
            refrac = (refrac - 1).clamp(min=0)
            return v, refrac, theta, spike_count, post

        # Compile with default mode (Triton kernel fusion, no CUDA graphs)
        # "reduce-overhead" uses CUDA graphs which conflict with recurrent
        # state loops (outputs fed back as inputs). "default" mode fuses
        # small kernels into larger Triton kernels for the same benefit.
        self._compiled_step_train = torch.compile(
            _step_train, mode="default", fullgraph=True)
        self._compiled_step_infer = torch.compile(
            _step_infer, mode="default", fullgraph=True)
        self._compiled_step_infer_batch = torch.compile(
            _step_infer_batch, mode="default", fullgraph=True)
        self._compiled = True

    @torch.no_grad()
    def forward(self, spike_trains_4d, train=True):
        """
        spike_trains_4d: (n_steps, C, H, W) input spike trains on device.
        Returns: spike_count, pooled_flat, output_spike_trains_4d
        """
        if not self._compiled:
            self._build_compiled_steps()

        dev = self.device
        n_steps = spike_trains_4d.shape[0]
        n_pos = self.n_pos
        n_f = self.n_filters

        # Unfold input to patches: (n_steps, n_pos, k_sq)
        all_patches = F.unfold(spike_trains_4d, self.ks).permute(0, 2, 1).contiguous()

        # ---- Initialize state ----
        v = torch.full((n_pos, n_f), self.v_rest, device=dev)
        refrac = torch.zeros(n_pos, n_f, dtype=torch.int32, device=dev)
        theta = self.theta

        x_pre_bar = torch.zeros(n_pos, self.k_sq, device=dev)
        u_plus_bar = torch.zeros(n_pos, n_f, device=dev)
        u_minus_bar = torch.zeros(n_pos, n_f, device=dev)

        dw_accum = torch.zeros(n_f, self.k_sq, device=dev)
        spike_count = torch.zeros(n_pos, n_f, device=dev)
        output_spikes = torch.zeros(n_steps, n_pos, n_f, device=dev)

        filters_t = self.filters.t().contiguous()

        if train:
            for t in range(n_steps):
                (v, refrac, theta, x_pre_bar, u_plus_bar, u_minus_bar,
                 spike_count, dw_accum, post) = self._compiled_step_train(
                    all_patches[t], v, refrac, theta, x_pre_bar,
                    u_plus_bar, u_minus_bar, spike_count, dw_accum, filters_t)
                output_spikes[t] = post
        else:
            for t in range(n_steps):
                v, refrac, theta, spike_count, post = self._compiled_step_infer(
                    all_patches[t], v, refrac, theta, spike_count, filters_t)
                output_spikes[t] = post

        self.theta = theta

        if train:
            dw = dw_accum / n_pos
            self.filters.add_(dw).clamp_(0.0, 1.0)
            self._normalize_filters()

            # Accumulate diagnostics (single GPU->CPU sync per image, not per timestep)
            dw_pos = dw.clamp(min=0)
            dw_neg = dw.clamp(max=0)
            net_pos = float(dw_pos.sum())
            net_neg = float(dw_neg.abs().sum())
            _accumulate_layer_diagnostics(
                (self._diag, self._epoch_diag),
                {
                    "total_spikes": float(spike_count.sum()),
                    "total_input_spikes": float(all_patches.sum()),
                    "raw_ltp_mag": net_pos,
                    "raw_ltd_mag": net_neg,
                    "net_dw_pos_mag": net_pos,
                    "net_dw_neg_mag": net_neg,
                    "voltage_sum": float(v.mean()),
                    "voltage_max": float(v.max()),
                    "u_plus_sum": float(u_plus_bar.mean()),
                    "u_minus_sum": float(u_minus_bar.mean()),
                    "pre_trace_sum": float(x_pre_bar.mean()),
                    "dw_pos_sum": float(dw_pos.mean()),
                    "dw_neg_sum": float(dw_neg.abs().mean()),
                    "n_images": 1,
                },
            )

        # Max pool spike counts
        sc_map = spike_count.t().reshape(1, n_f, self.H_out, self.W_out)
        pooled = F.max_pool2d(sc_map, 2).squeeze(0)
        pooled_flat = pooled.reshape(-1).cpu()

        output_spikes_4d = output_spikes.permute(0, 2, 1).reshape(
            n_steps, n_f, self.H_out, self.W_out)

        return spike_count, pooled_flat, output_spikes_4d

    @torch.no_grad()
    def forward_batch(self, spike_trains_batch, return_spikes=True):
        """Batched forward for inference only (no STDP, no theta update).

        spike_trains_batch: (B, n_steps, C, H, W) on device.
        return_spikes: if True, returns full output spikes (needed for L1->L2).
                       Set False for last layer to save ~1.3 GB.
        Returns: pooled_batch (B, n_pooled),
                 output_spikes_4d (B, n_steps, n_f, H_out, W_out) or None
        """
        if not self._compiled:
            self._build_compiled_steps()

        dev = self.device
        B, n_steps, C, H, W = spike_trains_batch.shape
        n_pos = self.n_pos
        n_f = self.n_filters

        # --- Per-timestep unfolding to avoid OOM ---
        # Pre-unfolding all timesteps creates (B, T, n_pos, k_sq) which for L2
        # is 64*200*400*800*4 = 15.3 GiB. Instead, unfold one frame at a time
        # (~82 MB per step for L2).

        # State: batched
        v = torch.full((B, n_pos, n_f), self.v_rest, device=dev)
        refrac = torch.zeros(B, n_pos, n_f, dtype=torch.int32, device=dev)
        theta = self.theta.unsqueeze(0).expand(B, -1, -1).clone()

        filters_t = self.filters.t().contiguous()
        spike_count = torch.zeros(B, n_pos, n_f, device=dev)
        if return_spikes:
            output_spikes = torch.zeros(B, n_steps, n_pos, n_f, device=dev)

        for t in range(n_steps):
            torch.compiler.cudagraph_mark_step_begin()
            # Unfold one timestep at a time: (B, C, H, W) -> (B, n_pos, k_sq)
            frame = spike_trains_batch[:, t]               # (B, C, H, W)
            pre = F.unfold(frame, self.ks)                 # (B, k_sq, n_pos)
            pre = pre.permute(0, 2, 1).contiguous()        # (B, n_pos, k_sq)
            v, refrac, theta, spike_count, post = self._compiled_step_infer_batch(
                pre, v, refrac, theta, spike_count, filters_t)
            if return_spikes:
                output_spikes[:, t] = post

        # Max pool: (B, n_f, H_out, W_out)
        sc_map = spike_count.permute(0, 2, 1).reshape(B, n_f, self.H_out, self.W_out)
        pooled = F.max_pool2d(sc_map, 2)
        pooled_flat = pooled.reshape(B, -1)

        if return_spikes:
            out_4d = output_spikes.permute(0, 1, 3, 2).reshape(
                B, n_steps, n_f, self.H_out, self.W_out)
            return pooled_flat, out_4d
        else:
            return pooled_flat, None


# =============================================================================
# FAITHFUL CLOPATH et al. (2010) VOLTAGE-BASED STDP LAYER
# =============================================================================

class ConvClopathSTDPLayer:
    """Faithful Clopath et al. (2010) voltage-based STDP with simplified AdEx.

    Fixes 5 deviations from the original paper vs ConvVoltageSTDPLayer:

      1. AdEx neuron: exponential spike initiation + adaptation + afterpotential
      2. LTP: A+ * x_pre_bar * [u(t) - θ+]+ * [ū+(t) - θ-]+
         (instantaneous voltage gate × filtered voltage gate)
      3. LTD: A- * X_pre(t) * [ū-(t) - θ-]+
         (triggered at pre-spike ARRIVAL, not at pre silence)
      4. Threshold rectification [· - θ]+ instead of normalization
      5. Hard weight bounds [0, w_max] instead of w(1-w) soft bounds

    The STDP computation happens BEFORE spike reset so that the instantaneous
    voltage term [u(t) - θ+]+ captures the supra-threshold voltage at spike time.

    AdEx is simplified: exponential term for sharp spike initiation,
    spike-triggered adaptation (b), and depolarizing afterpotential (z-current).
    Full biophysical conductances/capacitances are not used since the framework
    operates in arbitrary voltage units (v_rest=-65, v_thresh=-52).
    """

    def __init__(self, n_input_channels, n_filters, kernel_size, input_h, input_w,
                 dt=1.0, sim_time=350.0, intensity=128.0, norm=10.0,
                 A_plus=6.0e-3, A_minus=3.0e-3,
                 tau_pre=16.8, tau_plus=7.0, tau_minus=10.0,
                 v_rest=-65.0, v_reset=-65.0, v_thresh=-52.0, tau_v=100.0,
                 theta_plus=0.05, tc_theta_decay=1e7,
                 inh_weight=17.5, refrac_time=5,
                 # --- AdEx parameters ---
                 delta_T=2.0, v_T_offset=-2.0,
                 a_adapt=0.0, b_adapt=0.5, tau_w_adapt=144.0,
                 I_sp=0.5, tau_z=40.0,
                 # --- Clopath thresholds ---
                 theta_ltp=None, theta_ltd=None,
                 w_max=1.0,
                 # --- LTD mode ---
                 ltd_mode='spike',  # 'voltage' (original Clopath) or 'spike' (WTA-compatible)
                 use_weight_norm=True,  # False for faithful Clopath (hard bounds only)
                 theta_init=0.0,   # Initial adaptive threshold (high = suppress initial transient)
                 device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_input_ch = n_input_channels
        self.n_filters = n_filters
        self.ks = kernel_size
        self.k_sq = kernel_size * kernel_size * n_input_channels
        self.dt = dt
        self.n_steps = int(sim_time / dt)
        self.intensity = intensity
        self.norm = norm
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.w_max = w_max

        self.H_out = input_h - kernel_size + 1
        self.W_out = input_w - kernel_size + 1
        self.n_pos = self.H_out * self.W_out
        self.pool_h = self.H_out // 2
        self.pool_w = self.W_out // 2

        # Precompute decay factors
        self.decay_v = math.exp(-dt / tau_v)
        self.decay_theta = math.exp(-dt / tc_theta_decay)
        self.decay_pre = math.exp(-dt / tau_pre)
        self.decay_plus = math.exp(-dt / tau_plus)
        self.decay_minus = math.exp(-dt / tau_minus)
        self.decay_z = math.exp(-dt / tau_z)
        self.decay_w_adapt = math.exp(-dt / tau_w_adapt)

        # Voltage STDP amplitudes
        self.A_plus = A_plus
        self.A_minus = A_minus

        # Clopath voltage thresholds for rectification
        # theta_ltp: absolute voltage units (compared against raw v)
        # theta_ltd: RELATIVE to v_rest (compared against voltage traces which
        #            track v - v_rest). Default 0.0 = no LTD below resting potential.
        self.theta_ltp = theta_ltp if theta_ltp is not None else v_thresh
        self.theta_ltd = theta_ltd if theta_ltd is not None else 0.0
        if ltd_mode not in ("voltage", "spike"):
            raise ValueError("ltd_mode must be 'voltage' or 'spike'")
        self.ltd_mode = ltd_mode
        self.use_weight_norm = use_weight_norm

        # AdEx parameters
        self.delta_T = delta_T
        self.v_T = v_thresh + v_T_offset  # spike initiation threshold
        self.a_adapt = a_adapt
        self.b_adapt = b_adapt
        self.I_sp = I_sp

        # Inhibition / refractory
        self.inh_weight = inh_weight
        self.theta_plus = theta_plus
        self._v_reset_t = torch.tensor(v_reset, device=self.device)
        self._refrac_time_t = torch.tensor(refrac_time, dtype=torch.int32, device=self.device)

        # Filter weights
        self.filters = torch.rand(n_filters, self.k_sq, device=self.device) * 0.3
        if self.use_weight_norm:
            self._normalize_filters()

        # Adaptive threshold (initialized high to suppress initial transient
        # when no WTA is used — lets theta decay to natural levels)
        self.theta = torch.full((self.n_pos, n_filters), theta_init,
                                device=self.device)

        # torch.compile: built lazily on first forward()
        self._compiled = False

        # Diagnostics accumulators: interval resets for frequent logs, epoch
        # resets only at epoch summaries.
        self._reset_diagnostics(scope="all")

    def _reset_diagnostics(self, scope="interval"):
        if scope in ("all", "interval"):
            self._diag = _new_layer_diagnostics()
        if scope in ("all", "epoch"):
            self._epoch_diag = _new_layer_diagnostics()

    def get_diagnostics(self, scope="interval", reset=True):
        """Return accumulated diagnostics for interval or epoch scope."""
        source = self._epoch_diag if scope == "epoch" else self._diag
        d = _finalize_layer_diagnostics(self, source)
        if reset:
            self._reset_diagnostics(scope=scope)
        return d

    def _normalize_filters(self):
        if self.norm is not None:
            row_sums = self.filters.sum(1, keepdim=True).clamp_(min=1e-8)
            self.filters.mul_(self.norm / row_sums)
            self.filters.clamp_(0.0, self.w_max)

    def _build_compiled_steps(self):
        """Build torch.compiled timestep functions for Clopath STDP."""
        # Capture all scalar constants in closure
        decay_pre = self.decay_pre
        decay_v = self.decay_v
        decay_plus = self.decay_plus
        decay_minus = self.decay_minus
        decay_theta = self.decay_theta
        decay_z = self.decay_z
        decay_w_adapt = self.decay_w_adapt
        v_rest = self.v_rest
        v_thresh = self.v_thresh
        leak_rest = (1 - decay_v) * v_rest
        one_minus_dp = 1 - decay_plus
        one_minus_dm = 1 - decay_minus
        inh_weight = self.inh_weight
        inh_global = inh_weight * 0.1
        theta_plus_val = self.theta_plus
        v_reset_t = self._v_reset_t
        refrac_time_t = self._refrac_time_t
        A_plus = self.A_plus
        A_minus = self.A_minus
        theta_ltp = self.theta_ltp
        theta_ltd = self.theta_ltd
        delta_T = self.delta_T
        v_T = self.v_T
        a_adapt = self.a_adapt
        a_adapt_dt = a_adapt * (1 - decay_w_adapt)  # discretized coupling
        b_adapt = self.b_adapt
        I_sp = self.I_sp
        v_scale = abs(v_thresh - v_rest)  # max voltage gate magnitude
        use_spike_ltd = (self.ltd_mode == 'spike')

        inv_vscale = 1.0 / max(v_scale, 1e-6)

        def _step_train(pre_patches, v, refrac, theta, x_pre_bar,
                        u_plus_bar, u_minus_bar, spike_count, dw_accum,
                        raw_ltp_accum, raw_ltd_accum, filters_t,
                        w_adapt, z_after):
            # 1. Pre trace
            x_pre_bar = x_pre_bar * decay_pre + pre_patches

            # 2. AdEx neuron dynamics
            exc_input = pre_patches @ filters_t
            not_refrac = refrac == 0
            # Exponential spike initiation: only active above v_T, clamped
            # to prevent runaway (in biological AdEx, the spike mechanism
            # cuts off; here we cap the exponential contribution)
            exp_arg = ((v - v_T) / delta_T).clamp(max=3.0)  # exp(3)≈20
            exp_term = delta_T * torch.exp(exp_arg)
            # Cap total voltage to prevent NaN/inf from positive feedback
            v_new = v * decay_v + leak_rest + exc_input + exp_term - w_adapt + z_after
            v_new = v_new.clamp(min=v_rest - 100.0, max=v_thresh + 50.0)
            v = torch.where(not_refrac, v_new, v)
            # Adaptation dynamics
            w_adapt = w_adapt * decay_w_adapt + a_adapt_dt * (v - v_rest)
            # Afterpotential decay
            z_after = z_after * decay_z

            # 3. Voltage traces (deviation from rest, clamped to [0, v_scale])
            # Upper clamp: AdEx exponential can push v far above threshold
            # before discrete-time spike detection; unclamped v_dev up to 63
            # pumps traces to 20+, causing LTP magnitudes to explode (dw+ > 200).
            # Lower clamp: lateral inhibition + adaptation push v below v_rest;
            # negative traces would poison LTD rectification.
            v_dev = (v - v_rest).clamp(min=0, max=v_scale)
            u_plus_bar = u_plus_bar * decay_plus + v_dev * one_minus_dp
            u_minus_bar = u_minus_bar * decay_minus + v_dev * one_minus_dm

            # 4. Spike detection
            spikes = (v >= (v_thresh + theta)) & not_refrac
            post = spikes.float()
            spike_count = spike_count + post

            # 5. Faithful Clopath STDP (BEFORE reset)
            #
            # LTP: A+ * x̄_pre * [v(t) - θ_ltp]+ * [ū+(t) - θ_ltd]+
            #   Double voltage gate, normalized to [0,1] each.
            #   v_rect: instantaneous gate — only >0 at/near spike (v > θ_ltp)
            #   u_plus_rect: filtered gate — recent depolarization history
            #   These naturally restrict LTP to spiking neurons.
            #
            # LTD: A- * X_pre(t) * [ū-(t) - θ_ltd]+
            #   Voltage-gated (original Clopath, NOT silence-based).
            #   Requires soft lateral inhibition so non-winners retain
            #   elevated ū⁻. With hard WTA, all non-winners have ū⁻ ≈ 0
            #   → LTD = 0 → weights collapse. Soft inhibition preserves
            #   subthreshold voltage traces for proper LTD function.
            v_rect = (v - theta_ltp).clamp(min=0, max=v_scale) * inv_vscale
            u_plus_rect = (u_plus_bar - theta_ltd).clamp(min=0, max=v_scale) * inv_vscale
            ltp_coeff = v_rect * u_plus_rect                 # (n_pos, n_filters)
            dw_ltp = A_plus * (ltp_coeff.t() @ x_pre_bar)   # (n_filters, k_sq)

            u_minus_rect = (u_minus_bar - theta_ltd).clamp(min=0, max=v_scale) * inv_vscale
            if use_spike_ltd:
                # Compatibility mode used by earlier WTA experiments:
                # depress silent afferents only when the postsynaptic cell spikes.
                no_pre_trace = (x_pre_bar <= 1e-6).float()
                dw_ltd = A_minus * ((post * u_minus_rect).t() @ no_pre_trace)
            else:
                # Original Clopath LTD: pre-spike-triggered and voltage gated.
                dw_ltd = A_minus * (u_minus_rect.t() @ pre_patches)

            dw_accum = dw_accum + dw_ltp - dw_ltd
            raw_ltp_accum = raw_ltp_accum + dw_ltp.sum()
            raw_ltd_accum = raw_ltd_accum + dw_ltd.sum()

            # 6. Reset, refractory, adaptation, afterpotential
            v = torch.where(spikes, v_reset_t, v)
            refrac = torch.where(spikes, refrac_time_t, refrac)
            theta = torch.where(spikes, theta + theta_plus_val, theta) * decay_theta
            w_adapt = torch.where(spikes, w_adapt + b_adapt, w_adapt)
            z_after = torch.where(spikes, z_after + I_sp, z_after)

            # 7. Soft lateral inhibition (NOT hard WTA reset)
            # Suppresses non-winners with inhibitory current but preserves
            # their voltage traces for Clopath's voltage-gated LTD.
            v = v - post.sum(dim=1, keepdim=True) * inh_weight
            v = v - post.sum(dim=0, keepdim=True) * inh_global
            v = torch.where(spikes, v_reset_t, v)
            refrac = (refrac - 1).clamp(min=0)

            return (v, refrac, theta, x_pre_bar, u_plus_bar, u_minus_bar,
                    spike_count, dw_accum, raw_ltp_accum, raw_ltd_accum,
                    post, w_adapt, z_after)

        def _step_infer(pre_patches, v, refrac, theta, spike_count,
                        filters_t, w_adapt, z_after):
            exc_input = pre_patches @ filters_t
            not_refrac = refrac == 0
            exp_arg = ((v - v_T) / delta_T).clamp(max=3.0)
            exp_term = delta_T * torch.exp(exp_arg)
            v_new = v * decay_v + leak_rest + exc_input + exp_term - w_adapt + z_after
            v_new = v_new.clamp(min=v_rest - 100.0, max=v_thresh + 50.0)
            v = torch.where(not_refrac, v_new, v)
            w_adapt = w_adapt * decay_w_adapt + a_adapt_dt * (v - v_rest)
            z_after = z_after * decay_z
            spikes = (v >= (v_thresh + theta)) & not_refrac
            post = spikes.float()
            spike_count = spike_count + post
            v = torch.where(spikes, v_reset_t, v)
            refrac = torch.where(spikes, refrac_time_t, refrac)
            theta = torch.where(spikes, theta + theta_plus_val, theta) * decay_theta
            w_adapt = torch.where(spikes, w_adapt + b_adapt, w_adapt)
            z_after = torch.where(spikes, z_after + I_sp, z_after)
            v = v - post.sum(dim=1, keepdim=True) * inh_weight
            v = v - post.sum(dim=0, keepdim=True) * inh_global
            v = torch.where(spikes, v_reset_t, v)
            refrac = (refrac - 1).clamp(min=0)
            return v, refrac, theta, spike_count, post, w_adapt, z_after

        def _step_infer_batch(pre, v, refrac, theta, spike_count,
                              filters_t, w_adapt, z_after):
            exc = pre @ filters_t
            not_refrac = refrac == 0
            exp_arg = ((v - v_T) / delta_T).clamp(max=3.0)
            exp_term = delta_T * torch.exp(exp_arg)
            v_new = v * decay_v + leak_rest + exc + exp_term - w_adapt + z_after
            v_new = v_new.clamp(min=v_rest - 100.0, max=v_thresh + 50.0)
            v = torch.where(not_refrac, v_new, v)
            w_adapt = w_adapt * decay_w_adapt + a_adapt_dt * (v - v_rest)
            z_after = z_after * decay_z
            spikes = (v >= (v_thresh + theta)) & not_refrac
            post = spikes.float()
            spike_count = spike_count + post
            v = torch.where(spikes, v_reset_t, v)
            refrac = torch.where(spikes, refrac_time_t, refrac)
            theta = torch.where(spikes, theta + theta_plus_val, theta) * decay_theta
            w_adapt = torch.where(spikes, w_adapt + b_adapt, w_adapt)
            z_after = torch.where(spikes, z_after + I_sp, z_after)
            v = v - post.sum(dim=2, keepdim=True) * inh_weight
            v = v - post.sum(dim=1, keepdim=True) * inh_global
            v = torch.where(spikes, v_reset_t, v)
            refrac = (refrac - 1).clamp(min=0)
            return v, refrac, theta, spike_count, post, w_adapt, z_after

        self._compiled_step_train = torch.compile(
            _step_train, mode="default", fullgraph=True)
        self._compiled_step_infer = torch.compile(
            _step_infer, mode="default", fullgraph=True)
        self._compiled_step_infer_batch = torch.compile(
            _step_infer_batch, mode="default", fullgraph=True)
        self._compiled = True

    @torch.no_grad()
    def forward(self, spike_trains_4d, train=True):
        """Forward pass — same interface as ConvVoltageSTDPLayer."""
        if not self._compiled:
            self._build_compiled_steps()

        dev = self.device
        n_steps = spike_trains_4d.shape[0]
        n_pos = self.n_pos
        n_f = self.n_filters

        all_patches = F.unfold(spike_trains_4d, self.ks).permute(0, 2, 1).contiguous()

        # Initialize state
        v = torch.full((n_pos, n_f), self.v_rest, device=dev)
        refrac = torch.zeros(n_pos, n_f, dtype=torch.int32, device=dev)
        theta = self.theta
        x_pre_bar = torch.zeros(n_pos, self.k_sq, device=dev)
        u_plus_bar = torch.zeros(n_pos, n_f, device=dev)
        u_minus_bar = torch.zeros(n_pos, n_f, device=dev)
        w_adapt = torch.zeros(n_pos, n_f, device=dev)
        z_after = torch.zeros(n_pos, n_f, device=dev)
        dw_accum = torch.zeros(n_f, self.k_sq, device=dev)
        raw_ltp_accum = torch.zeros((), device=dev)
        raw_ltd_accum = torch.zeros((), device=dev)
        spike_count = torch.zeros(n_pos, n_f, device=dev)
        output_spikes = torch.zeros(n_steps, n_pos, n_f, device=dev)
        filters_t = self.filters.t().contiguous()

        if train:
            for t in range(n_steps):
                (v, refrac, theta, x_pre_bar, u_plus_bar, u_minus_bar,
                 spike_count, dw_accum, raw_ltp_accum, raw_ltd_accum,
                 post, w_adapt, z_after
                 ) = self._compiled_step_train(
                    all_patches[t], v, refrac, theta, x_pre_bar,
                    u_plus_bar, u_minus_bar, spike_count, dw_accum,
                    raw_ltp_accum, raw_ltd_accum,
                    filters_t, w_adapt, z_after)
                output_spikes[t] = post
        else:
            for t in range(n_steps):
                (v, refrac, theta, spike_count, post,
                 w_adapt, z_after) = self._compiled_step_infer(
                    all_patches[t], v, refrac, theta, spike_count,
                    filters_t, w_adapt, z_after)
                output_spikes[t] = post

        self.theta = theta

        if train:
            dw = dw_accum / n_pos
            self.filters.add_(dw).clamp_(0.0, self.w_max)
            if self.use_weight_norm:
                self._normalize_filters()

            # Diagnostics
            dw_pos = dw.clamp(min=0)
            dw_neg = dw.clamp(max=0)
            _accumulate_layer_diagnostics(
                (self._diag, self._epoch_diag),
                {
                    "total_spikes": float(spike_count.sum()),
                    "total_input_spikes": float(all_patches.sum()),
                    "raw_ltp_mag": float(raw_ltp_accum / n_pos),
                    "raw_ltd_mag": float(raw_ltd_accum / n_pos),
                    "net_dw_pos_mag": float(dw_pos.sum()),
                    "net_dw_neg_mag": float(dw_neg.abs().sum()),
                    "voltage_sum": float(v.mean()),
                    "voltage_max": float(v.max()),
                    "u_plus_sum": float(u_plus_bar.mean()),
                    "u_minus_sum": float(u_minus_bar.mean()),
                    "pre_trace_sum": float(x_pre_bar.mean()),
                    "dw_pos_sum": float(dw_pos.mean()),
                    "dw_neg_sum": float(dw_neg.abs().mean()),
                    "n_images": 1,
                },
            )

        # Max pool spike counts
        sc_map = spike_count.t().reshape(1, n_f, self.H_out, self.W_out)
        pooled = F.max_pool2d(sc_map, 2).squeeze(0)
        pooled_flat = pooled.reshape(-1).cpu()

        output_spikes_4d = output_spikes.permute(0, 2, 1).reshape(
            n_steps, n_f, self.H_out, self.W_out)

        return spike_count, pooled_flat, output_spikes_4d

    @torch.no_grad()
    def forward_batch(self, spike_trains_batch, return_spikes=True):
        """Batched forward for inference (no STDP). Same interface."""
        if not self._compiled:
            self._build_compiled_steps()

        dev = self.device
        B, n_steps, C, H, W = spike_trains_batch.shape
        n_pos = self.n_pos
        n_f = self.n_filters

        v = torch.full((B, n_pos, n_f), self.v_rest, device=dev)
        refrac = torch.zeros(B, n_pos, n_f, dtype=torch.int32, device=dev)
        theta = self.theta.unsqueeze(0).expand(B, -1, -1).clone()
        w_adapt = torch.zeros(B, n_pos, n_f, device=dev)
        z_after = torch.zeros(B, n_pos, n_f, device=dev)
        filters_t = self.filters.t().contiguous()
        spike_count = torch.zeros(B, n_pos, n_f, device=dev)
        if return_spikes:
            output_spikes = torch.zeros(B, n_steps, n_pos, n_f, device=dev)

        for t in range(n_steps):
            torch.compiler.cudagraph_mark_step_begin()
            frame = spike_trains_batch[:, t]
            pre = F.unfold(frame, self.ks).permute(0, 2, 1).contiguous()
            (v, refrac, theta, spike_count, post,
             w_adapt, z_after) = self._compiled_step_infer_batch(
                pre, v, refrac, theta, spike_count,
                filters_t, w_adapt, z_after)
            if return_spikes:
                output_spikes[:, t] = post

        sc_map = spike_count.permute(0, 2, 1).reshape(B, n_f, self.H_out, self.W_out)
        pooled = F.max_pool2d(sc_map, 2)
        pooled_flat = pooled.reshape(B, -1)

        if return_spikes:
            out_4d = output_spikes.permute(0, 1, 3, 2).reshape(
                B, n_steps, n_f, self.H_out, self.W_out)
            return pooled_flat, out_4d
        else:
            return pooled_flat, None


# =============================================================================
# 2-LAYER CONV VOLTAGE STDP + SURROGATE READOUT
# =============================================================================

class TwoLayerConvVoltageSTDP:
    """2-layer conv voltage STDP network built from config."""

    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg

        data_cfg = cfg['data']
        voltage_cfg = cfg.get('voltage', {})
        l1_cfg = cfg['model']['layer1']
        l2_cfg = cfg['model']['layer2']
        readout_cfg = cfg.get('readout', {})

        self.sim_time = data_cfg['sim_time']
        self.intensity = data_cfg.get('intensity', 128.0)
        self.n_steps = int(self.sim_time)

        # Input spatial dimensions (per-dataset). MNIST/FMNIST default to
        # 1×28×28; CIFAR-10 RGB needs 3×32×32. Settable via config:
        #   data.input_h, data.input_w, data.input_channels
        # (Plus dog adds channels — see preprocess block below.)
        self.input_h = int(data_cfg.get('input_h', 28))
        self.input_w = int(data_cfg.get('input_w', 28))
        self.input_channels = int(data_cfg.get('input_channels', 1))

        # Optional DoG pre-filter on input images (before Poisson encoding).
        # Default "none" => behavior identical to baseline. Set
        # data.preprocess: "dog" to enable a single ON-center DoG channel
        # with per-image min-max normalization back to [0, 1].
        self.preprocess = str(data_cfg.get('preprocess', 'none')).lower()
        self.dog_kernels = None       # list of (groups, 1, k_i, k_i) kernels, one per scale
        self.dog_kernel = None        # backward-compat single-scale (= dog_kernels[0])
        self.dog_mode = 'on'
        self.dog_opponent = False
        self.in_channels = self.input_channels  # Default = raw input channels
        if self.preprocess == 'dog':
            dog_cfg = data_cfg.get('dog', {}) or {}

            # NEW: opponent-channel transform (RGB → (R-G, B-Y, L)).
            # Requires input_channels=3. Mozafari/Falez/Kheradpisheh style.
            self.dog_opponent = bool(dog_cfg.get('opponent', False))
            if self.dog_opponent and self.input_channels != 3:
                raise ValueError(f"dog.opponent=true requires input_channels=3; "
                                 f"got {self.input_channels}")

            # NEW: multi-scale DoG via list of [sigma1, sigma2] pairs.
            # Backwards-compat: if not provided, falls back to single scale
            # using the existing sigma1/sigma2/kernel_size keys.
            scales_cfg = dog_cfg.get('scales', None)
            if scales_cfg is None:
                # legacy single-scale spec
                k_legacy = int(dog_cfg.get('kernel_size', 7))
                s1_legacy = float(dog_cfg.get('sigma1', 1.0))
                s2_legacy = float(dog_cfg.get('sigma2', 2.0))
                scales_spec = [(s1_legacy, s2_legacy, k_legacy)]
            else:
                # multi-scale: each entry is [s1, s2] or [s1, s2, k]
                scales_spec = []
                for entry in scales_cfg:
                    if len(entry) == 2:
                        s1, s2 = entry
                        # Choose a kernel size that covers ~3σ around the larger
                        # sigma; round up to nearest odd integer >= 7.
                        k_auto = int(max(7, np.ceil(6 * float(s2))))
                        if k_auto % 2 == 0:
                            k_auto += 1
                        scales_spec.append((float(s1), float(s2), k_auto))
                    elif len(entry) == 3:
                        s1, s2, k_i = entry
                        scales_spec.append((float(s1), float(s2), int(k_i)))
                    else:
                        raise ValueError(f"dog.scales entry must be [s1,s2] or "
                                         f"[s1,s2,k]; got {entry}")

            # Number of "raw input channels to filter":
            #   if opponent: 3 (R-G, B-Y, L)
            #   else:        input_channels (per-channel DoG, e.g. RGB → 3)
            opp_or_raw_channels = 3 if self.dog_opponent else self.input_channels

            # Build one DoG kernel per scale, replicated across channels for
            # grouped conv2d (groups = opp_or_raw_channels).
            self.dog_kernels = []     # list of tensor kernels
            self.dog_pads = []        # list of padding sizes (k_i // 2)
            self.dog_scale_descs = [] # human-readable
            for s1, s2, k_i in scales_spec:
                r = np.arange(k_i) + 1
                xg = np.tile(r, [k_i, 1])
                yg = xg.T
                d2 = (xg - k_i / 2.0 - 0.5) ** 2 + (yg - k_i / 2.0 - 0.5) ** 2
                filt = (1.0 / np.sqrt(2 * np.pi)) * (
                    (1.0 / s1) * np.exp(-d2 / (2 * s1 ** 2))
                    - (1.0 / s2) * np.exp(-d2 / (2 * s2 ** 2))
                )
                filt -= filt.mean()
                filt /= np.max(np.abs(filt))
                w_1c = torch.from_numpy(filt).float().view(1, 1, k_i, k_i).to(device)
                if opp_or_raw_channels > 1:
                    w = w_1c.expand(opp_or_raw_channels, 1, k_i, k_i).contiguous()
                else:
                    w = w_1c
                self.dog_kernels.append(w)
                self.dog_pads.append(k_i // 2)
                self.dog_scale_descs.append(f"(s={s1:.2f},{s2:.2f},k={k_i})")

            # Keep dog_kernel pointing at the first scale (backward-compat for
            # any code path that checks `is not None`).
            self.dog_kernel = self.dog_kernels[0]
            self.dog_pad = self.dog_pads[0]

            # Threshold-relative normalization knobs.
            self.dog_threshold = float(dog_cfg.get('threshold', 0.0))
            self.dog_scale_mode = str(dog_cfg.get('scale_mode', 'per_image')).lower()
            self.dog_scale = float(dog_cfg.get('dog_scale', 1.0))
            self.dog_subtract_threshold = bool(dog_cfg.get('subtract_threshold', True))
            self.dog_mode = str(dog_cfg.get('mode', 'on')).lower()  # 'on' or 'on_off'
            if self.dog_mode not in ('on', 'on_off'):
                raise ValueError(f"dog.mode must be 'on' or 'on_off', got {self.dog_mode}")

            # Final in_channels after DoG:
            #   = opp_or_raw_channels × num_scales × (2 if on_off else 1)
            num_scales = len(self.dog_kernels)
            polarity_mult = 2 if self.dog_mode == 'on_off' else 1
            self.in_channels = opp_or_raw_channels * num_scales * polarity_mult

            print(f"  [DoG] Pre-filter enabled: opponent={self.dog_opponent}, "
                  f"scales={self.dog_scale_descs}, mode={self.dog_mode}, "
                  f"threshold={self.dog_threshold}, scale_mode={self.dog_scale_mode}")
            print(f"  [DoG] input_channels={self.input_channels} → "
                  f"opp_or_raw={opp_or_raw_channels} × scales={num_scales} × "
                  f"polarity={polarity_mult} → dog_channels={self.in_channels}")

        # Shared voltage defaults (can be overridden per-layer)
        v_defaults = {
            'A_plus': voltage_cfg.get('A_plus', 0.006),
            'A_minus': voltage_cfg.get('A_minus', 0.003),
            'tau_pre': voltage_cfg.get('tau_pre', 16.8),
            'tau_plus': voltage_cfg.get('tau_plus', 3.0),
            'tau_minus': voltage_cfg.get('tau_minus', 8.0),
        }

        # Select layer class based on voltage rule
        voltage_rule = voltage_cfg.get('rule', 'approximate')
        if voltage_rule == 'clopath':
            LayerClass = ConvClopathSTDPLayer
            adex_cfg = voltage_cfg.get('adex', {})
            extra_kwargs = {
                'delta_T': adex_cfg.get('delta_T', 2.0),
                'v_T_offset': adex_cfg.get('v_T_offset', -2.0),
                'a_adapt': adex_cfg.get('a_adapt', 0.0),
                'b_adapt': adex_cfg.get('b_adapt', 0.5),
                'tau_w_adapt': adex_cfg.get('tau_w_adapt', 144.0),
                'I_sp': adex_cfg.get('I_sp', 0.5),
                'tau_z': adex_cfg.get('tau_z', 40.0),
                'theta_ltp': voltage_cfg.get('theta_ltp', None),
                'theta_ltd': voltage_cfg.get('theta_ltd', None),
                'w_max': voltage_cfg.get('w_max', 1.0),
                'ltd_mode': voltage_cfg.get('ltd_mode', 'spike'),
                'use_weight_norm': voltage_cfg.get('use_weight_norm', True),
            }
            print(f"  [Clopath] Using faithful Clopath STDP (AdEx + voltage thresholds)")
        else:
            LayerClass = ConvVoltageSTDPLayer
            extra_kwargs = {}
            print(f"  [Voltage] Using approximate voltage-modulated STDP")

        # Layer 1
        ks1 = l1_cfg['kernel_size']
        # theta_init: per-layer override for initial adaptive threshold
        l1_extra = dict(extra_kwargs)
        l1_extra['theta_init'] = l1_cfg.get('theta_init', 0.0)
        self.layer1 = LayerClass(
            n_input_channels=self.in_channels,  # 1 normally, 2 when DoG mode='on_off', 3 for RGB
            n_filters=l1_cfg['n_filters'],
            kernel_size=ks1,
            input_h=self.input_h, input_w=self.input_w,
            sim_time=self.sim_time,
            intensity=self.intensity,
            norm=l1_cfg.get('norm', 10.0),
            A_plus=l1_cfg.get('A_plus', v_defaults['A_plus']),
            A_minus=l1_cfg.get('A_minus', v_defaults['A_minus']),
            tau_pre=l1_cfg.get('tau_pre', v_defaults['tau_pre']),
            tau_plus=l1_cfg.get('tau_plus', v_defaults['tau_plus']),
            tau_minus=l1_cfg.get('tau_minus', v_defaults['tau_minus']),
            v_rest=l1_cfg.get('v_rest', -65.0),
            v_reset=l1_cfg.get('v_reset', -65.0),
            v_thresh=l1_cfg.get('v_thresh', -52.0),
            tau_v=l1_cfg.get('tau_v', 100.0),
            theta_plus=l1_cfg.get('theta_plus', 0.05),
            tc_theta_decay=l1_cfg.get('tc_theta_decay', 1e7),
            inh_weight=l1_cfg.get('inh_weight', 17.5),
            refrac_time=l1_cfg.get('refrac_time', 5),
            device=device,
            **l1_extra,
        )
        l1_h, l1_w = self.layer1.H_out, self.layer1.W_out

        # Layer 2
        ks2 = l2_cfg['kernel_size']
        l2_extra = dict(extra_kwargs)
        l2_extra['theta_init'] = l2_cfg.get('theta_init', 0.0)
        self.layer2 = LayerClass(
            n_input_channels=l1_cfg['n_filters'],
            n_filters=l2_cfg['n_filters'],
            kernel_size=ks2,
            input_h=l1_h, input_w=l1_w,
            sim_time=self.sim_time,
            intensity=1.0,
            norm=l2_cfg.get('norm', 20.0),
            A_plus=l2_cfg.get('A_plus', v_defaults['A_plus']),
            A_minus=l2_cfg.get('A_minus', v_defaults['A_minus']),
            tau_pre=l2_cfg.get('tau_pre', v_defaults['tau_pre']),
            tau_plus=l2_cfg.get('tau_plus', v_defaults['tau_plus']),
            tau_minus=l2_cfg.get('tau_minus', v_defaults['tau_minus']),
            v_rest=l2_cfg.get('v_rest', -65.0),
            v_reset=l2_cfg.get('v_reset', -65.0),
            v_thresh=l2_cfg.get('v_thresh', -52.0),
            tau_v=l2_cfg.get('tau_v', 100.0),
            theta_plus=l2_cfg.get('theta_plus', 0.05),
            tc_theta_decay=l2_cfg.get('tc_theta_decay', 1e7),
            inh_weight=l2_cfg.get('inh_weight', 17.5),
            refrac_time=l2_cfg.get('refrac_time', 5),
            device=device,
            **l2_extra,
        )
        l2_h, l2_w = self.layer2.H_out, self.layer2.W_out

        # Feature dimensions
        self.n_feat_l1 = l1_cfg['n_filters'] * (l1_h // 2) * (l1_w // 2)
        self.n_feat_l2 = l2_cfg['n_filters'] * (l2_h // 2) * (l2_w // 2)
        self.n_features = self.n_feat_l1 + self.n_feat_l2

        # Readout SNN
        self.readout = ReadoutSNN(
            n_input=self.n_features,
            n_hidden=readout_cfg.get('hidden_size', 256),
            n_output=10,
            n_steps=readout_cfg.get('n_steps', 50),
            tau=readout_cfg.get('tau', 10.0),
        ).to(device)

        train_cfg = cfg['training']
        opt_name = readout_cfg.get('optimizer', 'adam').lower()
        lr = train_cfg.get('readout_lr', 1e-3)
        wd = readout_cfg.get('weight_decay', 0.0)
        if opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.readout.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(
                self.readout.parameters(), lr=lr, weight_decay=wd)
        label_smoothing = readout_cfg.get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # LR scheduler is built later (needs readout_epochs) by the training loop

    def _rgb_to_opponent(self, img_4d):
        """Convert (N, 3, H, W) RGB in [0,1] to (N, 3, H, W) opponent channels:
        ch0 = (R - G) / 2,
        ch1 = (B - (R+G)/2) / 2,
        ch2 = (R + G + B) / 3      (luminance).
        Each opponent channel is then per-image min-max renormalized to [0, 1]
        so downstream DoG thresholding/normalisation behaves comparably across
        channels.
        """
        R = img_4d[:, 0:1]
        G = img_4d[:, 1:2]
        B = img_4d[:, 2:3]
        ch0 = (R - G) * 0.5
        ch1 = (B - (R + G) * 0.5) * 0.5
        ch2 = (R + G + B) / 3.0
        opp = torch.cat([ch0, ch1, ch2], dim=1)  # (N, 3, H, W)
        # Per-image, per-channel min-max to [0, 1]
        N, C, H, W = opp.shape
        flat = opp.view(N, C, -1)
        mn = flat.min(dim=2, keepdim=True).values
        mx = flat.max(dim=2, keepdim=True).values
        denom = (mx - mn).clamp_min(1e-8)
        opp = ((flat - mn) / denom).view(N, C, H, W)
        return opp

    def _apply_dog(self, img_4d):
        """Apply DoG + border mask + threshold + scale.

        img_4d: (N, input_channels, H, W) tensor on device.
        Returns (N, in_channels, H, W) where in_channels depends on:
          - opponent (True/False)
          - number of scales
          - on / on_off mode

        Pipeline:
          1. If opponent: convert (N, 3, H, W) RGB → opponent (N, 3, H, W).
          2. For each DoG scale: apply kernel via grouped conv2d.
          3. Border mask + threshold + normalise per scale.
          4. If on_off: append the negated response.
          5. Concatenate across scales along channel dim.
        """
        if not self.dog_kernels:
            return img_4d

        # 1. Opponent transform (if enabled)
        if self.dog_opponent:
            base = self._rgb_to_opponent(img_4d)  # (N, 3, H, W)
            groups = 3
        else:
            base = img_4d
            groups = self.input_channels if self.dog_kernels[0].shape[0] > 1 else 1

        # 2-4. Per-scale DoG → threshold/normalise → optional on_off
        per_scale_outputs = []
        for kernel, pad in zip(self.dog_kernels, self.dog_pads):
            raw_scale = F.conv2d(base, kernel, padding=pad, groups=groups)
            per_scale_outputs.append(raw_scale)

        # Apply border mask + threshold + scale to all scales jointly.
        # We process the concatenated tensor for efficiency.
        raw = torch.cat(per_scale_outputs, dim=1)  # (N, groups*num_scales, H, W)
        # Border mask: zero outer 5 pixels.
        b = 5
        h, w = raw.shape[-2:]
        if h > 2 * b and w > 2 * b:
            mask = torch.zeros_like(raw)
            mask[..., b:h - b, b:w - b] = 1.0
            raw = raw * mask

        def _process(channel):
            """Apply threshold + scale to a signed response. Returns [0,1] tensor."""
            thr = self.dog_threshold
            out = torch.where(channel >= thr, channel, torch.zeros_like(channel))
            if self.dog_scale_mode == 'fixed':
                if self.dog_subtract_threshold and thr > 0.0:
                    out = (out - thr).clamp_min(0.0)
                out = out / max(self.dog_scale, 1e-8)
                out = out.clamp(0.0, 1.0)
            else:  # per_image
                N = out.shape[0]
                flat = out.view(N, -1)
                mn = flat.min(dim=1, keepdim=True).values
                mx = flat.max(dim=1, keepdim=True).values
                denom = (mx - mn).clamp_min(1e-8)
                out = ((flat - mn) / denom).view_as(out)
            return out

        on = _process(raw)
        if self.dog_mode == 'on_off':
            off = _process(-raw)  # OFF = negated ON-center response
            return torch.cat([on, off], dim=1)  # (N, 2, H, W)
        return on

    def _poisson_encode(self, image):
        """Encode a single image as Poisson spike trains: (n_steps, C, H, W).

        C = self.in_channels (1 grayscale / 2 DoG-on_off / 3 RGB CIFAR).
        H, W = self.input_h, self.input_w (28 for MNIST/FMNIST; 32 for CIFAR).
        """
        H, W, Cin = self.input_h, self.input_w, self.input_channels
        img_2d = image.view(1, Cin, H, W).to(self.device)
        img_2d = self._apply_dog(img_2d)  # (1, C, H, W) — no-op if preprocess=none
        C = img_2d.shape[1]
        rates = (img_2d * self.intensity / 1000.0).clamp_(0, 1)
        spike_trains = torch.bernoulli(rates.expand(self.n_steps, C, H, W))
        return spike_trains

    def _poisson_encode_batch(self, images):
        """Encode a batch of images: (B, Cin*H*W) -> (B, n_steps, C, H, W)."""
        H, W, Cin = self.input_h, self.input_w, self.input_channels
        B = images.shape[0]
        imgs = images.view(B, Cin, H, W).to(self.device)
        imgs = self._apply_dog(imgs)  # (B, C, H, W) — no-op if preprocess=none
        C = imgs.shape[1]
        imgs = imgs.view(B, 1, C, H, W)
        rates = (imgs * self.intensity / 1000.0).clamp_(0, 1)
        rates = rates.expand(B, self.n_steps, C, H, W)
        spike_trains = torch.bernoulli(rates)
        return spike_trains

    @torch.no_grad()
    def extract_features(self, image, train_stdp=False):
        """Single-image feature extraction (used during STDP training)."""
        spike_trains = self._poisson_encode(image)
        _, pooled1, l1_out = self.layer1.forward(spike_trains, train=train_stdp)
        _, pooled2, _ = self.layer2.forward(l1_out, train=train_stdp)
        return torch.cat([pooled1, pooled2])

    @torch.no_grad()
    def extract_features_batch(self, images):
        """Batched feature extraction for inference (no STDP). Much faster.

        images: (B, 784) tensor.
        Returns: (B, n_features) tensor on device.
        """
        spike_trains = self._poisson_encode_batch(images)  # (B, T, 1, 28, 28)
        pooled1, l1_out = self.layer1.forward_batch(spike_trains, return_spikes=True)
        # L2: skip storing output_spikes (saves ~1.3 GiB), only need pooled
        pooled2, _ = self.layer2.forward_batch(l1_out, return_spikes=False)
        del l1_out  # free L1 output immediately
        return torch.cat([pooled1, pooled2], dim=1)

    def save_checkpoint(self, path, extra=None):
        ckpt = {
            "l1_filters": self.layer1.filters.cpu(),
            "l1_theta": self.layer1.theta.cpu(),
            "l2_filters": self.layer2.filters.cpu(),
            "l2_theta": self.layer2.theta.cpu(),
            "readout_state": self.readout.state_dict(),
            "l1_ks": self.layer1.ks,
            "l2_ks": self.layer2.ks,
            "l1_n_filters": self.layer1.n_filters,
            "l2_n_filters": self.layer2.n_filters,
            "config": self.cfg,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.layer1.filters = ckpt["l1_filters"].to(self.device)
        self.layer1.theta = ckpt["l1_theta"].to(self.device)
        self.layer2.filters = ckpt["l2_filters"].to(self.device)
        self.layer2.theta = ckpt["l2_theta"].to(self.device)
        self.readout.load_state_dict(ckpt["readout_state"])
        print(f"Loaded checkpoint from {path}")


# =============================================================================
# MAIN
# =============================================================================

def _find_latest_sample_ckpt(save_dir, layer_name, epoch):
    """Find the latest `ckpt_{layer_name}_ep{epoch}_s{sample}.pt` for mid-epoch resume.

    Returns (path, sample_index) or None if no sample checkpoint exists.
    The sample_index is the number of images already processed, so the inner
    loop should resume at `i = sample_index`.
    """
    pattern = os.path.join(save_dir, f'ckpt_{layer_name}_ep{epoch}_s*.pt')
    files = glob.glob(pattern)
    best_s = -1
    best_f = None
    for f in files:
        m = re.search(r'_s(\d+)\.pt$', f)
        if m:
            s = int(m.group(1))
            if s > best_s:
                best_s = s
                best_f = f
    if best_f is None:
        return None
    return best_f, best_s


def load_mnist(data_dir="./data", dataset="mnist"):
    """Load MNIST, Fashion-MNIST, or CIFAR-10. Name kept for backwards compat.

    For CIFAR-10 we convert to grayscale (luminance) and downsample to 28×28
    so the existing conv pipeline's hardcoded (1, 1, 28, 28) reshape works
    without modification. This is the Stage-3 smoke-gate-quality adapter;
    a proper RGB 32×32 extension can be added later if numbers warrant.
    """
    name = (dataset or "mnist").lower()
    if name == "mnist":
        cls = datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    elif name in ("fmnist", "fashion_mnist", "fashionmnist"):
        cls = datasets.FashionMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    elif name in ("cifar10", "cifar"):
        cls = datasets.CIFAR10
        # CIFAR-10 is 32x32x3 RGB. The encoder is parameterised by
        # data.input_h, data.input_w, data.input_channels in the config,
        # so for CIFAR we keep all 3 RGB channels and the native 32x32
        # spatial dims. Flatten to 3*32*32 = 3072-dim vector (the encoder
        # reshapes back to (B, 3, 32, 32) before Poisson-encoding).
        # NO DoG, NO downsampling — apples-to-apples with the CIFAR FC runs.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    train_set = cls(data_dir, train=True, download=True, transform=transform)
    test_set  = cls(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def main():
    parser = argparse.ArgumentParser(description="2-Layer Conv Voltage STDP")
    parser.add_argument('--config', default='conv/configs/conv_voltage_config.yaml',
                        help='Path to YAML config')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: tiny data, 2 readout epochs, no wandb, '
                             'no checkpoints. Runs the FULL pipeline end-to-end '
                             'in ~2-5 min to validate a new config before '
                             'committing to a multi-hour run.')
    parser.add_argument('--smoke-n-train', type=int, default=200,
                        help='Samples for Phase A/B under --smoke (default: 200).')
    parser.add_argument('--smoke-n-test', type=int, default=100,
                        help='Test samples under --smoke (default: 100).')
    parser.add_argument('--smoke-readout-epochs', type=int, default=2,
                        help='Readout epochs under --smoke (default: 2).')
    parser.add_argument('--device', default=None,
                        help='Override experiment.device from config (e.g. cuda:0, cpu).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override experiment.seed from config (for multi-seed runs).')
    parser.add_argument('--name-suffix', type=str, default='', dest='name_suffix',
                        help='Suffix appended to experiment name for multi-seed runs '
                             '(e.g. _paper_seed1).')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar10'],
                        help='Dataset (default: mnist). CIFAR-10 is converted '
                             'to grayscale and downsampled to 28x28 so the '
                             'existing conv pipeline works without changes.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    exp_cfg = cfg['experiment']
    train_cfg = cfg['training']
    data_cfg = cfg['data']

    # --smoke: override to tiny values, rename experiment so checkpoints
    # land in a separate dir, disable wandb. Full pipeline otherwise.
    if args.device:
        exp_cfg['device'] = args.device
    if args.seed is not None:
        exp_cfg['seed'] = args.seed
    if args.name_suffix:
        exp_cfg['name'] = exp_cfg['name'] + args.name_suffix
    if args.smoke:
        exp_cfg['name'] = 'SMOKE_' + exp_cfg['name']
        exp_cfg['use_wandb'] = False
        train_cfg['n_train'] = args.smoke_n_train
        train_cfg['n_test'] = args.smoke_n_test
        train_cfg['readout_epochs'] = args.smoke_readout_epochs
        train_cfg['epochs_stdp'] = [1, 1]
        # Bump checkpoint_interval above n_train so no sample ckpts are written.
        train_cfg['checkpoint_interval'] = max(args.smoke_n_train * 10, 100000)
        train_cfg['log_interval'] = max(args.smoke_n_train // 4, 50)
        print("=" * 60)
        print("SMOKE TEST MODE")
        print(f"  n_train={train_cfg['n_train']}  n_test={train_cfg['n_test']}")
        print(f"  readout_epochs={train_cfg['readout_epochs']}  wandb=off")
        print(f"  exp_name={exp_cfg['name']}")
        print("=" * 60)

    # Set global seeds for reproducibility
    _global_seed = exp_cfg.get('seed', 42)
    torch.manual_seed(_global_seed)
    np.random.seed(_global_seed)
    print(f"Global seed: {_global_seed}")

    save_dir = os.path.join(exp_cfg['save_dir'], exp_cfg['name'])
    os.makedirs(save_dir, exist_ok=True)

    log_file, log_path = setup_logging(save_dir, exp_cfg['name'])

    requested_wandb = exp_cfg.get('use_wandb', False)
    use_wandb = False
    print(f"Loaded Experiment: {exp_cfg['name']}")
    print(f"Config: {args.config}")

    if requested_wandb and not HAS_WANDB:
        print("W&B requested, but wandb is not installed; continuing with file/console logs.")
    elif requested_wandb:
        try:
            print("W&B requested; using existing wandb login or WANDB_API_KEY if available.")
            wandb.init(project=exp_cfg.get('project', 'conv-voltage-stdp'),
                       name=exp_cfg['name'], config=cfg)
            use_wandb = True
        except Exception as exc:
            print(f"W&B initialization failed ({exc}); continuing with file/console logs.")

    device_str = exp_cfg.get('device', 'cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    ckpt_path = os.path.join(save_dir, 'conv_voltage_stdp_ckpt.pt')

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    train_set, test_set = load_mnist(dataset=args.dataset)
    print(f"  Dataset: {args.dataset}", flush=True)
    n_train = min(train_cfg.get('n_train', 60000), len(train_set))
    n_test = min(train_cfg.get('n_test', 10000), len(test_set))

    # ----------------------------------------------------------------
    # Build network from config
    # ----------------------------------------------------------------
    net = TwoLayerConvVoltageSTDP(cfg, device)

    l1_cfg = cfg['model']['layer1']
    l2_cfg = cfg['model']['layer2']
    voltage_cfg = cfg.get('voltage', {})
    readout_cfg = cfg.get('readout', {})
    resolved_metadata = {
        "voltage_rule": voltage_cfg.get('rule', 'approximate'),
        "ltd_mode": voltage_cfg.get('ltd_mode', 'n/a'),
        "use_weight_norm": voltage_cfg.get('use_weight_norm', True),
        "sim_time": data_cfg['sim_time'],
        "intensity": data_cfg.get('intensity', 128.0),
        "layer1_theta_plus": l1_cfg.get('theta_plus', 0.05),
        "layer1_tc_theta_decay": l1_cfg.get('tc_theta_decay', 1e7),
        "layer1_inh_weight": l1_cfg.get('inh_weight', 17.5),
        "layer1_norm": l1_cfg.get('norm', None),
        "layer2_theta_plus": l2_cfg.get('theta_plus', 0.05),
        "layer2_tc_theta_decay": l2_cfg.get('tc_theta_decay', 1e7),
        "layer2_inh_weight": l2_cfg.get('inh_weight', 17.5),
        "layer2_norm": l2_cfg.get('norm', None),
    }

    if use_wandb:
        wandb.config.update(
            {f"resolved/{k}": v for k, v in resolved_metadata.items()},
            allow_val_change=True,
        )
        for k, v in resolved_metadata.items():
            wandb.run.summary[f"config/{k}"] = v

    print(f"\n{'='*60}")
    print(f"  Conv Voltage STDP (Clopath et al. 2010)")
    print(f"{'='*60}")
    print(f"  Layer 1: {l1_cfg['n_filters']} x {l1_cfg['kernel_size']}x{l1_cfg['kernel_size']}"
          f"  ->  {net.layer1.H_out}x{net.layer1.W_out}")
    print(f"  Layer 2: {l2_cfg['n_filters']} x {l2_cfg['kernel_size']}x{l2_cfg['kernel_size']}"
          f"  ->  {net.layer2.H_out}x{net.layer2.W_out}")
    print(f"  Features: L1={net.n_feat_l1} + L2={net.n_feat_l2} = {net.n_features}")
    print(f"  Voltage: A+={voltage_cfg.get('A_plus', 0.006)}, "
          f"A-={voltage_cfg.get('A_minus', 0.003)}")
    print(f"  Traces: tau_pre={voltage_cfg.get('tau_pre', 16.8)}, "
          f"tau+={voltage_cfg.get('tau_plus', 3.0)}, "
          f"tau-={voltage_cfg.get('tau_minus', 8.0)}")
    print(f"  Rule: {resolved_metadata['voltage_rule']} | "
          f"LTD: {resolved_metadata['ltd_mode']} | "
          f"weight_norm={resolved_metadata['use_weight_norm']}")
    print(f"  Sim time: {data_cfg['sim_time']} steps")
    print(f"  STDP epochs: {train_cfg['epochs_stdp']}")
    print(f"  Readout: {net.n_features} -> "
          f"{readout_cfg.get('hidden_size', 256)} -> 10  "
          f"({train_cfg.get('readout_epochs', 30)} epochs)")
    print(f"{'='*60}")

    log_interval = train_cfg.get('log_interval', 500)
    ckpt_interval = train_cfg.get('checkpoint_interval', 10000)
    epochs_stdp = train_cfg['epochs_stdp']

    total_start = time.time()

    # ================================================================
    # PHASE A: UNSUPERVISED VOLTAGE STDP TRAINING
    # ================================================================
    if not os.path.exists(ckpt_path):
        print(f"\n{'='*60}")
        print(f"PHASE A: Voltage STDP Training")
        print(f"{'='*60}")

        for layer_idx in range(2):
            layer_name = f"L{layer_idx+1}"
            n_epochs = epochs_stdp[layer_idx]
            print(f"\n  Training {layer_name} ({n_epochs} epochs)...")

            for epoch in range(n_epochs):
                # --- Resume: skip epochs that already have checkpoints ---
                epoch_ckpt = os.path.join(
                    save_dir, f'ckpt_{layer_name}_ep{epoch+1}.pt')
                if os.path.exists(epoch_ckpt):
                    print(f"    [Resume] Found {epoch_ckpt}, loading & skipping "
                          f"{layer_name} epoch {epoch+1}")
                    net.load_checkpoint(epoch_ckpt)
                    continue

                # Mid-epoch resume: if a sample checkpoint exists for this
                # (layer, epoch), load it and restart the inner loop at the
                # sample index that was in progress.
                start_i = 0
                sample_resume = _find_latest_sample_ckpt(
                    save_dir, layer_name, epoch + 1)
                if sample_resume is not None:
                    resume_path, start_i = sample_resume
                    print(f"    [Resume] Found sample checkpoint {resume_path}; "
                          f"resuming {layer_name} epoch {epoch+1} at sample {start_i}")
                    net.load_checkpoint(resume_path)

                gen = torch.Generator().manual_seed(
                    exp_cfg.get('seed', 42) + epoch + layer_idx * 100)
                indices = torch.randperm(len(train_set), generator=gen)[:n_train]

                active_layer = net.layer1 if layer_idx == 0 else net.layer2
                active_layer._reset_diagnostics(scope="all")

                t0 = time.time()
                for i in range(start_i, n_train):
                    image, label = train_set[indices[i].item()]
                    spike_trains = net._poisson_encode(image)

                    if layer_idx == 0:
                        net.layer1.forward(spike_trains, train=True)
                    else:
                        _, _, l1_out = net.layer1.forward(spike_trains, train=False)
                        net.layer2.forward(l1_out, train=True)

                    if (i + 1) % log_interval == 0:
                        speed = (i + 1 - start_i) / max(time.time() - t0, 1e-6)
                        d = active_layer.get_diagnostics()
                        warn_labels = []
                        if d["zero_ltd_warning"]:
                            warn_labels.append("zero_ltd")
                        if d["dead_firing_warning"]:
                            warn_labels.append("dead_firing")
                        if d["theta_saturated_warning"]:
                            warn_labels.append("theta_saturated")
                        if d["weight_collapse_warning"]:
                            warn_labels.append("weight_collapse")

                        # Console: compact multi-line diagnostic
                        print(f"    {layer_name} Ep{epoch+1} [{i+1:5d}/{n_train}] "
                              f"Speed: {speed:.1f} img/s")
                        print(f"      spikes: {d['avg_spikes_per_image']:.1f}/img | "
                              f"input_spk: {d['avg_input_spikes_per_image']:.1f}/img | "
                              f"fire_rate: {d['firing_rate']:.4f}")
                        print(f"      voltage: avg={d['avg_voltage']:.3f} "
                              f"peak={d['peak_voltage']:.3f} | "
                              f"u+={d['avg_u_plus']:.4f} u-={d['avg_u_minus']:.4f} | "
                              f"pre_trace={d['avg_pre_trace']:.4f}")
                        print(f"      raw ltp/ltd: {d['raw_ltp_fraction']:.3f}/"
                              f"{d['raw_ltd_fraction']:.3f} | "
                              f"raw+={d['avg_raw_ltp_mag']:.6f} "
                              f"raw-={d['avg_raw_ltd_mag']:.6f}")
                        print(f"      net dw: +{d['avg_net_dw_pos']:.6f} "
                              f"-{d['avg_net_dw_neg']:.6f} | "
                              f"net+/-={d['net_dw_pos_fraction']:.3f}/"
                              f"{d['net_dw_neg_fraction']:.3f}")
                        print(f"      weights: mean={d['weight_mean']:.4f} "
                              f"std={d['weight_std']:.4f} "
                              f"[{d['weight_min']:.4f}, {d['weight_max']:.4f}] | "
                              f"near0={d['weight_near_0_pct']:.1f}% "
                              f"near1={d['weight_near_1_pct']:.1f}%")
                        print(f"      convergence: {d['convergence']:.6f} | "
                              f"theta: mean={d['theta_mean']:.4f} "
                              f"max={d['theta_max']:.4f}")
                        if warn_labels:
                            print(f"      warnings: {', '.join(warn_labels)}")

                        if use_wandb:
                            step = i + 1 + epoch * n_train
                            wandb.log({
                                f"{layer_name}/speed": speed,
                                # Activity
                                f"{layer_name}/avg_spikes_per_image": d["avg_spikes_per_image"],
                                f"{layer_name}/avg_input_spikes": d["avg_input_spikes_per_image"],
                                f"{layer_name}/firing_rate": d["firing_rate"],
                                # Voltage traces
                                f"{layer_name}/avg_voltage": d["avg_voltage"],
                                f"{layer_name}/peak_voltage": d["peak_voltage"],
                                f"{layer_name}/avg_u_plus": d["avg_u_plus"],
                                f"{layer_name}/avg_u_minus": d["avg_u_minus"],
                                f"{layer_name}/avg_pre_trace": d["avg_pre_trace"],
                                # Raw plasticity balance before LTP/LTD cancellation
                                f"{layer_name}/raw_ltp_fraction": d["raw_ltp_fraction"],
                                f"{layer_name}/raw_ltd_fraction": d["raw_ltd_fraction"],
                                f"{layer_name}/raw_ltp_magnitude": d["raw_ltp_mag"],
                                f"{layer_name}/raw_ltd_magnitude": d["raw_ltd_mag"],
                                f"{layer_name}/avg_raw_ltp_magnitude": d["avg_raw_ltp_mag"],
                                f"{layer_name}/avg_raw_ltd_magnitude": d["avg_raw_ltd_mag"],
                                # Backward-compatible names map to raw plasticity.
                                f"{layer_name}/ltp_fraction": d["ltp_fraction"],
                                f"{layer_name}/ltd_fraction": d["ltd_fraction"],
                                f"{layer_name}/ltp_magnitude": d["ltp_mag"],
                                f"{layer_name}/ltd_magnitude": d["ltd_mag"],
                                # Net weight movement after LTP/LTD cancellation
                                f"{layer_name}/net_dw_pos_fraction": d["net_dw_pos_fraction"],
                                f"{layer_name}/net_dw_neg_fraction": d["net_dw_neg_fraction"],
                                f"{layer_name}/net_dw_pos_magnitude": d["net_dw_pos_mag"],
                                f"{layer_name}/net_dw_neg_magnitude": d["net_dw_neg_mag"],
                                f"{layer_name}/avg_net_dw_pos": d["avg_net_dw_pos"],
                                f"{layer_name}/avg_net_dw_neg": d["avg_net_dw_neg"],
                                # Weights
                                f"{layer_name}/weight_mean": d["weight_mean"],
                                f"{layer_name}/weight_std": d["weight_std"],
                                f"{layer_name}/weight_min": d["weight_min"],
                                f"{layer_name}/weight_max": d["weight_max"],
                                f"{layer_name}/weight_near_0_pct": d["weight_near_0_pct"],
                                f"{layer_name}/weight_near_1_pct": d["weight_near_1_pct"],
                                f"{layer_name}/convergence": d["convergence"],
                                # Adaptive threshold
                                f"{layer_name}/theta_mean": d["theta_mean"],
                                f"{layer_name}/theta_max": d["theta_max"],
                                # Numeric warning fields for dashboard filters
                                f"{layer_name}/warnings/zero_ltd": d["zero_ltd_warning"],
                                f"{layer_name}/warnings/dead_firing": d["dead_firing_warning"],
                                f"{layer_name}/warnings/theta_saturated": d["theta_saturated_warning"],
                                f"{layer_name}/warnings/weight_collapse": d["weight_collapse_warning"],
                            }, step=step)

                    # Periodic checkpoint during STDP training
                    if (i + 1) % ckpt_interval == 0:
                        interval_path = os.path.join(
                            save_dir, f'ckpt_{layer_name}_ep{epoch+1}_s{i+1}.pt')
                        net.save_checkpoint(interval_path, extra={
                            'phase': 'stdp',
                            'layer': layer_idx,
                            'epoch': epoch + 1,
                            'sample': i + 1,
                        })
                        print(f"    [Checkpoint] Saved {interval_path}")

                elapsed = time.time() - t0
                print(f"    {layer_name} epoch {epoch+1} done in "
                      f"{elapsed:.0f}s ({elapsed/60:.1f} min)")

                # Epoch-end diagnostics summary. This uses a separate epoch
                # accumulator, so interval logs do not erase the summary.
                d = active_layer.get_diagnostics(scope="epoch")
                print(f"    --- {layer_name} Epoch {epoch+1} Summary ---")
                print(f"      firing_rate={d['firing_rate']:.4f} | "
                      f"spk/img={d['avg_spikes_per_image']:.1f} | "
                      f"input_spk/img={d['avg_input_spikes_per_image']:.1f}")
                print(f"      raw_ltp/ltd={d['raw_ltp_fraction']:.3f}/"
                      f"{d['raw_ltd_fraction']:.3f} | "
                      f"raw+={d['avg_raw_ltp_mag']:.6f} "
                      f"raw-={d['avg_raw_ltd_mag']:.6f}")
                print(f"      net_dw=+{d['avg_net_dw_pos']:.6f}/"
                      f"-{d['avg_net_dw_neg']:.6f} | "
                      f"convergence={d['convergence']:.6f}")
                print(f"      weights: {d['weight_mean']:.4f} +/- {d['weight_std']:.4f} "
                      f"[{d['weight_min']:.4f}, {d['weight_max']:.4f}] | "
                      f"near0={d['weight_near_0_pct']:.1f}% near1={d['weight_near_1_pct']:.1f}%")
                print(f"      theta: mean={d['theta_mean']:.4f} max={d['theta_max']:.4f}")
                epoch_warn_labels = []
                if d["zero_ltd_warning"]:
                    epoch_warn_labels.append("zero_ltd")
                if d["dead_firing_warning"]:
                    epoch_warn_labels.append("dead_firing")
                if d["theta_saturated_warning"]:
                    epoch_warn_labels.append("theta_saturated")
                if d["weight_collapse_warning"]:
                    epoch_warn_labels.append("weight_collapse")
                if epoch_warn_labels:
                    print(f"      warnings: {', '.join(epoch_warn_labels)}")

                if use_wandb:
                    wandb.log({
                        f"{layer_name}/epoch": epoch + 1,
                        f"{layer_name}/epoch_convergence": d["convergence"],
                        f"{layer_name}/epoch_firing_rate": d["firing_rate"],
                        f"{layer_name}/epoch_weight_mean": d["weight_mean"],
                        f"{layer_name}/epoch_weight_std": d["weight_std"],
                        f"{layer_name}/epoch_raw_ltp_fraction": d["raw_ltp_fraction"],
                        f"{layer_name}/epoch_raw_ltd_fraction": d["raw_ltd_fraction"],
                        f"{layer_name}/epoch_raw_ltp_magnitude": d["raw_ltp_mag"],
                        f"{layer_name}/epoch_raw_ltd_magnitude": d["raw_ltd_mag"],
                        f"{layer_name}/epoch_net_dw_pos": d["avg_net_dw_pos"],
                        f"{layer_name}/epoch_net_dw_neg": d["avg_net_dw_neg"],
                        f"{layer_name}/epoch_theta_mean": d["theta_mean"],
                        f"{layer_name}/epoch_theta_max": d["theta_max"],
                        f"{layer_name}/epoch_warning_zero_ltd": d["zero_ltd_warning"],
                        f"{layer_name}/epoch_warning_dead_firing": d["dead_firing_warning"],
                        f"{layer_name}/epoch_warning_theta_saturated": d["theta_saturated_warning"],
                        f"{layer_name}/epoch_warning_weight_collapse": d["weight_collapse_warning"],
                    })

                # Save after each epoch
                epoch_path = os.path.join(
                    save_dir, f'ckpt_{layer_name}_ep{epoch+1}.pt')
                net.save_checkpoint(epoch_path, extra={
                    'phase': 'stdp',
                    'layer': layer_idx,
                    'epoch': epoch + 1,
                })
                print(f"    [Checkpoint] Epoch done -> {epoch_path}")

            # Save after each layer finishes
            layer_path = os.path.join(save_dir, f'ckpt_after_{layer_name}.pt')
            net.save_checkpoint(layer_path, extra={
                'phase': 'stdp',
                'layer': layer_idx,
                'status': 'layer_complete',
            })
            print(f"  [Checkpoint] {layer_name} complete -> {layer_path}")

        net.save_checkpoint(ckpt_path)
        print(f"\n  STDP weights saved to {ckpt_path}")
    else:
        print(f"\n=== Loading Pre-trained Voltage STDP ===")
        net.load_checkpoint(ckpt_path)

    # ================================================================
    # PHASE B: FEATURE EXTRACTION
    # ================================================================
    print(f"\n{'='*60}")
    print(f"PHASE B: Feature Extraction")
    print(f"{'='*60}")

    feat_batch_size = cfg['training'].get('feat_batch_size', 64)

    def extract_all(dataset, n_samples, label=""):
        print(f"  Extracting {label} features ({n_samples} samples, "
              f"batch={feat_batch_size})...")
        all_features = []
        all_labels = []
        t0 = time.time()

        for start in range(0, n_samples, feat_batch_size):
            end = min(start + feat_batch_size, n_samples)
            batch_imgs = []
            batch_lbls = []
            for i in range(start, end):
                image, lbl = dataset[i]
                batch_imgs.append(image)
                batch_lbls.append(lbl)

            imgs = torch.stack(batch_imgs)  # (B, 784)
            feats = net.extract_features_batch(imgs)  # (B, n_features)
            all_features.append(feats)
            all_labels.extend(batch_lbls)

            if (end) % 5000 < feat_batch_size or end == n_samples:
                speed = end / (time.time() - t0)
                eta = (n_samples - end) / max(speed, 1) / 60
                print(f"    [{end}/{n_samples}] {speed:.1f} img/s, ETA: {eta:.1f} min")

        features = torch.cat(all_features, dim=0).to(device)
        labels = torch.tensor(all_labels, dtype=torch.long, device=device)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.0f}s. Shape: {features.shape}")
        return features, labels

    X_train, y_train = extract_all(train_set, n_train, "train")
    X_test, y_test = extract_all(test_set, n_test, "test")

    # --- Optional per-feature standardization (no Phase A/B retraining) ---
    readout_cfg = cfg.get('readout', {})
    if readout_cfg.get('normalize_features', False):
        feat_mean = X_train.mean(dim=0, keepdim=True)
        feat_std  = X_train.std(dim=0, keepdim=True).clamp(min=1e-6)
        X_train = (X_train - feat_mean) / feat_std
        X_test  = (X_test  - feat_mean) / feat_std
        print(f"  [FeatureNorm] standardized {X_train.shape[1]} features "
              f"(mean→0, std→1 on train; test reuses train stats)")

    # ================================================================
    # PHASE C: SURROGATE GRADIENT READOUT TRAINING
    # ================================================================
    print(f"\n{'='*60}")
    print(f"PHASE C: Surrogate Gradient Readout Training")
    print(f"{'='*60}")

    readout_epochs = train_cfg.get('readout_epochs', 30)
    batch_size = train_cfg.get('readout_batch_size', 128)

    # --- Build LR scheduler now that readout_epochs is known ---
    lr_schedule = readout_cfg.get('lr_schedule', None)
    if lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            net.optimizer, T_max=readout_epochs)
    elif lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            net.optimizer, mode='max', factor=0.5, patience=5)
    else:
        scheduler = None

    print(f"  Features: {X_train.shape[1]}, "
          f"Hidden: {net.readout.fc1.out_features}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Readout timesteps: {net.readout.n_steps}")
    print(f"  Optimizer: {type(net.optimizer).__name__}, "
          f"weight_decay={readout_cfg.get('weight_decay', 0.0)}, "
          f"label_smoothing={readout_cfg.get('label_smoothing', 0.0)}, "
          f"schedule={lr_schedule}")
    print(f"  Epochs: {readout_epochs}, Batch: {batch_size}, "
          f"LR: {train_cfg.get('readout_lr', 1e-3)}")

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)

    best_acc = 0.0
    best_path = os.path.join(save_dir, 'best_readout.pth')

    for epoch in range(readout_epochs):
        net.readout.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            net.optimizer.zero_grad()
            out = net.readout(xb)
            loss = net.criterion(out, yb)
            loss.backward()
            net.optimizer.step()

            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(xb)

        train_acc = correct / total * 100
        train_loss = total_loss / total

        # Test
        net.readout.eval()
        with torch.no_grad():
            test_out = net.readout(X_test)
            test_pred = test_out.argmax(1)
            test_acc = (test_pred == y_test).float().mean().item() * 100

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.readout.state_dict(), best_path)
            print(f"  -> New best model saved (acc={best_acc:.1f}%)")

        # --- LR scheduler step (after optimizer updates this epoch) ---
        if scheduler is not None:
            if isinstance(scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_acc)
            else:
                scheduler.step()
        cur_lr = net.optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch+1:2d}/{readout_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}% | "
              f"Best: {best_acc:.1f}% | "
              f"LR: {cur_lr:.2e}")

        # Per-class accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == readout_epochs:
            print(f"    Per-class accuracy:")
            per_class = {}
            for c in range(10):
                mask = y_test == c
                if mask.sum() > 0:
                    c_acc = (test_pred[mask] == c).float().mean().item() * 100
                    per_class[c] = c_acc
                    print(f"      digit {c}: {c_acc:5.1f}% ({int(mask.sum())} samples)")
                else:
                    per_class[c] = 0.0
            if use_wandb:
                for c, acc in per_class.items():
                    wandb.log({f"readout/class_{c}_acc": acc})

        # Periodic readout checkpoint
        if (epoch + 1) % 10 == 0:
            readout_ckpt = os.path.join(save_dir, f'readout_ep{epoch+1}.pth')
            torch.save(net.readout.state_dict(), readout_ckpt)
            print(f"  [Checkpoint] Readout saved -> {readout_ckpt}")

        if use_wandb:
            wandb.log({
                "readout/epoch": epoch + 1,
                "readout/train_loss": train_loss,
                "readout/train_acc": train_acc,
                "readout/test_acc": test_acc,
                "readout/best_acc": best_acc,
                "readout/lr": cur_lr,
            })

    # ================================================================
    # Summary
    # ================================================================
    total_time = time.time() - total_start
    net.save_checkpoint(ckpt_path, extra={"best_acc": best_acc})

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Experiment: {exp_cfg['name']}")
    print(f"  STDP Rule: Voltage-modulated (Clopath et al. 2010)")
    print(f"  Architecture: Conv {l1_cfg['kernel_size']}x{l1_cfg['kernel_size']}"
          f"x{l1_cfg['n_filters']} -> Conv {l2_cfg['kernel_size']}x{l2_cfg['kernel_size']}"
          f"x{l2_cfg['n_filters']} -> SNN Readout")
    print(f"  Features: {net.n_features} (L1: {net.n_feat_l1}, L2: {net.n_feat_l2})")
    print(f"  Best Test Accuracy: {best_acc:.1f}%")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f} hrs)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.finish()

    # --smoke: run assertions on pipeline health and report PASS / FAIL.
    if args.smoke:
        print(f"\n{'=' * 60}")
        print("SMOKE TEST ASSERTIONS")
        print('=' * 60)
        failures = []
        # 1. Best acc must be finite and above pure-random (10% for 10-class).
        #    With very few samples/epochs we only require > 12% to allow
        #    stochastic noise, not real convergence.
        if not math.isfinite(best_acc):
            failures.append(f"best_acc not finite: {best_acc}")
        elif best_acc < 12.0:
            failures.append(f"best_acc={best_acc:.1f}% ≤ 12% (random-level)")
        # 2. Feature vector size must match architecture claim.
        if net.n_features != net.n_feat_l1 + net.n_feat_l2:
            failures.append(f"n_features mismatch: {net.n_features} vs "
                            f"{net.n_feat_l1}+{net.n_feat_l2}")
        # 3. L1 weights must be finite + within the documented [0, w_max] range.
        try:
            import torch as _torch
            w1 = net.layer1.filters
            w2 = net.layer2.filters
            if not _torch.isfinite(w1).all():
                failures.append("L1 weights contain NaN/Inf")
            if not _torch.isfinite(w2).all():
                failures.append("L2 weights contain NaN/Inf")
            if w1.min().item() < -1e-6 or w1.max().item() > 1.0 + 1e-6:
                failures.append(
                    f"L1 weights out of [0,1]: [{w1.min():.4f}, {w1.max():.4f}]")
            if w2.min().item() < -1e-6 or w2.max().item() > 1.0 + 1e-6:
                failures.append(
                    f"L2 weights out of [0,1]: [{w2.min():.4f}, {w2.max():.4f}]")
            # 4. Weight std must be non-trivial (STDP did SOMETHING).
            if w1.std().item() < 1e-4:
                failures.append(f"L1 weight std too small: {w1.std().item():.6f} "
                                "(STDP likely not updating)")
        except Exception as exc:
            failures.append(f"weight-health check failed: {exc}")

        if failures:
            print("  [FAIL] " + str(len(failures)) + " issue(s):")
            for f in failures:
                print(f"    - {f}")
            print(f"\n{'=' * 60}\nSMOKE TEST: FAIL\n{'=' * 60}")
            sys.stdout = sys.stdout.original
            sys.stderr = sys.stderr.original
            log_file.close()
            sys.exit(1)
        else:
            print(f"  best_acc          = {best_acc:.1f}%   (>12% threshold)")
            print(f"  n_features        = {net.n_features}")
            print(f"  L1 weight range   = [{net.layer1.filters.min():.4f}, "
                  f"{net.layer1.filters.max():.4f}]  std={net.layer1.filters.std():.4f}")
            print(f"  L2 weight range   = [{net.layer2.filters.min():.4f}, "
                  f"{net.layer2.filters.max():.4f}]  std={net.layer2.filters.std():.4f}")
            print(f"  wall time         = {total_time:.0f}s")
            print(f"\n{'=' * 60}\nSMOKE TEST: PASS ✓\n{'=' * 60}")

    print(f"\nFull log saved to: {log_path}")
    sys.stdout = sys.stdout.original
    sys.stderr = sys.stderr.original
    log_file.close()


if __name__ == "__main__":
    main()
