"""
2-Layer Convolutional STDP with Surrogate Gradient Readout
==========================================================

Architecture:
  Layer 1: Conv 5x5 x 32 filters (Triplet STDP, unsupervised)
  Layer 2: Conv 5x5/7x7 x 64 filters (Triplet STDP, unsupervised)
  Readout: Fully connected SNN trained with surrogate gradient (supervised)

Phase 1: Train Layer 1 with STDP (freeze Layer 2, readout)
Phase 2: Train Layer 2 with STDP using Layer 1 features (freeze readout)
Phase 3: Train readout SNN with surrogate gradients (freeze STDP layers)

Usage:
    CUDA_VISIBLE_DEVICES=1 python conv/conv2layer_stdp_surrogate.py \
        --ks2 5 --epochs 3 --log_file conv2l_ks5.log
"""

import argparse
import logging
import os
import time
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms


def setup_logging(log_file: str = None):
    logger = logging.getLogger("conv2l_stdp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


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


# =============================================================================
# READOUT SNN (trained with surrogate gradient)
# =============================================================================

class ReadoutSNN(nn.Module):
    """Fully connected LIF readout SNN trained with surrogate gradients.

    Input: spike count features from conv STDP layers
    Output: class logits (membrane potential of output neurons accumulated over time)
    """

    def __init__(self, n_input, n_hidden, n_output, n_steps=50,
                 tau=10.0, dt=1.0):
        super().__init__()
        self.n_steps = n_steps
        self.decay = math.exp(-dt / tau)
        self.n_output = n_output

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

        # Initialize with small weights
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: (batch, n_input) — spike count features (converted to current input)
        Returns: (batch, n_output) — accumulated output membrane potentials
        """
        batch = x.shape[0]
        device = x.device

        # Normalize input to [0, 1] range for Poisson encoding
        x_norm = x / (x.max(dim=1, keepdim=True).values.clamp(min=1.0))

        # Hidden layer state
        v1 = torch.zeros(batch, self.fc1.out_features, device=device)
        # Output layer state
        v2 = torch.zeros(batch, self.n_output, device=device)
        # Accumulate output spikes
        out_spikes = torch.zeros(batch, self.n_output, device=device)

        for t in range(self.n_steps):
            # Poisson encode input features at each timestep
            inp_spikes = (torch.rand_like(x_norm) < x_norm).float()

            # Hidden layer
            v1 = self.decay * v1 + self.fc1(inp_spikes)
            s1 = spike_fn(v1 - 1.0)  # threshold = 1.0
            v1 = v1 * (1.0 - s1)  # reset

            # Output layer
            v2 = self.decay * v2 + self.fc2(s1)
            s2 = spike_fn(v2 - 1.0)
            v2 = v2 * (1.0 - s2)
            out_spikes = out_spikes + s2

        return out_spikes


# =============================================================================
# CONV STDP LAYER (unsupervised, one layer)
# =============================================================================

class ConvSTDPLayer:
    """Single convolutional STDP layer with triplet learning rule."""

    def __init__(self, n_input_channels, n_filters, kernel_size, input_h, input_w,
                 dt=1.0, sim_time=350.0, intensity=128.0, norm=10.0,
                 tau_plus=16.8, tau_minus=33.7, tau_x=101.0, tau_y=125.0,
                 A2_plus=6.1e-3, A2_minus=1.6e-3, A3_plus=3.5e-3, A3_minus=1.4e-4,
                 v_rest=-65.0, v_reset=-65.0, v_thresh=-52.0, tau_v=100.0,
                 theta_plus=0.05, tc_theta_decay=1e7,
                 inh_weight=17.5, refrac_time=5, device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_input_ch = n_input_channels
        self.n_filters = n_filters
        self.ks = kernel_size
        self.k_sq = kernel_size * kernel_size * n_input_channels  # full patch size
        self.dt = dt
        self.n_steps = int(sim_time / dt)
        self.intensity = intensity
        self.norm = norm
        self.v_rest = v_rest
        self.v_reset = v_reset

        self.H_out = input_h - kernel_size + 1
        self.W_out = input_w - kernel_size + 1
        self.n_pos = self.H_out * self.W_out
        self.pool_h = self.H_out // 2
        self.pool_w = self.W_out // 2

        # Precompute decays
        self.decay_v = math.exp(-dt / tau_v)
        self.decay_theta = math.exp(-dt / tc_theta_decay)
        self.decay_r1 = math.exp(-dt / tau_plus)
        self.decay_r2 = math.exp(-dt / tau_x)
        self.decay_o1 = math.exp(-dt / tau_minus)
        self.decay_o2 = math.exp(-dt / tau_y)

        self.A2p = A2_plus
        self.A2m = A2_minus
        self.A3p = A3_plus
        self.A3m = A3_minus
        self.inh_weight = inh_weight
        self.theta_plus = theta_plus
        self.v_thresh = v_thresh

        self._v_reset_t = torch.tensor(v_reset, device=self.device)
        self._refrac_time_t = torch.tensor(refrac_time, dtype=torch.int32, device=self.device)

        # Filter weights
        self.filters = torch.rand(n_filters, self.k_sq, device=self.device) * 0.3
        self._normalize_filters()

        # Adaptive threshold
        self.theta = torch.zeros(self.n_pos, n_filters, device=self.device)

    def _normalize_filters(self):
        if self.norm is not None:
            row_sums = self.filters.sum(1, keepdim=True).clamp_(min=1e-8)
            self.filters.mul_(self.norm / row_sums)
            self.filters.clamp_(0.0, 1.0)

    @torch.no_grad()
    def forward(self, spike_trains_4d, train=True):
        """
        spike_trains_4d: (n_steps, C, H, W) — input spike trains on device
        Returns: spike_count (n_pos, n_filters), pooled_flat (n_pooled,)
                 and output_spike_trains (n_steps, n_pos, n_filters) for next layer
        """
        dev = self.device
        n_steps = spike_trains_4d.shape[0]
        n_pos = self.n_pos
        n_f = self.n_filters
        k_sq = self.k_sq

        # Unfold to patches: (n_steps, n_pos, k_sq)
        all_patches = F.unfold(spike_trains_4d, self.ks).permute(0, 2, 1).contiguous()

        # Initialize state
        v = torch.full((n_pos, n_f), self.v_rest, device=dev)
        refrac = torch.zeros(n_pos, n_f, dtype=torch.int32, device=dev)
        r1p = torch.zeros(n_pos, k_sq, device=dev)
        r2p = torch.zeros(n_pos, k_sq, device=dev)
        o1 = torch.zeros(n_pos, n_f, device=dev)
        o2 = torch.zeros(n_pos, n_f, device=dev)
        dw_accum = torch.zeros(n_f, k_sq, device=dev)
        spike_count = torch.zeros(n_pos, n_f, device=dev)
        theta = self.theta

        # Store output spikes for next layer
        output_spikes = torch.zeros(n_steps, n_pos, n_f, device=dev)

        for t in range(n_steps):
            pre_patches = all_patches[t]

            # Forward
            exc_input = pre_patches @ self.filters.t()
            not_refrac = refrac == 0
            v = torch.where(not_refrac,
                            v * self.decay_v + (1 - self.decay_v) * self.v_rest + exc_input, v)

            v_th = self.v_thresh + theta
            spikes = (v >= v_th) & not_refrac
            post = spikes.float()
            spike_count = spike_count + post
            output_spikes[t] = post

            v = torch.where(spikes, self._v_reset_t, v)
            refrac = torch.where(spikes, self._refrac_time_t, refrac)
            theta = torch.where(spikes, theta + self.theta_plus, theta) * self.decay_theta

            # Lateral inhibition
            n_sp_local = post.sum(dim=1, keepdim=True)
            v = v - n_sp_local * self.inh_weight
            n_sp_global = post.sum(dim=0, keepdim=True)
            v = v - n_sp_global * (self.inh_weight * 0.1)
            v = torch.where(spikes, self._v_reset_t, v)

            refrac = torch.clamp(refrac - 1, min=0)

            if train:
                # Triplet STDP
                r2p_prev = r2p.clone()
                o2_prev = o2.clone()

                r1p = r1p * self.decay_r1
                r2p = r2p * self.decay_r2
                o1 = o1 * self.decay_o1
                o2 = o2 * self.decay_o2

                ltp_coeff = self.A2p + self.A3p * o2_prev
                dw_accum = dw_accum + (post * ltp_coeff).t() @ r1p

                ltd_term = pre_patches * (self.A2m + self.A3m * r2p_prev)
                dw_accum = dw_accum - o1.t() @ ltd_term

                r1p = r1p + pre_patches
                r2p = r2p + pre_patches
                o1 = o1 + post
                o2 = o2 + post

        self.theta = theta

        if train:
            self.filters = self.filters + dw_accum / n_pos
            self.filters.clamp_(0.0, 1.0)
            self._normalize_filters()

        # Max pool
        sc_map = spike_count.t().reshape(1, n_f, self.H_out, self.W_out)
        pooled = F.max_pool2d(sc_map, 2).squeeze(0)
        pooled_flat = pooled.reshape(-1).cpu()

        # Reshape output spikes for next layer: (n_steps, n_f, H_out, W_out)
        output_spikes_4d = output_spikes.permute(0, 2, 1).reshape(n_steps, n_f, self.H_out, self.W_out)

        return spike_count, pooled_flat, output_spikes_4d


# =============================================================================
# 2-LAYER CONV STDP + SURROGATE READOUT
# =============================================================================

class TwoLayerConvSTDP_SurrogateNet:
    """
    Layer 1: Conv STDP (32 x 5x5)
    Layer 2: Conv STDP (64 x ks2 x ks2)
    Readout: FC SNN with surrogate gradient
    """

    def __init__(self, n_filters1=32, ks1=5, n_filters2=64, ks2=5,
                 sim_time=350.0, intensity=128.0, norm=10.0,
                 readout_hidden=256, readout_steps=50, readout_lr=1e-3,
                 device=None, seed=42,
                 input_channels=1, input_h=28, input_w=28, n_output=10):
        self.n_output = n_output

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.intensity = intensity
        self.sim_time = sim_time
        self.n_steps = int(sim_time)

        # Input spatial dimensions (per-dataset). MNIST/FMNIST default to
        # 1×28×28; CIFAR-10 RGB needs 3×32×32.
        self.input_channels = int(input_channels)
        self.input_h = int(input_h)
        self.input_w = int(input_w)

        # Layer 1: input_h x input_w input → (h-ks1+1) x (w-ks1+1) x n_filters1
        self.layer1 = ConvSTDPLayer(
            n_input_channels=self.input_channels, n_filters=n_filters1, kernel_size=ks1,
            input_h=self.input_h, input_w=self.input_w,
            sim_time=sim_time, intensity=intensity,
            norm=norm, device=self.device,
        )
        l1_h, l1_w = self.layer1.H_out, self.layer1.W_out  # 24, 24 (MNIST) / 28, 28 (CIFAR)

        # Layer 2: l1_h x l1_w x n_filters1 input → (l1_h-ks2+1) x (l1_w-ks2+1) x n_filters2
        self.layer2 = ConvSTDPLayer(
            n_input_channels=n_filters1, n_filters=n_filters2, kernel_size=ks2,
            input_h=l1_h, input_w=l1_w, sim_time=sim_time,
            intensity=1.0,  # L2 input is already binary spikes, no intensity scaling
            norm=norm * 2,  # higher norm for larger patches
            device=self.device,
        )
        l2_h, l2_w = self.layer2.H_out, self.layer2.W_out

        # Feature dimension: L1 pooled + L2 pooled
        self.n_feat_l1 = n_filters1 * (l1_h // 2) * (l1_w // 2)
        self.n_feat_l2 = n_filters2 * (l2_h // 2) * (l2_w // 2)
        self.n_features = self.n_feat_l1 + self.n_feat_l2

        # Readout SNN
        self.readout = ReadoutSNN(
            n_input=self.n_features,
            n_hidden=readout_hidden,
            n_output=n_output,
            n_steps=readout_steps,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.readout.parameters(), lr=readout_lr)
        self.criterion = nn.CrossEntropyLoss()

        self.log = logging.getLogger("conv2l_stdp")

    def _poisson_encode(self, image):
        """Generate Poisson spike trains from image.

        Reshapes the flat image tensor to (1, C, H, W) where C/H/W are
        set per-dataset at __init__ (1×28×28 for MNIST/FMNIST; 3×32×32
        for CIFAR-10 RGB).
        """
        H, W, C = self.input_h, self.input_w, self.input_channels
        img_2d = image.view(1, C, H, W).to(self.device)
        rates = (img_2d * self.intensity / 1000.0).clamp_(0, 1)
        spike_trains = torch.bernoulli(rates.expand(self.n_steps, C, H, W))
        return spike_trains

    @torch.no_grad()
    def extract_features(self, image, train_stdp=False):
        """Run both STDP layers and return concatenated pooled features."""
        spike_trains = self._poisson_encode(image)

        # Layer 1
        _, pooled1, l1_output_spikes = self.layer1.forward(spike_trains, train=train_stdp)

        # Layer 2: uses L1 output spikes as input
        _, pooled2, _ = self.layer2.forward(l1_output_spikes, train=train_stdp)

        # Concatenate features
        features = torch.cat([pooled1, pooled2])
        return features

    def train_stdp_layer1(self, train_set, n_train=60000, epoch=0,
                          start_i=0, ckpt_dir=None, ckpt_interval=5000):
        """Train Layer 1 with STDP. Supports mid-epoch resume via start_i.

        If ckpt_dir is given, saves `ckpt_L1_ep{epoch+1}_s{i+1}.pt` every
        ckpt_interval samples. The watchdog wrapper relies on these to
        recover from CUDA crashes without losing >ckpt_interval samples
        of work.
        """
        self.log.info(f"  --- STDP Layer 1 Training (epoch {epoch+1}, start_i={start_i}) ---")
        # Per-epoch RNG seed must be reproducible across resumes
        gen = torch.Generator().manual_seed(self.seed + epoch)
        indices = torch.randperm(len(train_set), generator=gen)[:n_train]

        t0 = time.time()
        for i in range(start_i, n_train):
            image, label = train_set[indices[i].item()]
            spike_trains = self._poisson_encode(image)
            self.layer1.forward(spike_trains, train=True)

            if (i + 1) % 500 == 0:
                done = (i + 1 - start_i)
                speed = done / max(time.time() - t0, 1e-6)
                self.log.info(f"    L1 [{i+1:5d}/{n_train}] Speed: {speed:.1f} img/s")

            # Mid-epoch checkpoint save (resume support)
            if ckpt_dir is not None and (i + 1) % ckpt_interval == 0:
                _path = os.path.join(ckpt_dir, f"ckpt_L1_ep{epoch+1}_s{i+1}.pt")
                self.save_checkpoint(_path, extra={
                    "phase": "stdp", "layer": 1, "epoch": epoch + 1, "sample": i + 1,
                })

        elapsed = time.time() - t0
        self.log.info(f"    L1 epoch {epoch+1} done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    def train_stdp_layer2(self, train_set, n_train=60000, epoch=0,
                          start_i=0, ckpt_dir=None, ckpt_interval=5000):
        """Train Layer 2 with STDP, using frozen Layer 1 features.
        Supports mid-epoch resume via start_i and periodic ckpt save.
        """
        self.log.info(f"  --- STDP Layer 2 Training (epoch {epoch+1}, start_i={start_i}) ---")
        gen = torch.Generator().manual_seed(self.seed + 1000 + epoch)
        indices = torch.randperm(len(train_set), generator=gen)[:n_train]

        t0 = time.time()
        for i in range(start_i, n_train):
            image, label = train_set[indices[i].item()]
            spike_trains = self._poisson_encode(image)

            # Forward through frozen L1
            _, _, l1_output_spikes = self.layer1.forward(spike_trains, train=False)

            # Train L2
            self.layer2.forward(l1_output_spikes, train=True)

            if (i + 1) % 500 == 0:
                done = (i + 1 - start_i)
                speed = done / max(time.time() - t0, 1e-6)
                self.log.info(f"    L2 [{i+1:5d}/{n_train}] Speed: {speed:.1f} img/s")

            if ckpt_dir is not None and (i + 1) % ckpt_interval == 0:
                _path = os.path.join(ckpt_dir, f"ckpt_L2_ep{epoch+1}_s{i+1}.pt")
                self.save_checkpoint(_path, extra={
                    "phase": "stdp", "layer": 2, "epoch": epoch + 1, "sample": i + 1,
                })

        elapsed = time.time() - t0
        self.log.info(f"    L2 epoch {epoch+1} done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    def extract_all_features(self, dataset, n_samples, batch_label=""):
        """Extract features for all samples (for readout training)."""
        self.log.info(f"  Extracting {batch_label} features ({n_samples} samples)...")
        features = []
        labels = []
        t0 = time.time()
        for i in range(n_samples):
            image, label = dataset[i]
            feat = self.extract_features(image, train_stdp=False)
            features.append(feat)
            labels.append(label)
            if (i + 1) % 5000 == 0:
                speed = (i + 1) / (time.time() - t0)
                eta = (n_samples - i - 1) / speed / 60
                self.log.info(f"    [{i+1}/{n_samples}] Speed: {speed:.1f} img/s, ETA: {eta:.1f} min")

        features = torch.stack(features).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        elapsed = time.time() - t0
        self.log.info(f"    Done in {elapsed:.0f}s. Shape: {features.shape}")
        return features, labels

    def train_readout(self, X_train, y_train, X_test, y_test,
                      n_epochs=30, batch_size=128):
        """Train the readout SNN with surrogate gradients."""
        self.log.info(f"  --- Surrogate Gradient Readout Training ---")
        self.log.info(f"    Features: {X_train.shape[1]}, Hidden: {self.readout.fc1.out_features}")
        self.log.info(f"    Train: {len(X_train)}, Test: {len(X_test)}")
        self.log.info(f"    Readout timesteps: {self.readout.n_steps}")
        self.log.info(f"    Epochs: {n_epochs}, Batch: {batch_size}")

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_test_acc = 0.0
        for epoch in range(n_epochs):
            self.readout.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                out = self.readout(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(xb)

            train_acc = correct / total * 100
            train_loss = total_loss / total

            # Test
            self.readout.eval()
            with torch.no_grad():
                test_out = self.readout(X_test)
                test_acc = (test_out.argmax(1) == y_test).float().mean().item() * 100

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            self.log.info(
                f"    Epoch {epoch+1:2d}/{n_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Train: {train_acc:.1f}% | "
                f"Test: {test_acc:.1f}% | "
                f"Best: {best_test_acc:.1f}%"
            )

        return best_test_acc

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
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)


# =============================================================================
# Resume helpers (mirror conv/conv2layer_voltage_stdp.py pattern)
# =============================================================================

def _find_latest_sample_ckpt(ckpt_dir, layer_name, epoch):
    """Find the latest `ckpt_{layer_name}_ep{epoch}_s{sample}.pt`.

    Returns (path, sample_index) or None. sample_index is the count of
    images already processed; the inner loop should resume at i=sample.
    """
    import glob, re
    pattern = os.path.join(ckpt_dir, f"ckpt_{layer_name}_ep{epoch}_s*.pt")
    files = glob.glob(pattern)
    best_s, best_f = -1, None
    for f in files:
        m = re.search(r"_s(\d+)\.pt$", f)
        if m:
            s = int(m.group(1))
            if s > best_s:
                best_s, best_f = s, f
    if best_f is None:
        return None
    return best_f, best_s


def _load_layer1_from_ckpt(net, path, device):
    """Load only L1 filters + theta from a sample/epoch checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    net.layer1.filters = ckpt["l1_filters"].to(device)
    net.layer1.theta   = ckpt["l1_theta"].to(device)


def _load_layer2_from_ckpt(net, path, device):
    """Load L1 (frozen) + L2 filters + theta from a sample/epoch checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    net.layer1.filters = ckpt["l1_filters"].to(device)
    net.layer1.theta   = ckpt["l1_theta"].to(device)
    net.layer2.filters = ckpt["l2_filters"].to(device)
    net.layer2.theta   = ckpt["l2_theta"].to(device)


# =============================================================================
# MAIN
# =============================================================================

def load_mnist(data_dir="./data", dataset="mnist"):
    """Load MNIST, Fashion-MNIST, or CIFAR-10 (kept name for backwards compat).

    For CIFAR-10 we keep the native 32×32 RGB (no DoG, no resize, no grayscale).
    The network must be constructed with input_channels=3, input_h=32, input_w=32.
    """
    name = (dataset or "mnist").lower()
    from data.medmnist_loader import is_medmnist, load_medmnist
    if is_medmnist(name):                       # 28x28, RGB->gray, flattened (N,784)
        return load_medmnist(name, flatten=True, data_dir=data_dir)
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    train_set = cls(data_dir, train=True, download=True, transform=transform)
    test_set  = cls(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def get_input_dims(dataset: str):
    """Return (channels, H, W) for the given dataset."""
    name = (dataset or "mnist").lower()
    from data.medmnist_loader import is_medmnist
    if name in ("mnist", "fmnist", "fashion_mnist", "fashionmnist") or is_medmnist(name):
        return 1, 28, 28   # MedMNIST converted to 1-channel 28x28
    if name in ("cifar10", "cifar"):
        return 3, 32, 32
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def main():
    parser = argparse.ArgumentParser(description="2-Layer Conv STDP + Surrogate Readout")
    parser.add_argument("--n_filters1", type=int, default=32)
    parser.add_argument("--ks1", type=int, default=5)
    parser.add_argument("--n_filters2", type=int, default=64)
    parser.add_argument("--ks2", type=int, default=5, help="Kernel size for layer 2")
    parser.add_argument("--epochs", type=int, default=3, help="STDP training epochs per layer")
    parser.add_argument("--n_train", type=int, default=60000)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--sim_time", type=float, default=350.0)
    parser.add_argument("--intensity", type=float, default=128.0)
    parser.add_argument("--norm", type=float, default=10.0)
    parser.add_argument("--readout_hidden", type=int, default=256)
    parser.add_argument("--readout_steps", type=int, default=50)
    parser.add_argument("--readout_lr", type=float, default=1e-3)
    parser.add_argument("--readout_epochs", type=int, default=30)
    parser.add_argument("--readout_batch", type=int, default=128)
    parser.add_argument("--log_file", type=str, default="conv2l_stdp.log")
    parser.add_argument("--checkpoint", type=str, default="conv2l_stdp_ckpt.pt")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed (default: 42)")
    parser.add_argument("--device", type=str, default=None, help="CUDA device, e.g. cuda:0 (default: auto)")
    parser.add_argument("--name-suffix", type=str, default="", dest="name_suffix",
                        help="Suffix appended to log_file and checkpoint names for multi-seed runs")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fmnist", "cifar10",
                                 "organamnist", "tissuemnist", "bloodmnist", "pathmnist"],
                        help="Dataset (default: mnist)")
    args = parser.parse_args()

    # Apply name suffix to log and checkpoint paths
    # (os is imported at module level — no local re-import or
    # Python treats `os` as local in this function and bails later.)
    if args.name_suffix:
        base_log, ext_log = os.path.splitext(args.log_file)
        args.log_file = f"{base_log}{args.name_suffix}{ext_log}"
        base_ckpt, ext_ckpt = os.path.splitext(args.checkpoint)
        args.checkpoint = f"{base_ckpt}{args.name_suffix}{ext_ckpt}"

    log = setup_logging(log_file=args.log_file)

    # Set global seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"{'='*60}")
    log.info(f"2-Layer Conv STDP + Surrogate Gradient Readout")
    log.info(f"{'='*60}")
    log.info(f"  Device: {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"  Layer 1: {args.n_filters1} x {args.ks1}x{args.ks1}")
    log.info(f"  Layer 2: {args.n_filters2} x {args.ks2}x{args.ks2}")

    in_c, in_h, in_w = get_input_dims(args.dataset)
    l1_h = in_h - args.ks1 + 1
    l1_w = in_w - args.ks1 + 1
    l2_h = l1_h - args.ks2 + 1
    l2_w = l1_w - args.ks2 + 1
    log.info(f"  Input: {in_c}x{in_h}x{in_w}  ({args.dataset})")
    log.info(f"  L1 output: {l1_h}x{l1_w}x{args.n_filters1}")
    log.info(f"  L2 output: {l2_h}x{l2_w}x{args.n_filters2}")
    n_feat_l1 = args.n_filters1 * (l1_h // 2) * (l1_w // 2)
    n_feat_l2 = args.n_filters2 * (l2_h // 2) * (l2_w // 2)
    log.info(f"  Features: L1={n_feat_l1} + L2={n_feat_l2} = {n_feat_l1 + n_feat_l2}")
    log.info(f"  STDP epochs: {args.epochs}")
    log.info(f"  Readout: {n_feat_l1 + n_feat_l2} -> {args.readout_hidden} -> 10")
    log.info(f"  Readout epochs: {args.readout_epochs}, LR: {args.readout_lr}")
    log.info(f"{'='*60}")

    from data.medmnist_loader import is_medmnist, n_classes as _medn
    n_out = _medn(args.dataset) if is_medmnist(args.dataset) else 10
    net = TwoLayerConvSTDP_SurrogateNet(
        n_filters1=args.n_filters1, ks1=args.ks1,
        n_filters2=args.n_filters2, ks2=args.ks2,
        sim_time=args.sim_time, intensity=args.intensity, norm=args.norm,
        readout_hidden=args.readout_hidden, readout_steps=args.readout_steps,
        readout_lr=args.readout_lr, device=device, seed=args.seed,
        input_channels=in_c, input_h=in_h, input_w=in_w, n_output=n_out,
    )

    train_set, test_set = load_mnist(dataset=args.dataset)
    log.info(f"  Dataset: {args.dataset}")
    n_train = min(args.n_train, len(train_set))
    n_test = min(args.n_test, len(test_set))

    total_start = time.time()

    # ==========================================
    # Phase 1: Train Layer 1 with STDP
    # ==========================================
    log.info(f"\n{'='*60}")
    log.info(f"PHASE 1: STDP Layer 1 Training ({args.epochs} epochs)")
    log.info(f"{'='*60}")

    # Per-run resume directory (one dir per checkpoint name, holds
    # ckpt_L{1,2}_ep{N}_s{sample}.pt and ckpt_L{1,2}_ep{N}.pt files).
    base_ckpt, _ = os.path.splitext(args.checkpoint)
    resume_dir = base_ckpt + "_resume"
    os.makedirs(resume_dir, exist_ok=True)
    log.info(f"  Resume dir: {resume_dir}")

    for epoch in range(args.epochs):
        epoch_done_path = os.path.join(resume_dir, f"ckpt_L1_ep{epoch+1}.pt")
        if os.path.exists(epoch_done_path):
            log.info(f"  [Resume] L1 epoch {epoch+1} already done; loading {epoch_done_path}")
            _load_layer1_from_ckpt(net, epoch_done_path, device)
            continue

        # Mid-epoch resume?
        start_i = 0
        sample_resume = _find_latest_sample_ckpt(resume_dir, "L1", epoch + 1)
        if sample_resume is not None:
            resume_path, start_i = sample_resume
            log.info(f"  [Resume] Mid-epoch L1 ep{epoch+1}: loading {resume_path} "
                     f"(resuming at sample {start_i})")
            _load_layer1_from_ckpt(net, resume_path, device)

        net.train_stdp_layer1(train_set, n_train=n_train, epoch=epoch,
                              start_i=start_i, ckpt_dir=resume_dir)

        # Epoch complete: save the epoch-end ckpt and clean up sample ckpts
        net.save_checkpoint(epoch_done_path, extra={
            "phase": "stdp", "layer": 1, "epoch": epoch + 1,
        })
        log.info(f"  [Saved] {epoch_done_path}")
        # Remove this epoch's sample ckpts to save disk
        import glob
        for f in glob.glob(os.path.join(resume_dir, f"ckpt_L1_ep{epoch+1}_s*.pt")):
            try:
                os.remove(f)
            except OSError:
                pass

    # ==========================================
    # Phase 2: Train Layer 2 with STDP
    # ==========================================
    log.info(f"\n{'='*60}")
    log.info(f"PHASE 2: STDP Layer 2 Training ({args.epochs} epochs)")
    log.info(f"{'='*60}")
    for epoch in range(args.epochs):
        epoch_done_path = os.path.join(resume_dir, f"ckpt_L2_ep{epoch+1}.pt")
        if os.path.exists(epoch_done_path):
            log.info(f"  [Resume] L2 epoch {epoch+1} already done; loading {epoch_done_path}")
            _load_layer2_from_ckpt(net, epoch_done_path, device)
            continue

        start_i = 0
        sample_resume = _find_latest_sample_ckpt(resume_dir, "L2", epoch + 1)
        if sample_resume is not None:
            resume_path, start_i = sample_resume
            log.info(f"  [Resume] Mid-epoch L2 ep{epoch+1}: loading {resume_path} "
                     f"(resuming at sample {start_i})")
            _load_layer2_from_ckpt(net, resume_path, device)

        net.train_stdp_layer2(train_set, n_train=n_train, epoch=epoch,
                              start_i=start_i, ckpt_dir=resume_dir)

        net.save_checkpoint(epoch_done_path, extra={
            "phase": "stdp", "layer": 2, "epoch": epoch + 1,
        })
        log.info(f"  [Saved] {epoch_done_path}")
        import glob
        for f in glob.glob(os.path.join(resume_dir, f"ckpt_L2_ep{epoch+1}_s*.pt")):
            try:
                os.remove(f)
            except OSError:
                pass

    # ==========================================
    # Phase 3: Extract features + train readout
    # ==========================================
    log.info(f"\n{'='*60}")
    log.info(f"PHASE 3: Feature Extraction + Surrogate Readout Training")
    log.info(f"{'='*60}")

    X_train, y_train = net.extract_all_features(train_set, n_train, batch_label="train")
    X_test, y_test = net.extract_all_features(test_set, n_test, batch_label="test")

    best_acc = net.train_readout(X_train, y_train, X_test, y_test,
                                  n_epochs=args.readout_epochs,
                                  batch_size=args.readout_batch)

    # ==========================================
    # Save & Summary
    # ==========================================
    total_time = time.time() - total_start
    net.save_checkpoint(args.checkpoint, extra={"best_acc": best_acc})

    log.info(f"\n{'='*60}")
    log.info(f"FINAL RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  Architecture: Conv {args.ks1}x{args.ks1}x{args.n_filters1} -> Conv {args.ks2}x{args.ks2}x{args.n_filters2} -> SNN Readout")
    log.info(f"  Features: {n_feat_l1 + n_feat_l2} (L1: {n_feat_l1}, L2: {n_feat_l2})")
    log.info(f"  Best Test Accuracy (surrogate readout): {best_acc:.1f}%")
    log.info(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f} hrs)")
    log.info(f"  Previous results:")
    log.info(f"    1-layer Conv STDP + voting:     64.8%")
    log.info(f"    1-layer Conv STDP + Linear SVM: 94.5%")
    log.info(f"    FC Triplet STDP (800n, 2ep):    80.8%")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
