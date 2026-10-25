"""Post-training metrics for the scaffolding-ablation experiments.

All metrics here operate on artefacts saved by run_single.py (final weights,
per-neuron class-rate matrix, per-sample spike counts on test set). Nothing
here depends on the model object, so metrics can be recomputed later from
saved files.

Outputs go into the per-run directory as described in EXPERIMENTAL_SPEC §5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------
# Primary metrics (spec §4.1)
# ----------------------------------------------------------------------

def classification_accuracy(test_spike_counts: np.ndarray,
                            assignments: np.ndarray,
                            test_labels: np.ndarray,
                            n_classes: int = 10) -> float:
    """Post-hoc Diehl & Cook classification.

    test_spike_counts: (n_test, n_neurons)
    assignments:       (n_neurons,) int in {-1, 0..n_classes-1}
    test_labels:       (n_test,)
    n_classes:         number of classes (10 for MNIST/FMNIST/CIFAR/N-MNIST,
                       8 TissueMNIST, 9 PathMNIST, 11 OrganAMNIST).

    For each test sample, vote = sum of spikes from neurons assigned to each
    class; predicted class = argmax(vote).
    """
    n_test = test_spike_counts.shape[0]
    votes = np.zeros((n_test, n_classes), dtype=np.float32)
    for c in range(n_classes):
        mask = assignments == c
        if mask.any():
            votes[:, c] = test_spike_counts[:, mask].sum(axis=1)
    preds = votes.argmax(axis=1)
    return float((preds == test_labels).mean())


def per_neuron_selectivity(class_rates: np.ndarray,
                           min_total_rate: float = 1.0) -> np.ndarray:
    """Selectivity = max_class_rate / mean_class_rate per neuron.

    class_rates: (n_neurons, 10) of cumulative spike counts per class.

    Returns (n_neurons,) selectivity values. Neurons whose total firing
    over training is below `min_total_rate` are returned as NaN — the
    metric is meaningless for dead units (denominator near zero blows up
    the ratio and contaminates the population mean). See
    `results/scaffold_ablation_main_matrix.md` review note 5 for why this
    matters: under S_inh_only most neurons are silent (≤ 99%) and the
    handful that do fire produce per-neuron selectivities of 20+, making
    "broken" cells look more selective than working ones.

    To recover the OLD (mean-of-everything) behaviour set
    `min_total_rate=0.0`.
    """
    total_rate = class_rates.sum(axis=1)
    mean_rate = class_rates.mean(axis=1)
    max_rate = class_rates.max(axis=1)
    out = np.full(mean_rate.shape, np.nan, dtype=np.float32)
    valid = (mean_rate > 0) & (total_rate >= min_total_rate)
    out[valid] = max_rate[valid] / mean_rate[valid]
    return out


def rf_cosine_similarity(weights_2d: np.ndarray) -> tuple[np.ndarray, float]:
    """Pairwise cosine similarity between learned receptive fields.

    weights_2d: (n_input, n_neurons). Each column is one neuron's RF.
    Returns (similarity_matrix (n_neurons, n_neurons), mean_off_diagonal).
    """
    W = weights_2d.astype(np.float64)
    norms = np.linalg.norm(W, axis=0, keepdims=True).clip(min=1e-12)
    Wn = W / norms
    sim = Wn.T @ Wn
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_off = float(sim[mask].mean())
    return sim.astype(np.float32), mean_off


def class_coverage(assignments: np.ndarray, n_classes: int = 10) -> float:
    """Fraction of classes for which at least one neuron is primarily selective.

    NOTE: this metric is **only meaningful for cells whose accuracy is
    materially above the test-set majority-class baseline** (~11.35 % on
    MNIST test set). For collapsed cells the post-hoc labeller can
    distribute neurons across all 10 classes (`coverage = 1.0`) while
    every test image is still predicted as one class — see §0 of
    `results/scaffold_ablation_main_matrix.md`. Aggregators should mask
    or annotate this column for cells whose `accuracy < 1.5 × majority_baseline`.
    """
    present = set(int(x) for x in assignments if x >= 0)
    return len(present) / n_classes


# ----------------------------------------------------------------------
# Secondary metrics (spec §4.2)
# ----------------------------------------------------------------------

def weight_histogram(w: np.ndarray, bins: int = 100) -> dict:
    flat = w.reshape(-1)
    counts, edges = np.histogram(flat, bins=bins)
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    return {
        "bin_edges": edges.astype(np.float32),
        "counts": counts.astype(np.int64),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "entropy": entropy,
    }


def firing_rate_cv(test_spike_counts: np.ndarray) -> float:
    """Coefficient of variation (std/mean) of mean firing rate across neurons."""
    rate_per_neuron = test_spike_counts.mean(axis=0)
    mean_rate = rate_per_neuron.mean()
    if mean_rate == 0:
        return float("nan")
    return float(rate_per_neuron.std() / mean_rate)


def nmi(test_spike_counts: np.ndarray, test_labels: np.ndarray) -> float:
    """Normalised Mutual Information between argmax-neuron and true label.

    For each test sample, the 'cluster id' is the argmax firing neuron
    (sample-level, NOT class-assignment). This is the unsupervised
    clustering NMI defended in Q4.
    """
    cluster_id = test_spike_counts.argmax(axis=1)
    # Inline NMI to avoid sklearn dependency if not available
    return _nmi_score(cluster_id, test_labels)


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    n_a = len(a_vals)
    n_b = len(b_vals)
    contingency = np.zeros((n_a, n_b), dtype=np.float64)
    for i in range(len(a)):
        contingency[a_inv[i], b_inv[i]] += 1
    p_ab = contingency / contingency.sum()
    p_a = p_ab.sum(axis=1, keepdims=True)
    p_b = p_ab.sum(axis=0, keepdims=True)
    denom = p_a @ p_b
    # MI = sum p_ab * log(p_ab / (p_a p_b))  over cells where p_ab>0
    mi = 0.0
    nz = p_ab > 0
    mi = float((p_ab[nz] * np.log(p_ab[nz] / denom[nz])).sum())
    return mi


def _nmi_score(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized MI using the geometric mean of entropies (sklearn default)."""
    ha = _entropy(a)
    hb = _entropy(b)
    if ha == 0 or hb == 0:
        return 0.0
    mi = _mutual_information(a, b)
    return float(mi / np.sqrt(ha * hb))


# ----------------------------------------------------------------------
# Qualitative outputs (spec §4.3)
# ----------------------------------------------------------------------

def _infer_rf_shape(n_in: int) -> tuple | None:
    """Best-effort guess of (C, H, W) or (H, W) for a flat input dimension.

    Supports the four datasets we use:
      - MNIST/FMNIST (784 = 28×28) → (28, 28)
      - CIFAR-10 RGB (3072 = 3×32×32) → (3, 32, 32)
      - N-MNIST event-sum (2312 = 2×34×34) → (2, 34, 34)

    Returns None if no known shape matches; caller should skip viz.
    """
    if n_in == 28 * 28:
        return (28, 28)
    if n_in == 3 * 32 * 32:
        return (3, 32, 32)
    if n_in == 2 * 34 * 34:
        return (2, 34, 34)
    # Generic square fallback
    s = int(round(n_in ** 0.5))
    if s * s == n_in:
        return (s, s)
    return None


def _rf_to_2d(rf: np.ndarray, shape: tuple) -> np.ndarray:
    """Reshape a flat RF to (H, W) for display.

    Multi-channel (C, H, W) inputs are averaged across channels so the grid
    still renders as a single 2D heatmap per neuron.
    """
    if len(shape) == 2:
        return rf.reshape(shape)
    # (C, H, W) → mean over C
    return rf.reshape(shape).mean(axis=0)


def save_rf_grid_png(weights_2d: np.ndarray, assignments: np.ndarray,
                     out_path: str, rows: int = 10, cols: int = 10,
                     img_hw: tuple | None = None):
    """Save a rows×cols grid of learned RFs.

    Ordering (Q13 + spec §4.3): group by primary class, then within class
    sort by total synaptic input (sum of weights). Neurons never assigned
    (-1) sort last.

    `img_hw` can be (H, W) for grayscale or (C, H, W) for multi-channel
    inputs (averaged across C for display). If `img_hw` is None, the shape
    is inferred from `weights_2d.shape[0]` via `_infer_rf_shape`; if no
    known shape matches, the function returns silently (no PNG written).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_in, n_out = weights_2d.shape
    if img_hw is None:
        img_hw = _infer_rf_shape(n_in)
        if img_hw is None:
            # Caller didn't pass a shape and we can't guess one. Skip silently
            # — visualization is non-essential, never block a run on it.
            return

    total_rate = weights_2d.sum(axis=0)
    primary = assignments.copy()
    primary_sort_key = np.where(primary < 0, 999, primary)
    order = np.lexsort((-total_rate, primary_sort_key))
    ordered_neurons = order[: rows * cols]

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if i >= len(ordered_neurons):
            ax.axis("off")
            continue
        n = ordered_neurons[i]
        rf = _rf_to_2d(weights_2d[:, n], img_hw)
        ax.imshow(rf, cmap="viridis")
        cls = int(assignments[n]) if assignments[n] >= 0 else -1
        ax.set_title(f"n={n} c={cls}", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------
# Aggregate the full metrics bundle for one run
# ----------------------------------------------------------------------

def compute_all(run_dir: Path,
                final_weights: np.ndarray,
                train_class_rates: np.ndarray,
                assignments: np.ndarray,
                test_spike_counts: np.ndarray,
                test_labels: np.ndarray,
                n_classes: int | None = None) -> dict:
    """Compute every metric required by spec §4. Saves arrays to run_dir and
    returns a dict of scalar metrics for metrics.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Infer n_classes from train_class_rates buffer width (shape: n_neurons, n_classes)
    if n_classes is None:
        n_classes = int(train_class_rates.shape[1])

    acc = classification_accuracy(test_spike_counts, assignments, test_labels, n_classes=n_classes)
    sel = per_neuron_selectivity(train_class_rates)
    sim_matrix, mean_off = rf_cosine_similarity(final_weights)
    coverage = class_coverage(assignments, n_classes=n_classes)
    hist = weight_histogram(final_weights)
    fr_cv = firing_rate_cv(test_spike_counts)
    nmi_val = nmi(test_spike_counts, test_labels)

    # Save arrays
    np.save(run_dir / "selectivity.npy", sel)
    np.save(run_dir / "similarity_matrix.npy", sim_matrix)
    np.save(run_dir / "weight_histogram.npy",
            np.stack([hist["bin_edges"][:-1], hist["counts"].astype(np.float32)], axis=0))
    np.savez(run_dir / "weights_final.npz", w=final_weights)

    metrics = {
        "accuracy": acc,
        "mean_cosine_similarity": mean_off,
        "class_coverage": coverage,
        "firing_rate_cv": fr_cv,
        "nmi": nmi_val,
        "weight_mean": hist["mean"],
        "weight_std": hist["std"],
        "weight_min": hist["min"],
        "weight_max": hist["max"],
        "weight_entropy": hist["entropy"],
        "n_neurons_assigned": int((assignments >= 0).sum()),
        "selectivity_mean": float(np.nanmean(sel)) if np.any(~np.isnan(sel)) else float("nan"),
        "selectivity_median": float(np.nanmedian(sel)) if np.any(~np.isnan(sel)) else float("nan"),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
