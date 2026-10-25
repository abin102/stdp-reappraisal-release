"""Run one (rule × scaffold × seed × N) cell of the factorial matrix.

Outputs are written to:
    {save_dir}/{rule}_{scaffold}_N{n_neurons}_seed{seed}/

    config.yaml           - exact resolved config used (reproducibility)
    log.txt               - training log
    weights_final.npz     - final W of shape (n_input, n_neurons)
    metrics.json          - all scalar metrics
    selectivity.npy       - per-neuron selectivity vector
    similarity_matrix.npy - (n_neurons, n_neurons) pairwise cosine similarity
    weight_histogram.npy  - (2, bins) = [edges, counts]
    training_curve.csv    - per-log_interval rolling summary
    receptive_fields.png  - 10x10 RF grid
    test_spike_counts.npz - per-test-sample spike counts (reproducible metrics)
    checkpoint.pt         - full model state (reproducibility)
    progress_checkpoint.pt - mid-Phase-1 resume state, atomically rewritten
                            every PROGRESS_SAVE_INTERVAL samples (default 5000).
                            Removed on successful completion. If the run is
                            killed mid-Phase-1, re-running the same cell will
                            load this and fast-forward to the saved sample.
    completed.flag        - empty file; presence = run completed

Usage:
    python -m experiments.scripts.run_single --rule triplet --scaffold full \
        --seed 0 --n_neurons 400

    python -m experiments.scripts.run_single --rule triplet --scaffold full \
        --seed 0 --n_neurons 400 --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision import datasets, transforms

from experiments.scripts.config import resolve
from experiments.scripts import metrics as M
from models.fc_stdp_layer import FCStdpLayer


# ----------------------------------------------------------------------

def set_global_seeds(seed: int):
    """Per spec §7.7 and the agreed plan — deterministic seeds across libs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir: str, dataset: str = "mnist"):
    """Load MNIST, Fashion-MNIST, CIFAR-10, or N-MNIST with a flat-vector transform.

    Args:
        data_dir: directory passed to torchvision (will download if missing).
        dataset: 'mnist' (default, backwards-compatible), 'fmnist', 'cifar10', or 'nmnist'.

    For CIFAR-10 we keep all 3 RGB channels and flatten to a 3072-dim
    vector (3 * 32 * 32). This keeps the methodology apples-to-apples
    with our MNIST/FMNIST runs (no DoG, no thresholding, no explicit
    preprocessing beyond min-max normalisation to [0,1] by ToTensor).

    For N-MNIST we sum all events over the full sample duration into a
    (2, 34, 34) rate histogram, per-sample min-max normalise to [0,1],
    and flatten to a 2312-dim vector. This sacrifices N-MNIST's temporal
    structure but holds the input encoding constant across all four
    datasets (MNIST/FMNIST/CIFAR-10/N-MNIST) so the scaffold-ablation
    comparison isn't confounded by encoding differences.
    """
    name = dataset.lower()
    if name == "mnist":
        cls = datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    elif name in ("mnist_blend", "mnistblend"):
        # Synthetic pixel-sparsity manipulation dataset.
        #
        # Each MNIST image x is blended with deterministic per-image uniform
        # noise z_i ~ U(0,1) (seed = sample index, fixed across runs):
        #     y_i = (1 - alpha) * x_i + alpha * z_i
        # Where alpha ∈ [0, 1] is read from STDP_BLEND_ALPHA env var. At
        # alpha=0 the data is pure MNIST (frac<0.05 ≈ 82 %); at alpha=1 the
        # data is pure noise (frac<0.05 ≈ 5 %). This sweeps the input
        # pixel-intensity sparsity smoothly while keeping the dataset
        # geometry / class structure / per-image dimensionality fixed.
        #
        # Class labels are unchanged. At high alpha the classification
        # signal is washed out, but the experiment is *not* about
        # downstream accuracy — it's about whether the pair-STDP
        # convergence regime tracks the sparsity continuously.
        import os
        alpha = float(os.environ.get("STDP_BLEND_ALPHA", "0.0"))

        from torch.utils.data import Dataset
        class _MNISTBlend(Dataset):
            def __init__(self, save_to, train):
                self.inner = datasets.MNIST(save_to, train=train,
                                            download=True,
                                            transform=transforms.ToTensor())
                self.alpha = alpha
                self.train = train

            def __len__(self):
                return len(self.inner)

            def __getitem__(self, idx):
                img, lbl = self.inner[idx]
                if self.alpha > 0:
                    g = torch.Generator(device="cpu").manual_seed(
                        int(idx) * 1009 + (0 if self.train else 7))
                    z = torch.rand(img.shape, generator=g)
                    img = (1.0 - self.alpha) * img + self.alpha * z
                return img.view(-1), lbl

        train_set = _MNISTBlend(data_dir, train=True)
        test_set  = _MNISTBlend(data_dir, train=False)
        return train_set, test_set
    elif name in ("fmnist", "fashion_mnist", "fashionmnist"):
        cls = datasets.FashionMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    elif name in ("kmnist", "k_mnist"):
        # Kuzushiji-MNIST: Japanese cursive Hiragana characters,
        # 28x28 grayscale, 10 classes, 60k train / 10k test.
        # Third sparse-stroke dataset for voltage-Clopath topology
        # confirmation (after MNIST + N-MNIST).
        # NOTE: as of 2026-05-17, the official CODH JP mirror times out
        # from this network — use EMNIST-letters instead.
        cls = datasets.KMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    elif name in ("emnist_letters", "emnist-letters", "emnistletters"):
        # EMNIST-letters: NIST handwritten letter dataset, 26 classes,
        # 28x28 grayscale, 124800 train / 20800 test.
        # Third sparse-stroke dataset for voltage-Clopath topology
        # confirmation (after MNIST + N-MNIST), replacing KMNIST whose
        # JP mirror is unreachable.
        # Labels are 1-indexed in EMNIST-letters (1..26); we shift to 0..25
        # via target_transform so n_classes=26 fits cleanly.
        from torchvision.datasets import EMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        target_transform = transforms.Lambda(lambda y: int(y) - 1)
        train_set = EMNIST(data_dir, split="letters", train=True,
                            download=True, transform=transform,
                            target_transform=target_transform)
        test_set  = EMNIST(data_dir, split="letters", train=False,
                            download=True, transform=transform,
                            target_transform=target_transform)
        return train_set, test_set
    elif name in ("cifar10", "cifar"):
        cls = datasets.CIFAR10
        # CIFAR-10 is 32x32x3 RGB. Keep all 3 color channels and flatten to
        # 3072-dim (3*32*32). Per-channel min-max normalisation is done by
        # transforms.ToTensor() (which maps [0,255] uint8 -> [0,1] float32).
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    elif name in ("nmnist", "n_mnist"):
        import tonic
        # Tonic returns a structured numpy array of events with fields
        # (x, y, t, p). sensor_size is (W, H, P) = (34, 34, 2).
        sx, sy, sp = tonic.datasets.NMNIST.sensor_size

        def event_sum_transform(events):
            img = np.zeros((sp, sy, sx), dtype=np.float32)
            np.add.at(img,
                      (events['p'].astype(np.int64),
                       events['y'].astype(np.int64),
                       events['x'].astype(np.int64)),
                      1.0)
            m = img.max()
            if m > 0:
                img /= m
            return torch.from_numpy(img.reshape(-1))

        train_set = tonic.datasets.NMNIST(save_to=data_dir, train=True,
                                          transform=event_sum_transform)
        test_set = tonic.datasets.NMNIST(save_to=data_dir, train=False,
                                         transform=event_sum_transform)
        return train_set, test_set
    elif name in ("tissuemnist", "tissue_mnist", "pathmnist", "path_mnist",
                  "organamnist", "organa_mnist",
                  "bloodmnist", "blood_mnist"):
        # MedMNIST 28x28 datasets — uniform wrapper for grayscale (TissueMNIST,
        # 8 classes kidney cortex; OrganAMNIST, 11 classes abdominal CT) and
        # RGB (PathMNIST, 9 classes histopathology; BloodMNIST, 8 classes
        # blood cells). All texture-classified by design — dense-texture
        # confirmation for the topology hypothesis (cf. log.md 2026-05-16..18).
        #
        # NOTE: BloodMNIST is converted to grayscale via PIL.convert("L")
        # to keep the input dimensionality at 784 (matching FMNIST /
        # OrganAMNIST / EMNIST-letters) so the cross-dataset comparison is
        # apples-to-apples on dimensionality. PathMNIST's collapse at 2352-dim
        # suggests RGB MedMNIST + 1-epoch budget is a poor combination.
        # Returns (flat float32 tensor in [0,1], scalar int label).
        from torch.utils.data import Dataset

        if name in ("tissuemnist", "tissue_mnist"):
            from medmnist import TissueMNIST as _MedCls
            _name_label = "tissuemnist"
            _force_grayscale = False  # native grayscale
        elif name in ("organamnist", "organa_mnist"):
            from medmnist import OrganAMNIST as _MedCls
            _name_label = "organamnist"
            _force_grayscale = False  # native grayscale
        elif name in ("bloodmnist", "blood_mnist"):
            from medmnist import BloodMNIST as _MedCls
            _name_label = "bloodmnist"
            _force_grayscale = True   # native RGB → force luminance
        else:
            from medmnist import PathMNIST as _MedCls
            _name_label = "pathmnist"
            _force_grayscale = True   # native RGB → force luminance to 784-dim,
                                      # consistent with the other medical sets
                                      # (OrganA/Blood/Tissue) for apples-to-apples

        class _MedMNISTWrap(Dataset):
            """torchvision-style wrapper around medmnist 28x28 datasets."""
            def __init__(self, save_to, train, transform):
                split = "train" if train else "test"
                # as_rgb=False keeps native channels:
                #   TissueMNIST / OrganAMNIST → grayscale (784-dim)
                #   PathMNIST / BloodMNIST   → RGB (2352-dim, unless
                #                                  _force_grayscale=True)
                self.inner = _MedCls(split=split, download=True,
                                     root=save_to, as_rgb=False)
                self.transform = transform

            def __len__(self):
                return len(self.inner)

            def __getitem__(self, idx):
                img, lbl = self.inner[idx]
                if _force_grayscale and img.mode != "L":
                    img = img.convert("L")
                x = self.transform(img)
                # medmnist returns label as 1-d ndarray e.g. array([3]).
                y = int(np.asarray(lbl).reshape(-1)[0])
                return x, y

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_set = _MedMNISTWrap(data_dir, train=True, transform=transform)
        test_set  = _MedMNISTWrap(data_dir, train=False, transform=transform)
        return train_set, test_set
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}; expected 'mnist', 'fmnist', "
            f"'cifar10', 'nmnist', 'kmnist', 'emnist_letters', 'tissuemnist', "
            f"'pathmnist', 'organamnist', or 'bloodmnist'")
    train_set = cls(data_dir, train=True, download=True, transform=transform)
    test_set  = cls(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def get_input_dim(dataset: str) -> int:
    """Return the flat input dimensionality for a given dataset.

    MNIST/FMNIST/TissueMNIST/OrganAMNIST/PathMNIST = 28*28 = 784 (grayscale;
        PathMNIST/BloodMNIST are RGB natively but the loader forces luminance)
    CIFAR-10 (RGB)                         = 3*32*32 = 3072
    N-MNIST                                = 2*34*34 = 2312 (2 polarities × 34×34 spatial)
    """
    name = dataset.lower()
    if name in ("mnist", "fmnist", "fashion_mnist", "fashionmnist",
                "kmnist", "k_mnist",
                "mnist_blend", "mnistblend",
                "emnist_letters", "emnist-letters", "emnistletters",
                "tissuemnist", "tissue_mnist",
                "organamnist", "organa_mnist",
                "bloodmnist", "blood_mnist",
                "pathmnist", "path_mnist"):   # now grayscale 784 (loader forces L)
        return 784
    if name in ("cifar10", "cifar"):
        return 3072
    if name in ("nmnist", "n_mnist"):
        return 2312
    raise ValueError(f"Unknown dataset for input dim: {dataset!r}")


def get_n_classes(dataset: str) -> int:
    """Return the number of classes for a given dataset.

    MNIST/FMNIST/CIFAR10/N-MNIST = 10
    TissueMNIST / BloodMNIST     = 8
    PathMNIST                     = 9
    OrganAMNIST                   = 11
    EMNIST-letters                = 26
    """
    name = dataset.lower()
    if name in ("mnist", "fmnist", "fashion_mnist", "fashionmnist",
                "kmnist", "k_mnist",
                "mnist_blend", "mnistblend",
                "cifar10", "cifar", "nmnist", "n_mnist"):
        return 10
    if name in ("emnist_letters", "emnist-letters", "emnistletters"):
        return 26
    if name in ("tissuemnist", "tissue_mnist",
                "bloodmnist", "blood_mnist"):
        return 8
    if name in ("pathmnist", "path_mnist"):
        return 9
    if name in ("organamnist", "organa_mnist"):
        return 11
    raise ValueError(f"Unknown dataset for n_classes: {dataset!r}")


def load_mnist(data_dir: str):
    """Backwards-compatible wrapper around load_dataset."""
    return load_dataset(data_dir, dataset="mnist")


def setup_logger(log_path: Path, run_name: str) -> logging.Logger:
    logger = logging.getLogger(f"run.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w"); fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ----------------------------------------------------------------------

def train_and_eval(cfg: dict, run_dir: Path, log: logging.Logger) -> dict:
    device = torch.device(cfg["experiment"]["device"]) \
             if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"device = {device}")

    # Auto-set model.n_input from dataset (CIFAR-10 RGB = 3072, MNIST/FMNIST = 784)
    # Auto-set data.n_classes (10 for most; 8 TissueMNIST / 9 PathMNIST / 11 OrganAMNIST)
    dataset_name = cfg.get("data", {}).get("dataset", "mnist")
    expected_n_input = get_input_dim(dataset_name)
    n_classes = get_n_classes(dataset_name)
    cfg.setdefault("model", {})["n_input"] = expected_n_input
    cfg.setdefault("data", {})["n_classes"] = n_classes

    t0 = time.time()
    net = FCStdpLayer(cfg, device=device)
    log.info(f"net: rule={net.rule_name} neuron={net.neuron_model} "
             f"N={net.n_neurons} inh={net.inh_weight} theta+={net.theta_plus} "
             f"decay_theta={net.decay_theta:.4f}")

    log.info(f"dataset = {dataset_name} (n_input={expected_n_input})")
    train_set, test_set = load_dataset(cfg["data"]["data_dir"], dataset_name)
    n_train = int(cfg["training"]["n_train"])
    n_test = int(cfg["training"]["n_test"])
    threshold = cfg.get("data", {}).get("threshold", None)

    # Deterministic training order from seed
    seed_int = int(cfg["experiment"]["seed"])
    g = torch.Generator().manual_seed(seed_int + 1_000)
    train_order = torch.randperm(len(train_set), generator=g).tolist()[: n_train]
    # Test images in a fixed random order (seed + 3000), not in stored order. EMNIST-Letters
    # and N-MNIST store their test sets sorted by class; the adaptive threshold keeps adapting
    # during evaluation, so a class-sorted pass changes the spike counts class block by class
    # block (2026-10-19). The subset is unchanged: the first min(n_test, len) test images.
    g_test = torch.Generator().manual_seed(seed_int + 3_000)
    test_order = torch.randperm(min(n_test, len(test_set)), generator=g_test).tolist()

    # ----- Mid-Phase-1 resume support ----------------------------------
    # If a previous attempt was killed mid-Phase-1, a "progress_checkpoint.pt"
    # in run_dir holds the model state (weights/theta/rates/_fire_total/
    # _n_images_seen) and the sample index we had completed. On (re)start,
    # load it and fast-forward to that sample. The save itself is atomic
    # (tmp + os.replace) so a crash mid-save can't corrupt the file.
    progress_ckpt_path = run_dir / "progress_checkpoint.pt"
    # Save interval: every PROGRESS_SAVE_INTERVAL samples. At ~6 samples/s on
    # cuda:1 this is ~14 min between writes -> at most ~14 min of work lost
    # per kill. Override via env var PROGRESS_SAVE_INTERVAL if desired.
    progress_save_interval = int(os.environ.get("PROGRESS_SAVE_INTERVAL", "5000"))

    i_start = 0
    fire_window = 0
    curve_mode = "w"  # default: fresh file with header
    if progress_ckpt_path.exists():
        try:
            pck = torch.load(progress_ckpt_path,
                             map_location=device,
                             weights_only=False)
            # Safety: refuse to resume if the run signature (seed, n_train,
            # n_neurons, rule) doesn't match — protects against accidental
            # cross-config resumes.
            saved_sig = pck.get("_run_sig", {})
            run_sig = {
                "seed": seed_int,
                "n_train": n_train,
                "n_neurons": int(net.n_neurons),
                "rule": net.rule_name,
                "n_input": int(expected_n_input),
            }
            if saved_sig and saved_sig != run_sig:
                log.warning(
                    f"RESUME aborted: signature mismatch "
                    f"saved={saved_sig} current={run_sig}. "
                    f"Renaming stale progress_checkpoint.pt and starting fresh."
                )
                progress_ckpt_path.rename(
                    progress_ckpt_path.with_suffix(".pt.stale"))
            else:
                net.load_checkpoint(str(progress_ckpt_path))
                i_start = int(pck.get("i_progress", -1)) + 1
                fire_window = int(pck.get("fire_window", 0))
                curve_mode = "a"  # append, don't re-write header
                log.info(
                    f"RESUME: progress_checkpoint.pt found; continuing "
                    f"Phase 1 from sample {i_start}/{n_train} "
                    f"(saved n_images_seen={pck.get('n_images_seen','?')})"
                )
        except Exception as e:
            log.warning(
                f"RESUME failed ({type(e).__name__}: {e}); "
                f"restarting Phase 1 from sample 0"
            )
            i_start = 0
            fire_window = 0
            curve_mode = "w"

    # When n_train exceeds dataset size, randperm has already capped train_order.
    # Pin the loop iteration count to the actual order length so OrganAMNIST
    # (34561 train samples) does not try to index past len(train_order) when
    # cfg defaults n_train=60000.
    n_train_effective = len(train_order)
    if n_train_effective < n_train:
        log.info(f"NOTE: n_train={n_train} > dataset size; capping to "
                 f"{n_train_effective}")
        n_train = n_train_effective

    # Frozen-init control (random/untrained-filter baseline): when
    # training.frozen_init=true, Phase 1 runs with STDP DISABLED (train=False),
    # so weights stay at the random D&C init; the pass still accumulates class
    # rates + lets theta adapt so post-hoc assignments / Phase-3 features are
    # produced by the IDENTICAL pipeline. Isolates "what untrained filters give"
    # vs the STDP-trained runs. Drive via --override training.frozen_init=true.
    frozen_init = bool(cfg.get("training", {}).get("frozen_init", False))
    if frozen_init:
        log.info("FROZEN-INIT control: STDP disabled (train=False); weights stay "
                 "at random init. Phase 1 only accumulates rates/theta for "
                 "post-hoc assignments — this is the untrained-filter baseline.")

    # Optional hard binary threshold to artificially sparsify dense inputs
    # (data.threshold). image -> (image > thr).float(), applied IDENTICALLY in
    # Phase 1 (train) and Phase 3 (test) so input statistics match. None = off.
    # NOTE: choose thr from the dataset's pixel distribution — e.g. BloodMNIST
    # (mean~0.70) needs thr~0.86 to reach MNIST-like ~12% active; thr=0.5 there
    # leaves ~80% active (denser, not sparser).
    threshold = cfg.get("data", {}).get("threshold", None)
    if threshold is not None:
        threshold = float(threshold)
        log.info(f"INPUT THRESHOLD active: binarizing images at > {threshold} "
                 f"(sparsification intervention)")

    # Accumulate class rates on the training order for post-hoc labelling (spec §4.1)
    if i_start == 0:
        log.info(f"PHASE 1: unsupervised training on {n_train} samples "
                 f"({cfg['training']['n_epochs']} epoch)...")
    else:
        log.info(f"PHASE 1 (resumed): {n_train - i_start} samples remaining "
                 f"of {n_train}")

    curve_path = run_dir / "training_curve.csv"
    curve_f = open(curve_path, curve_mode, newline="")
    curve_writer = csv.writer(curve_f)
    if curve_mode == "w":
        curve_writer.writerow(["sample", "mean_weight", "weight_std",
                               "mean_theta", "mean_fire_per_image"])

    log_interval = int(cfg["training"]["log_interval"])
    t_phase1 = time.time()

    def _save_progress(idx_in_epoch: int, fw: int):
        """Atomic mid-Phase-1 progress checkpoint write (tmp + os.replace)."""
        tmp = progress_ckpt_path.with_suffix(".pt.tmp")
        run_sig = {
            "seed": seed_int,
            "n_train": n_train,
            "n_neurons": int(net.n_neurons),
            "rule": net.rule_name,
            "n_input": int(expected_n_input),
        }
        net.save_checkpoint(
            str(tmp),
            extra={
                "i_progress": int(idx_in_epoch),
                "fire_window": int(fw),
                "_run_sig": run_sig,
            },
        )
        os.replace(tmp, progress_ckpt_path)

    for idx_in_epoch in range(i_start, n_train):
        idx = train_order[idx_in_epoch]
        image, label = train_set[idx]
        if threshold is not None:
            image = (image > threshold).float()
        spike_count = net.run_sample(image, train=not frozen_init)
        net.update_rates(spike_count, label)
        fire_window += float(spike_count.sum().item())

        if (idx_in_epoch + 1) % log_interval == 0:
            mean_w = float(net.w.mean().item())
            std_w = float(net.w.std().item())
            mean_theta = float(net.theta.mean().item())
            mean_fire = fire_window / log_interval
            log.info(f"  [{idx_in_epoch+1:>6}/{n_train}] "
                     f"w={mean_w:.4f}±{std_w:.4f}  theta={mean_theta:.4f}  "
                     f"fire/img={mean_fire:.1f}")
            curve_writer.writerow([idx_in_epoch + 1, mean_w, std_w, mean_theta, mean_fire])
            curve_f.flush()
            fire_window = 0

        if (idx_in_epoch + 1) % progress_save_interval == 0:
            _save_progress(idx_in_epoch, fire_window)
    curve_f.close()
    log.info(f"phase 1 done in {(time.time()-t_phase1)/60:.1f} min")

    # Post-hoc neuron labelling
    log.info("PHASE 2: computing neuron class assignments (post-hoc labelling)")
    net.compute_assignments()
    log.info(f"  neurons assigned: {int((net.assignments >= 0).sum().item())}/{net.n_neurons}")

    # Optional frozen pass over the training images (evaluation.save_train_features,
    # default off). Weights are frozen (train=False); the adaptive threshold keeps
    # adapting, as in the test pass below. Saves train_spike_counts.npz so that the
    # linear probe and the post-hoc labels can both be fitted on training-image
    # features and scored on the test set (added 2026-10-18 for the probe re-run).
    if bool(cfg.get("evaluation", {}).get("save_train_features", False)):
        log.info(f"PHASE 2b: frozen pass over the {len(train_order)} training images")
        t_phase2b = time.time()
        train_spike_counts = np.zeros((len(train_order), net.n_neurons), dtype=np.float32)
        train_labels_arr = np.zeros(len(train_order), dtype=np.int64)
        for i, idx in enumerate(train_order):
            image, label = train_set[idx]
            if threshold is not None:
                image = (image > threshold).float()
            sc = net.run_sample(image, train=False)
            train_spike_counts[i] = sc.detach().cpu().numpy()
            train_labels_arr[i] = label
        np.savez_compressed(run_dir / "train_spike_counts.npz",
                            train_spike_counts=train_spike_counts,
                            train_labels=train_labels_arr)
        log.info(f"phase 2b done in {(time.time()-t_phase2b)/60:.1f} min")

    # Evaluation on test set
    log.info(f"PHASE 3: evaluating on {len(test_order)} test samples")
    t_phase3 = time.time()
    test_spike_counts = np.zeros((len(test_order), net.n_neurons), dtype=np.float32)
    test_labels_arr = np.zeros(len(test_order), dtype=np.int64)
    for i, idx in enumerate(test_order):
        image, label = test_set[idx]
        if threshold is not None:
            image = (image > threshold).float()
        sc = net.run_sample(image, train=False)
        test_spike_counts[i] = sc.detach().cpu().numpy()
        test_labels_arr[i] = label
    log.info(f"phase 3 done in {(time.time()-t_phase3)/60:.1f} min")

    # Save checkpoint for reproducibility (per user request: "document every
    # result with its checkpoint")
    net.save_checkpoint(str(run_dir / "checkpoint.pt"), extra={"config": cfg})

    # Metrics (spec §4)
    final_weights = net.w.detach().cpu().numpy()     # (n_input, n_neurons)
    class_rates = net.rates.detach().cpu().numpy()    # (n_neurons, 10)
    assignments = net.assignments.detach().cpu().numpy()
    metrics = M.compute_all(
        run_dir=run_dir,
        final_weights=final_weights,
        train_class_rates=class_rates,
        assignments=assignments,
        test_spike_counts=test_spike_counts,
        test_labels=test_labels_arr,
    )

    # Save raw test spike counts FIRST (before optional viz step) so any
    # failure in the receptive-field PNG never loses the evaluation data
    # needed for logreg recomputation. Lesson from the 2026-05-12 CIFAR-10
    # smoke gate: save_rf_grid_png crashed on the 3072-dim weight column
    # but Phase 3 had already completed; spike counts were lost.
    np.savez_compressed(run_dir / "test_spike_counts.npz",
                        test_spike_counts=test_spike_counts,
                        test_labels=test_labels_arr,
                        test_indices=np.asarray(test_order, dtype=np.int64),
                        class_rates=class_rates,
                        assignments=assignments)

    # Receptive-field grid PNG (best-effort; never block the run on viz)
    try:
        M.save_rf_grid_png(
            weights_2d=final_weights,
            assignments=assignments,
            out_path=str(run_dir / "receptive_fields.png"),
            rows=int(cfg["evaluation"]["rf_grid_rows"]),
            cols=int(cfg["evaluation"]["rf_grid_cols"]),
        )
    except Exception as e:
        log.warning(f"save_rf_grid_png failed (non-fatal): "
                    f"{type(e).__name__}: {e}")

    metrics["elapsed_minutes"] = (time.time() - t0) / 60
    log.info(f"DONE | acc={metrics['accuracy']*100:.2f}%  nmi={metrics['nmi']:.3f}  "
             f"coverage={metrics['class_coverage']*100:.0f}%  "
             f"cosine_sim={metrics['mean_cosine_similarity']:.3f}  "
             f"elapsed={metrics['elapsed_minutes']:.1f}min")

    # Re-write metrics.json now that elapsed_minutes is known, so downstream
    # aggregators can recover it without parsing master_log.tsv.
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Mark complete
    (run_dir / "completed.flag").write_text(
        f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"accuracy={metrics['accuracy']*100:.2f}%\n"
    )

    # Phase 1 + 2 + 3 all succeeded; the mid-Phase-1 progress checkpoint is
    # no longer useful — remove it so a follow-up rerun doesn't try to
    # resume from it. (The full checkpoint.pt remains, as does any
    # progress_checkpoint.pt.stale renamed by the safety check on resume.)
    try:
        if progress_ckpt_path.exists():
            progress_ckpt_path.unlink()
    except Exception as e:
        log.warning(f"could not remove progress_checkpoint.pt: "
                    f"{type(e).__name__}: {e}")
    return metrics


# ----------------------------------------------------------------------

def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply `dotted.key=value` overrides to cfg in place.

    Value is YAML-parsed, so numeric/bool strings become their Python type.
    Used for ad-hoc parameter sweeps without editing config files.
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"--override must be key.path=value; got {ov!r}")
        key, _, raw = ov.partition("=")
        path = key.strip().split(".")
        value = yaml.safe_load(raw)
        d = cfg
        for p in path[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[path[-1]] = value


def run_cell(rule: str, scaffold: str, seed: int, n_neurons: int,
             device: str | None = None, smoke: bool = False,
             save_root: str | None = None,
             weight_norm_off: bool = False,
             n_train_override: int | None = None,
             name_suffix: str | None = None,
             overrides: list[str] | None = None,
             dataset: str | None = None) -> dict:
    """Programmatic entry point (used by run_matrix.py).

    n_train_override: custom train-set size (takes precedence over smoke
                      defaults). Used for validation smokes larger than 5k.
    name_suffix:      appended to experiment name to distinguish ad-hoc runs
                      from the canonical matrix cells.
    overrides:        list of `dotted.key=yaml_value` strings applied to the
                      resolved config after all other flags (for ad-hoc
                      parameter sweeps; see --override in CLI).
    """
    cfg = resolve(rule, scaffold)
    cfg["experiment"]["seed"] = int(seed)
    if dataset:
        # Wire dataset into cfg["data"] (created if missing) so load_dataset() picks it up
        cfg.setdefault("data", {})["dataset"] = dataset
    ds_tag = ""
    if dataset and dataset.lower() not in ("mnist",):
        ds_tag = f"_{dataset.lower()}"
    cfg["experiment"]["name"] = f"{rule}_S_{scaffold}_N{n_neurons}_seed{seed}{ds_tag}" \
                                + ("_wnoff" if weight_norm_off else "")
    cfg["model"]["n_neurons"] = int(n_neurons)
    if device is not None:
        cfg["experiment"]["device"] = device
    if smoke:
        cfg["training"]["n_train"] = int(cfg["smoke"]["n_train_override"])
        cfg["training"]["n_test"] = 2000
        cfg["experiment"]["name"] += "_smoke"
    if n_train_override is not None:
        cfg["training"]["n_train"] = int(n_train_override)
        # Scale test set moderately for validation smokes
        if int(n_train_override) <= 25000:
            cfg["training"]["n_test"] = 5000
    if weight_norm_off:
        cfg["weight_norm"]["enabled"] = False
    if overrides:
        _apply_overrides(cfg, overrides)
    if name_suffix:
        cfg["experiment"]["name"] += f"_{name_suffix}"

    save_root = Path(save_root or cfg["experiment"]["save_dir"])
    run_dir = save_root / cfg["experiment"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact resolved config used
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    log = setup_logger(run_dir / "log.txt", cfg["experiment"]["name"])
    log.info("=" * 70)
    log.info(f"RUN: {cfg['experiment']['name']}")
    log.info(f"smoke={smoke}, n_train={cfg['training']['n_train']}, "
             f"n_test={cfg['training']['n_test']}, seed={seed}")
    log.info(f"save_dir={run_dir}")
    log.info("=" * 70)

    set_global_seeds(int(seed))

    # Skip if already completed (spec §7.4)
    if (run_dir / "completed.flag").exists():
        log.info("completed.flag already present — skipping")
        with open(run_dir / "metrics.json") as f:
            return json.load(f)

    try:
        return train_and_eval(cfg, run_dir, log)
    except Exception as e:
        log.exception(f"FAILED: {e}")
        (run_dir / "FAILED.flag").write_text(f"{type(e).__name__}: {e}\n")
        raise


def main():
    p = argparse.ArgumentParser(description="Run a single (rule, scaffold, seed) cell.")
    p.add_argument("--rule", required=True, choices=["pair_exp", "triplet", "voltage"])
    p.add_argument("--scaffold", required=True,
                   choices=["full", "inh_only", "thr_only", "none"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_neurons", type=int, default=400)
    p.add_argument("--device", default=None, help="e.g. cuda:1")
    p.add_argument("--smoke", action="store_true",
                   help="Short run for debugging (5k training samples)")
    p.add_argument("--weight-norm-off", action="store_true",
                   help="Supplementary: disable L1 weight normalization (Q5 supp)")
    p.add_argument("--n-train", type=int, default=None,
                   help="Override training-set size (e.g. 20000 for mid-scale "
                        "validation). Takes precedence over --smoke.")
    p.add_argument("--name-suffix", default=None,
                   help="Suffix appended to the experiment name (keeps ad-hoc "
                        "runs distinct from canonical matrix cells).")
    p.add_argument("--override", action="append", default=[],
                   metavar="KEY.PATH=VALUE",
                   help="Apply ad-hoc config override after resolve(). Value "
                        "is YAML-parsed. Can repeat. Example: "
                        "--override scaffold.theta_plus=0.01 "
                        "--override scaffold.tc_theta_decay=1.0e6")
    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "fmnist", "cifar10", "nmnist",
                            "kmnist", "emnist_letters", "mnist_blend",
                            "tissuemnist", "pathmnist", "organamnist",
                            "bloodmnist"],
                   help="Dataset to train and evaluate on (default: mnist). "
                        "CIFAR-10 keeps all 3 RGB channels (3072-dim flat input). "
                        "N-MNIST sums events to a 2312-dim rate vector "
                        "(2 polarities × 34×34 spatial).")
    args = p.parse_args()
    run_cell(args.rule, args.scaffold, args.seed, args.n_neurons,
             device=args.device, smoke=args.smoke,
             weight_norm_off=args.weight_norm_off,
             n_train_override=args.n_train,
             name_suffix=args.name_suffix,
             overrides=args.override,
             dataset=args.dataset)


if __name__ == "__main__":
    main()
