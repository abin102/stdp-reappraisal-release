#!/usr/bin/env python3
"""Extract TRAIN-set spike-count features for a completed run, from its saved
checkpoint (no STDP retraining).

Background: the linear-probe readout was originally fit on an internal split of
the *test* spike counts (see aggregate_results.py / aggregate_medical_matrix.py).
To move to the standard convention -- fit the probe on train features, evaluate
on the full held-out test set -- we need per-image TRAIN features, which the
original runs never saved (only ``class_rates`` aggregates, accumulated while
weights were still changing). This script produces ``train_spike_counts.npz``
for a run by:

  1. loading the run's frozen weights from ``checkpoint.pt`` (config embedded),
  2. rebuilding the exact ``train_order`` (randperm, seed+1000) run_single used,
  3. forward-passing the first ``--n_subset`` train images with train=False,
     applying the same optional input threshold,
  4. saving ``train_spike_counts.npz`` (train_spike_counts, train_labels).

Idempotent: skips if ``train_spike_counts.npz`` already exists (unless --force).
Mirrors the Phase-3 extraction loop in run_single.py:549-557 exactly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# reuse the exact dataset loaders / dim helpers the training pipeline used
from experiments.scripts.run_single import (
    load_dataset, get_input_dim, get_n_classes,
)
from models.fc_stdp_layer import FCStdpLayer


def extract_one(run_dir: Path, n_subset: int, device_str: str,
                force: bool = False) -> str:
    out = run_dir / "train_spike_counts.npz"
    if out.exists() and not force:
        return f"SKIP (exists): {run_dir.name}"

    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        return f"NO CKPT: {run_dir.name}"

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Config is embedded in the checkpoint (run_single saved extra={"config": cfg}
    # AFTER setting model.n_input / data.n_classes), so it rebuilds the net exactly.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        return f"NO CONFIG IN CKPT: {run_dir.name}"
    cfg["experiment"]["device"] = device_str  # honor the requested GPU

    dataset_name = cfg.get("data", {}).get("dataset", "mnist")
    cfg.setdefault("model", {})["n_input"] = get_input_dim(dataset_name)
    cfg.setdefault("data", {})["n_classes"] = get_n_classes(dataset_name)

    net = FCStdpLayer(cfg, device=device)
    net.load_checkpoint(str(ckpt_path))  # loads w, theta, assignments, rates

    train_set, _ = load_dataset(cfg["data"]["data_dir"], dataset_name)

    # Reproduce run_single.py:381-382 train_order exactly.
    seed_int = int(cfg["experiment"]["seed"])
    n_train = int(cfg["training"]["n_train"])
    g = torch.Generator().manual_seed(seed_int + 1_000)
    train_order = torch.randperm(len(train_set), generator=g).tolist()[: n_train]
    train_order = train_order[: n_subset]  # first n_subset of the SAME order STDP saw

    threshold = cfg.get("data", {}).get("threshold", None)
    if threshold is not None:
        threshold = float(threshold)

    t0 = time.time()
    sc_arr = np.zeros((len(train_order), net.n_neurons), dtype=np.float32)
    lbl_arr = np.zeros(len(train_order), dtype=np.int64)
    for i, idx in enumerate(train_order):
        image, label = train_set[idx]
        if threshold is not None:
            image = (image > threshold).float()
        sc = net.run_sample(image, train=False)   # forward pass, STDP OFF
        sc_arr[i] = sc.detach().cpu().numpy()
        lbl_arr[i] = label

    np.savez_compressed(out,
                        train_spike_counts=sc_arr,
                        train_labels=lbl_arr)
    dt = (time.time() - t0) / 60.0
    return (f"OK: {run_dir.name}  ({len(train_order)} imgs, "
            f"{dataset_name}, {dt:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="run directories to extract")
    ap.add_argument("--n_subset", type=int, default=10000,
                    help="number of train images to extract (default 10000)")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for d in args.run_dirs:
        rd = Path(d)
        try:
            print(extract_one(rd, args.n_subset, args.device, args.force),
                  flush=True)
        except Exception as e:  # never let one bad run kill the batch
            print(f"FAIL: {rd.name}  {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
