#!/usr/bin/env python3
"""Re-extract BOTH training- and test-image spike counts from a finished run's checkpoint,
so that runs scored with the retired voltage-STDP probe (70/30 of the first 5,000 test images)
can be scored with the paper's single linear probe (fit on 10,000 training images, scored on
the full test set) without retraining.

Procedure, mirroring run_single.py Phase 2b then Phase 3 in one process:
  1. load checkpoint.pt (weights, adaptive threshold theta, embedded config);
  2. frozen pass (train=False: no weight change) over the FIRST 10,000 images of the run's own
     presentation order (randperm with seed+1000, run_single.py), i.e. the images STDP saw first;
     for runs trained on fewer than 10,000 images this continues into images STDP never saw;
  3. frozen pass over the full test set in the fixed random order of run_single.py (seed + 3000);
     theta carries over from step 2, as in run_single.py.
Outputs (new names, existing files are never touched):
  train_spike_counts_pf.npz (train_spike_counts, train_labels)
  test_spike_counts_pf.npz  (test_spike_counts, test_labels)
Equivalence: every per-image state except theta is reset at the start of each image
(models/fc_stdp_layer.py run_sample). theta was saved after the original test pass. With the
voltage-STDP configurations (scaffold.tc_theta_decay = 1e4 ms, 500 ms per image) its history
fades within about 100 images. With pair/triplet (1e7 ms) it does not, which is the same
convention as the scaffold matrix's training-image features.
The Poisson draws are seeded (seed + 2000) for reproducibility, so they are not the original draws.

Usage (repo root): PYTHONPATH=. python scripts/extract/reextract_features.py DIR [DIR ...] --device cuda:0
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
from experiments.scripts.run_single import load_dataset, get_input_dim, get_n_classes
from models.fc_stdp_layer import FCStdpLayer

N_TRAIN_IMGS = 10_000


def frozen_pass(net, data, order, threshold):
    sc_arr = np.zeros((len(order), net.n_neurons), dtype=np.float32)
    lbl = np.zeros(len(order), dtype=np.int64)
    for i, idx in enumerate(order):
        image, label = data[idx]
        if threshold is not None:
            image = (image > threshold).float()
        sc_arr[i] = net.run_sample(image, train=False).detach().cpu().numpy()
        lbl[i] = label
    return sc_arr, lbl


def extract_one(run_dir: Path, device_str: str, force: bool) -> str:
    out_tr, out_te = run_dir / "train_spike_counts_pf.npz", run_dir / "test_spike_counts_pf.npz"
    if out_tr.exists() and out_te.exists() and not force:
        return f"SKIP (exists): {run_dir.name}"
    ckpt_path = run_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        return f"NO CKPT: {run_dir.name}"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        return f"NO CONFIG IN CKPT: {run_dir.name}"
    cfg["experiment"]["device"] = device_str
    ds = cfg.get("data", {}).get("dataset", "mnist")
    cfg.setdefault("model", {})["n_input"] = get_input_dim(ds)
    cfg.setdefault("data", {})["n_classes"] = get_n_classes(ds)
    net = FCStdpLayer(cfg, device=device)
    net.load_checkpoint(str(ckpt_path))
    train_set, test_set = load_dataset(cfg["data"]["data_dir"], ds)
    seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed + 2_000)
    g = torch.Generator().manual_seed(seed + 1_000)          # run_single.py train_order
    order = torch.randperm(len(train_set), generator=g).tolist()[:N_TRAIN_IMGS]
    thr = cfg.get("data", {}).get("threshold", None)
    thr = None if thr is None else float(thr)
    t0 = time.time()
    Xtr, ytr = frozen_pass(net, train_set, order, thr)
    # Same fixed random test order as run_single.py (seed + 3000); some test sets are class-sorted.
    te_order = torch.randperm(len(test_set), generator=torch.Generator().manual_seed(seed + 3_000)).tolist()
    Xte, yte = frozen_pass(net, test_set, te_order, thr)
    np.savez_compressed(out_tr, train_spike_counts=Xtr, train_labels=ytr)
    np.savez_compressed(out_te, test_spike_counts=Xte, test_labels=yte,
                        test_indices=np.asarray(te_order, dtype=np.int64))
    return (f"OK: {run_dir.name} ({ds}, n_train_stdp={cfg['training']['n_train']}, "
            f"{len(order)} train + {len(yte)} test imgs, {(time.time() - t0) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for d in a.run_dirs:
        try:
            print(extract_one(Path(d), a.device, a.force), flush=True)
        except Exception as e:  # one bad run must not stop the batch
            print(f"FAIL: {Path(d).name}  {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
