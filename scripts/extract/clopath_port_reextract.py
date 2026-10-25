#!/usr/bin/env python3
"""Re-extract training- and test-image spike counts for the Clopath-neuron port runs
(scripts/train/clopath_port_mnist.py train), so they can be scored with the paper's single linear probe
(fit on 10,000 training images, scored on the full test set) without retraining.

For one run directory (meta.json, weights.npz):
  * trained networks: W_final from weights.npz; STDP-off networks: init_weights(seed), exactly as
    in training; all built with plastic=False, in the order of meta["specs"];
  * frozen pass over the first 10,000 images of the run's presentation order (randperm with
    seed + 1000, as in training), then over the full test set; the simulation runs on across
    images, as in the original test pass, starting from the initial state (the end-of-training
    state was not saved; plasticity is off, so the homeostatic average has no effect);
  * spike counts over the whole 500-ms slot per image, as in the original test pass.
Outputs (existing files are never touched):
  train_spike_counts_pf.npz  counts (B, 10000, 400) uint16, labels, specs
  test_spike_counts_pf.npz   counts (B, n_test, 400) uint16, labels, specs
The Poisson draws are seeded with seed + 2000, so they are not the original draws.
Runs step by step (no CUDA graph) unless --graph is given: on 2026-10-20 the graph replay gave the
same counts for every image (explore_k0) or failed (grid10k, thetatar10k). A pass whose counts are
nonzero but identical on every image is reported as FAIL and not saved.

Usage (repo root): PYTHONPATH=. python scripts/extract/clopath_port_reextract.py DIR [DIR ...] --device cuda:0
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

from scripts.train import clopath_port_mnist as M  # noqa: E402
cp = M.cp
N_TRAIN_IMGS = 10_000


def one(d: Path, device: str, force: bool, use_graph: bool = False) -> str:
    out_tr, out_te = d / "train_spike_counts_pf.npz", d / "test_spike_counts_pf.npz"
    if out_tr.exists() and out_te.exists() and not force:
        return f"SKIP (exists): {d}"
    meta = json.load(open(d / "meta.json"))
    specs, seed = meta["specs"], int(meta["seed"])
    if int(meta.get("n_passes", 1)) != 1:
        return f"FAIL: {d} has n_passes={meta['n_passes']} (not handled)"
    dev = torch.device(device)
    tr = [s for s in specs if s["kind"] == "trained"]
    rd = [s for s in specs if s["kind"] == "random"]
    assert specs == tr + rd, "specs are not ordered trained-then-random"
    common = dict(plastic=False, dtype=torch.float64, device=dev, u_floor=cp.U_FLOOR_INH)
    nets = []
    if tr:
        W = torch.from_numpy(np.load(d / "weights.npz")["W_final"]).to(torch.float64)
        assert W.shape[0] == len(tr), f"{W.shape[0]} weight sets for {len(tr)} trained specs"
        nets.append(cp.ClopathNet(W, g=[s["g"] for s in tr], k_inh=[s["k_inh"] for s in tr], **common))
    if rd:
        W0 = M.init_weights(seed)
        nets.append(cp.ClopathNet(W0.unsqueeze(0).expand(len(rd), -1, -1),
                                  g=[s["g"] for s in rd], k_inh=[s["k_inh"] for s in rd], **common))
    net = nets[0] if len(nets) == 1 else cp.ClopathNet.concat(nets, plastic=False)
    Xtr, ytr, Xte, yte = M.load_data(meta.get("dataset", "mnist"))
    order = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(seed + 1000)).tolist()[:N_TRAIN_IMGS]
    Xd, Xted = Xtr.to(dev), Xte.to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 2000)
    runner = M.ImageRunner([net], 784, dev, use_graph=use_graph)
    t0 = time.time()

    def frozen(X, idx):
        c = np.zeros((len(specs), len(idx), 400), np.uint16)
        for i, j in enumerate(idx):
            runner.run(M.encode(X[j], gen))
            c[:, i] = runner.counts[0].cpu().numpy().astype(np.uint16)
        return c

    ctr = frozen(Xd, order)
    cte = frozen(Xted, range(len(Xte)))
    for name, c in (("train", ctr), ("test", cte)):
        tot = c.sum(axis=2, dtype=np.int64)                  # (B, n_images) spikes per image
        if tot.any() and not (tot != tot[:, :1]).any():
            return f"FAIL: {d} {name} counts are identical on every image (nothing saved)"
    sp = json.dumps(specs)
    np.savez_compressed(out_tr, counts=ctr, labels=ytr[order].numpy(), specs=sp)
    np.savez_compressed(out_te, counts=cte, labels=yte.numpy(), specs=sp)
    return f"OK: {d} ({len(specs)} networks, {len(order)} train + {len(Xte)} test, {(time.time() - t0) / 60:.1f} min)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--graph", action="store_true", help="CUDA-graph replay (gave wrong counts on 2026-10-20)")
    a = ap.parse_args()
    for d in a.run_dirs:
        try:
            print(one(Path(d), a.device, a.force, a.graph), flush=True)
        except Exception as e:
            print(f"FAIL: {d}  {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
