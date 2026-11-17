"""SoftHebb with the matched random baseline (Section 3.9, Table 5, Methods 2.13, Section S18): does SoftHebb training
add accuracy over the same network with untrained weights, on medical images?
Release copy of experiments/softhebb_pilot/run_softhebb.py; only the paths, the check of the third-party demo and the
messages for missing data differ (results/softhebb/README.md).

Network, plasticity rule, learning rates and readout are the released single-file demo of Journe et al. (ICLR 2023),
`third_party/SoftHebb/demo.py` (three SoftHebb convolution layers, 96-384-1536 channels, and a linear classifier),
imported unchanged. The SoftHebb repository has no licence and is not redistributed here: clone it into
third_party/SoftHebb (or set SOFTHEBB_DIR), see results/softhebb/README.md. This script only changes the data, the
device, the seed handling, and adds the control.

  trained: one unsupervised pass over the training images (batch 10), as in the demo, then the convolutions are frozen.
  random:  the same pass with the weight updates switched off. The weights stay at their initial values; the
           BatchNorm running statistics are still estimated on the same images, as in the trained network.
  pixels:  no network; the same readout procedure on the 3 x 32 x 32 input, standardized per channel with the training
           images' mean and standard deviation (stage 2, README.md). The linear layer has PyTorch's default init.
  Readout (all): the demo's linear classifier on the frozen features, 50 epochs of Adam at 0.001 with the demo's step
           schedule and dropout 0.5, fitted on the same training images; scored on the full official test split at
           the last epoch (no epoch is chosen on test data).

Data: MedMNIST at 28 px (official train and test splits; the validation split is not used), zero-padded to 32 x 32 so
the demo's network is used as it is; grayscale sets are repeated over three channels. CIFAR-10 is the reproduction
check (the paper reports 80.3% for this network). Pixels in [0, 1], as in the demo.
--n_train N takes the first N images of a seeded permutation of the training split for both phases (10,000 matches
the STDP paper's matrix); 0 uses the whole split.

  PYTHONPATH=. python scripts/train/softhebb_run.py --dataset bloodmnist --mode trained --seed 0 --n_train 10000 --device cuda:0
Writes <out>/<dataset>__<mode>__n<n_train>__seed<seed>/result.json and test_logits.npz (default <out>:
experiments/runs/softhebb). Refuses to overwrite. The shipped results are in results/softhebb/runs/.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch import nn, optim

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                       # repository root
SOFTHEBB_DIR = os.environ.get("SOFTHEBB_DIR", os.path.join(REPO, "third_party", "SoftHebb"))
DEMO_SHA256 = "0e44b4e6e830670f2bcd57b8fe845e29472c8c77d108b3db3a2dcb12a5d3630b"   # demo.py at commit f425230


def _check_demo():
    import hashlib
    demo = os.path.join(SOFTHEBB_DIR, "demo.py")
    if not os.path.exists(demo):
        raise SystemExit(f"SoftHebb's demo.py not found in {SOFTHEBB_DIR}. It is not redistributed (no licence). Get it with\n"
                         "  git clone https://github.com/NeuromorphicComputing/SoftHebb.git third_party/SoftHebb\n"
                         "  git -C third_party/SoftHebb checkout f4252307633ba38a26ab52504f5fd40352f5a6fd\n"
                         "or set SOFTHEBB_DIR to a folder holding that commit (results/softhebb/README.md).")
    if hashlib.sha256(open(demo, "rb").read()).hexdigest() != DEMO_SHA256:
        raise SystemExit(f"{demo} differs from the file the paper's runs used (commit f425230, SHA-256 {DEMO_SHA256}); "
                         "check out that commit.")


_check_demo()
sys.path.insert(0, SOFTHEBB_DIR)
import demo as SH                                                   # noqa: E402  (its training loop is under __main__)

MED = {"organamnist": 11, "bloodmnist": 8, "pathmnist": 9, "tissuemnist": 8}
UNSUP_LR = (-0.08, -0.005, -0.01)                                   # the demo's values; negative: its SGD descends
SUP_EPOCHS = 50


def load(dataset):
    """-> train X, y, test X, y as uint8 (N, H, W, C) and int64."""
    if dataset in MED:
        sys.path.insert(0, REPO)
        from data.medmnist_loader import _ensure_npz                # downloads data/<dataset>.npz (28 px) if absent
        d = np.load(_ensure_npz(dataset, os.path.join(REPO, "data")))
        f = lambda a: a if a.ndim == 4 else np.repeat(a[..., None], 3, axis=3)
        return f(d["train_images"]), d["train_labels"].ravel().astype(np.int64), \
            f(d["test_images"]), d["test_labels"].ravel().astype(np.int64), MED[dataset]
    if dataset == "cifar10":
        root = os.path.join(REPO, "data", "cifar-10-batches-py")
        if not os.path.isdir(root):
            raise SystemExit(f"{root} not found. Download CIFAR-10 first:\n"
                             "  python -c \"import torchvision; torchvision.datasets.CIFAR10('data', download=True)\"")
        rd = lambda n: pickle.load(open(os.path.join(root, n), "rb"), encoding="bytes")
        tr = [rd(f"data_batch_{i}") for i in range(1, 6)]
        te = rd("test_batch")
        img = lambda b: b[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.concatenate([img(b) for b in tr]), np.concatenate([np.array(b[b"labels"]) for b in tr]).astype(np.int64), \
            img(te), np.array(te[b"labels"]).astype(np.int64), 10
    raise SystemExit(f"unknown dataset {dataset}")


def to_tensor(x, dev):
    t = torch.tensor(x, dtype=torch.float, device=dev).div_(255).movedim(-1, 1)
    pad = (32 - t.shape[-1]) // 2
    return nn.functional.pad(t, (pad, pad, pad, pad)) if pad else t


def batches(n, size, gen, shuffle):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", required=True, choices=["trained", "random", "pixels"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "runs", "softhebb"))
    ap.add_argument("--sup_epochs", type=int, default=SUP_EPOCHS, help="50 is the protocol; lower only for a smoke test")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    rd = os.path.join(a.out, f"{a.dataset}__{a.mode}__n{a.n_train}__seed{a.seed}")
    if os.path.exists(os.path.join(rd, "result.json")):
        raise SystemExit(f"{rd} already has a result; not overwriting")
    os.makedirs(rd, exist_ok=True)

    Xtr, ytr, Xte, yte, n_cls = load(a.dataset)
    if a.n_train:
        keep = np.random.RandomState(a.seed).permutation(len(ytr))[:a.n_train]
        Xtr, ytr = Xtr[keep], ytr[keep]
    xtr, xte = to_tensor(Xtr, dev), to_tensor(Xte, dev)
    ytr_t = torch.tensor(ytr, device=dev)
    if a.mode == "pixels":
        return run_pixels(a, xtr, ytr_t, xte, yte, n_cls, dev, rd)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    model = SH.DeepSoftHebb()
    if n_cls != 10:                                                 # the demo's classifier, with this dataset's classes
        model.classifier = nn.Linear(24576, n_cls)
        model.classifier.weight.data = 0.11048543456039805 * torch.rand(n_cls, 24576)
    model.to(dev)
    convs = [model.conv1, model.conv2, model.conv3]
    w0 = [c.weight.detach().clone() for c in convs]

    # --- unsupervised pass (the demo's loop). "random": the same pass without the optimizer step ---
    unsup = SH.TensorLRSGD([{"params": c.parameters(), "lr": lr} for c, lr in zip(convs, UNSUP_LR)], lr=0)

    def set_lr():
        """The demo's WeightNormDependentLR (power_lr = 0.5), written out: each neuron's learning rate is the layer's
        initial rate times |weight norm - 1| ** 0.5, recomputed after every update. The demo's class cannot be used
        with this PyTorch: it passes an argument the scheduler base class no longer takes, and the base class now
        refuses a per-neuron (tensor) rate when stepping."""
        for g, lr0, c in zip(unsup.param_groups, UNSUP_LR, convs):
            w = c.weight.detach()
            g["lr"] = lr0 * ((torch.abs(torch.linalg.norm(w.view(w.shape[0], -1), dim=1, ord=2) - 1) + 1e-10) ** 0.5)[:, None, None, None]

    set_lr()
    t0 = time.time()
    model.train()
    for idx in batches(len(ytr), 10, gen, True):
        unsup.zero_grad()
        with torch.no_grad():
            model(xtr[idx.to(dev)])
        if a.mode == "trained":
            unsup.step()
            set_lr()
    unsup.zero_grad()
    moved = [float((c.weight.detach() - w).abs().max()) for c, w in zip(convs, w0)]
    if a.mode == "random" and max(moved) != 0.0:
        raise SystemExit(f"random control: weights moved {moved}")
    if a.mode == "trained" and min(moved) == 0.0:
        raise SystemExit(f"trained: a layer did not move {moved}")
    norms = [float(torch.linalg.norm(c.weight.detach().flatten(1), dim=1).mean()) for c in convs]
    t_unsup = time.time() - t0

    # --- linear classifier on the frozen features (the demo's loop) ---
    for c in convs:
        c.requires_grad = False
        c.eval()
    for b in (model.bn1, model.bn2, model.bn3):
        b.eval()
    sup = optim.Adam(model.classifier.parameters(), lr=0.001)
    sup_sched = SH.CustomStepLR(sup, nb_epochs=a.sup_epochs)
    crit = nn.CrossEntropyLoss()
    curve = []
    for ep in range(a.sup_epochs):
        model.classifier.train(); model.dropout.train()
        correct = 0
        for idx in batches(len(ytr), 64, gen, True):
            idx = idx.to(dev)
            sup.zero_grad()
            out = model(xtr[idx])
            crit(out, ytr_t[idx]).backward()
            sup.step()
            correct += int((out.argmax(1) == ytr_t[idx]).sum())
        sup_sched.step()
        if ep % 10 == 0 or ep == a.sup_epochs - 1:
            model.eval()
            with torch.no_grad():
                z = torch.cat([model(xte[i:i + 1000]) for i in range(0, len(yte), 1000)]).cpu().numpy()
            acc = float((z.argmax(1) == yte).mean() * 100)
            curve.append(dict(epoch=ep + 1, train_acc=100 * correct / len(ytr), test_acc=acc))
            print(f"[{a.dataset} {a.mode} n{a.n_train} s{a.seed}] epoch {ep + 1}: train {curve[-1]['train_acc']:.2f}% "
                  f"test {acc:.2f}%", flush=True)
    if not np.isfinite(z).all():
        raise SystemExit("non-finite test logits")
    res = dict(dataset=a.dataset, mode=a.mode, seed=a.seed, n_train=len(ytr), n_test=len(yte), n_classes=n_cls,
               test_acc_last=curve[-1]["test_acc"], train_acc_last=curve[-1]["train_acc"], curve=curve,
               weight_change_max=moved, weight_norm_mean=norms, sup_epochs=a.sup_epochs, device=str(dev),
               seconds_unsup=t_unsup, seconds_total=time.time() - t0, torch=torch.__version__)
    np.savez_compressed(os.path.join(rd, "test_logits.npz"), logits=z.astype(np.float32), targets=yte)
    with open(os.path.join(rd, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}), flush=True)


def run_pixels(a, xtr, ytr_t, xte, yte, n_cls, dev, rd):
    """Pixel readout (stage 2): the readout loop of main() with dropout 0.5 and one linear layer on the standardized
    input instead of the frozen network's features. Same seed handling, batches, optimizer, schedule and scoring."""
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    mu, sd = xtr.mean(dim=(0, 2, 3), keepdim=True), xtr.std(dim=(0, 2, 3), keepdim=True)
    ftr, fte = ((xtr - mu) / sd).flatten(1), ((xte - mu) / sd).flatten(1)
    clf = nn.Sequential(nn.Dropout(0.5), nn.Linear(ftr.shape[1], n_cls)).to(dev)
    sup = optim.Adam(clf.parameters(), lr=0.001)
    sup_sched = SH.CustomStepLR(sup, nb_epochs=a.sup_epochs)
    crit = nn.CrossEntropyLoss()
    curve = []
    for ep in range(a.sup_epochs):
        clf.train()
        correct = 0
        for idx in batches(len(ytr_t), 64, gen, True):
            idx = idx.to(dev)
            sup.zero_grad()
            out = clf(ftr[idx])
            crit(out, ytr_t[idx]).backward()
            sup.step()
            correct += int((out.argmax(1) == ytr_t[idx]).sum())
        sup_sched.step()
        if ep % 10 == 0 or ep == a.sup_epochs - 1:
            clf.eval()
            with torch.no_grad():
                z = torch.cat([clf(fte[i:i + 1000]) for i in range(0, len(yte), 1000)]).cpu().numpy()
            acc = float((z.argmax(1) == yte).mean() * 100)
            curve.append(dict(epoch=ep + 1, train_acc=100 * correct / len(ytr_t), test_acc=acc))
            print(f"[{a.dataset} {a.mode} n{a.n_train} s{a.seed}] epoch {ep + 1}: train {curve[-1]['train_acc']:.2f}% "
                  f"test {acc:.2f}%", flush=True)
    if not np.isfinite(z).all():
        raise SystemExit("non-finite test logits")
    res = dict(dataset=a.dataset, mode=a.mode, seed=a.seed, n_train=len(ytr_t), n_test=len(yte), n_classes=n_cls,
               test_acc_last=curve[-1]["test_acc"], train_acc_last=curve[-1]["train_acc"], curve=curve,
               n_features=int(ftr.shape[1]), sup_epochs=a.sup_epochs, device=str(dev),
               seconds_total=time.time() - t0, torch=torch.__version__)
    np.savez_compressed(os.path.join(rd, "test_logits.npz"), logits=z.astype(np.float32), targets=yte)
    with open(os.path.join(rd, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
