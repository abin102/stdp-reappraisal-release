"""Torch linear readout on saved conv features — sklearn's BLAS segfaults on the
11008-dim arrays in this (repaired) venv, even torch-free. This trains a linear
classifier with SGD instead: hinge loss (= linear SVM) and softmax (= logreg /
multinomial LR). Runs on GPU (CUDA BLAS, robust) — no sklearn.

Usage:  python scripts/score/classify_feats_torch.py <label> [device]
Out:    results/conv_svm_<label>.json  (+ stdout)
"""
import sys, json, time
import numpy as np, torch

def train_linear(Xtr, ytr, Xte, yte, loss_name, epochs=150, bs=256,
                 lr=1e-3, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    clf = torch.nn.Linear(Xtr.shape[1], int(ytr.max().item()) + 1).to(Xtr.device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    lossf = (torch.nn.CrossEntropyLoss() if loss_name == "softmax"
             else torch.nn.MultiMarginLoss())  # hinge = linear SVM
    n = len(ytr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=Xtr.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(clf(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (clf(Xte).argmax(1) == yte).float().mean().item() * 100

def main():
    label = sys.argv[1]
    device = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
    z = np.load(f"results/conv_feats_{label}.npz")
    nz = float(100 * (z["Xtr"] > 0).mean())
    Xtr = torch.tensor(z["Xtr"], dtype=torch.float32, device=device)
    ytr = torch.tensor(z["ytr"], dtype=torch.long, device=device)
    Xte = torch.tensor(z["Xte"], dtype=torch.float32, device=device)
    yte = torch.tensor(z["yte"], dtype=torch.long, device=device)
    # standardize (fit on train)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    t0 = time.time()
    svm = train_linear(Xtr, ytr, Xte, yte, "hinge")
    lr = train_linear(Xtr, ytr, Xte, yte, "softmax")
    out = {"label": label, "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "n_features": int(Xtr.shape[1]), "nonzero_pct": nz,
           "linear_svm": svm, "logreg": lr}
    json.dump(out, open(f"results/conv_svm_{label}.json", "w"), indent=2)
    print(f"[{label}] linearSVM(hinge)={svm:.2f}%  logreg(softmax)={lr:.2f}%  "
          f"nz={nz:.1f}%  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
