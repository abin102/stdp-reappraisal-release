"""Weak-readout test on conv features: linear SVM + logreg on the 11008-dim
pooled conv spike-count features, for a given checkpoint (trained voltage or
frozen/random conv). The conv pipeline's surrogate readout saturates (~97% for
both trained and random); a LINEAR readout exposes whether the TRAINED voltage
features carry more linearly-separable class structure than RANDOM conv features.

Usage:
  python scripts/extract/conv_svm_readout.py --ckpt <path> --label trained --device cuda:0
"""
import argparse, json, time, warnings
import numpy as np, torch, yaml
warnings.filterwarnings("ignore")

import sys
from conv.conv2layer_voltage_stdp import TwoLayerConvVoltageSTDP
from torchvision import datasets, transforms

# rule -> (net class, default config). Only the voltage conv network is used in the paper.
NET_MAP = {
    "voltage": (TwoLayerConvVoltageSTDP, "conv/configs/conv_voltage_config.yaml"),
}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint to load; omit with --random")
    ap.add_argument("--label", required=True)
    ap.add_argument("--net", default="voltage", choices=list(NET_MAP))
    ap.add_argument("--random", action="store_true",
                    help="skip checkpoint load -> random-init filters (within-rule baseline)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42, help="random-init filter seed (for n>1 baselines)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    NetClass, default_cfg = NET_MAP[args.net]
    cfg = yaml.safe_load(open(args.config or default_cfg))
    cfg.setdefault("experiment", {})["device"] = str(device)

    torch.manual_seed(args.seed)  # reproducible random-init filters (vary for n>1)
    net = NetClass(cfg, device)
    if args.random:
        print(f"[{args.label}] RANDOM-init {args.net} filters (no checkpoint)")
    else:
        net.load_checkpoint(args.ckpt)
        print(f"[{args.label}] loaded {args.ckpt}")

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Lambda(lambda x: x.view(-1))])
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)

    def extract(ds, n):
        feats, labs = [], []
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, n, args.batch):
                e = min(s + args.batch, n)
                imgs = torch.stack([ds[i][0] for i in range(s, e)]).to(device)
                feats.append(net.extract_features_batch(imgs).cpu().numpy())
                labs.extend(ds[i][1] for i in range(s, e))
                if e % 5000 < args.batch:
                    print(f"    [{args.label}] {e}/{n} ({e/(time.time()-t0):.0f} img/s)")
        return np.concatenate(feats), np.array(labs)

    print(f"[{args.label}] extracting features (train={args.n_train}, test={args.n_test})...")
    Xtr, ytr = extract(train, args.n_train)
    Xte, yte = extract(test, args.n_test)
    print(f"[{args.label}] feature shape {Xtr.shape}; sparsity nonzero%={100*(Xtr>0).mean():.1f}")

    # Save features; classify in a SEPARATE torch-free process
    # (scripts/classify_feats.py). Running sklearn's BLAS inside this CUDA/torch
    # process segfaults (OpenMP/BLAS thread conflict) — hence the two-step split.
    np.savez_compressed(f"results/conv_feats_{args.label}.npz",
                        Xtr=Xtr.astype(np.float32), ytr=ytr,
                        Xte=Xte.astype(np.float32), yte=yte)
    print(f"[{args.label}] saved -> results/conv_feats_{args.label}.npz  "
          f"(classify: python scripts/classify_feats.py {args.label})")

if __name__ == "__main__":
    main()
