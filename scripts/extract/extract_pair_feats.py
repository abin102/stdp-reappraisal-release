"""Extract conv PAIR-STDP features. The surrogate-net pipeline has a different
API than voltage/triplet (kwargs constructor + single-image extract_features,
no extract_features_batch), so it needs its own extractor. Loads a checkpoint
(or RANDOM init) and saves results/conv_feats_<label>.npz for the torch weak-
readout classifier (scripts/score/classify_feats_torch.py).

Usage:
  python scripts/extract/extract_pair_feats.py <label> <ckpt|RANDOM> [device] [n_train] [seed]
"""
import sys, time, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
from conv.conv2layer_stdp_surrogate import (
    TwoLayerConvSTDP_SurrogateNet,
    _load_layer1_from_ckpt, _load_layer2_from_ckpt, load_mnist)

def main():
    label = sys.argv[1]
    ckpt = sys.argv[2]
    device = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cuda:1")
    n_train = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    n_test = 10000

    # seed controls the RANDOM-init filters when ckpt == "RANDOM"; it must be
    # varied to obtain more than one random baseline. Default 42 reproduces the
    # originally published single-seed features.
    torch.manual_seed(seed); np.random.seed(seed)
    net = TwoLayerConvSTDP_SurrogateNet(
        n_filters1=32, ks1=5, n_filters2=64, ks2=5,
        sim_time=350.0, intensity=128.0, norm=10.0,
        readout_hidden=256, readout_steps=50, readout_lr=1e-3,
        device=device, seed=seed, input_channels=1, input_h=28, input_w=28)
    if ckpt != "RANDOM":
        _load_layer1_from_ckpt(net, ckpt, device)
        _load_layer2_from_ckpt(net, ckpt, device)
        print(f"[{label}] loaded {ckpt}", flush=True)
    else:
        print(f"[{label}] RANDOM-init pair filters (no checkpoint)", flush=True)

    train_set, test_set = load_mnist(dataset="mnist")

    def extract(ds, n, tag):
        feats, labs, t0 = [], [], time.time()
        for i in range(n):
            img, lbl = ds[i]
            f = net.extract_features(img, train_stdp=False)
            feats.append(f.detach().cpu().numpy()); labs.append(int(lbl))
            if (i + 1) % 4000 == 0:
                print(f"  [{label}] {tag} {i+1}/{n} "
                      f"({(i+1)/(time.time()-t0):.0f} img/s)", flush=True)
        return np.asarray(feats, dtype=np.float32), np.asarray(labs)

    Xtr, ytr = extract(train_set, n_train, "train")
    Xte, yte = extract(test_set, n_test, "test")
    print(f"[{label}] shape {Xtr.shape}  nz={100*(Xtr>0).mean():.1f}%", flush=True)
    np.savez_compressed(f"results/conv_feats_{label}.npz",
                        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"[{label}] saved -> results/conv_feats_{label}.npz", flush=True)

if __name__ == "__main__":
    main()
