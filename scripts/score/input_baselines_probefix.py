#!/usr/bin/env python3
"""Input-level baselines for the rebuilt Table 2, under the paper's single linear probe (CPU only).

Same protocol as scripts/score/probefix_aggregate.py, applied to the network's INPUT instead of its output:
  * training images: the first 10,000 of run_single.py's presentation order (randperm, seed + 1000),
    i.e. the images the probe-fix networks of that seed were trained on;
  * test images: the FULL test set;
  * probe: logistic regression, L2, C = 1, raw values, max_iter=100000; seeds 0-4.
Two inputs per image:
  * encoded spike input: the spike counts of the input layer, drawn exactly as
    models/fc_stdp_layer.py run_sample draws them (per 1-ms step, Bernoulli with probability
    clamp(pixel * 63.75 / 1000, 0, 1), over the 350 steps with input) -> Binomial(350, p);
    full input dimension (784, or 2,312 for N-MNIST);
  * raw pixels: the same image vectors, unencoded (kept in the JSON; not planned for Table 2).
Replaces scripts/input_baselines_trainfit.py, which capped the test set at 10,000 images,
truncated N-MNIST to 784 inputs, used one seed and max_iter 2000.

Output: results/input_baselines_probefix.json
Usage (repo root): PYTHONPATH=. python scripts/score/input_baselines_probefix.py
"""
import json, warnings
import numpy as np
import torch
from joblib import Parallel, delayed
from scripts.score._blas import fixed_threads

warnings.filterwarnings("ignore")
INTENSITY, DT, SIM = 63.75, 1.0, 350          # data.intensity, data.dt, data.sim_time (all datasets)
N_STEPS = int(SIM / DT)
N_TRAIN, SEEDS, MAX_ITER = 10000, range(5), 100000
DATASETS = ["mnist", "nmnist", "emnist_letters", "fmnist",
            "organamnist", "bloodmnist", "tissuemnist", "pathmnist"]


def one_dataset(ds):
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from experiments.scripts.run_single import load_dataset
    train, test = load_dataset("./data", ds)

    def vecs(dset, idx):
        X = np.stack([np.asarray(dset[i][0], dtype=np.float32).ravel() for i in idx])
        y = np.array([int(np.asarray(dset[i][1]).ravel()[0]) for i in idx])
        return X, y

    Xte_raw, yte = vecs(test, range(len(test)))
    p_te = np.clip(Xte_raw * INTENSITY / 1000.0 * DT, 0, 1)
    res = dict(dim=int(Xte_raw.shape[1]), n_test=int(len(yte)), seeds={})
    for s in SEEDS:
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(s + 1000)).tolist()[:N_TRAIN]
        Xtr_raw, ytr = vecs(train, order)
        rng = np.random.RandomState(s)
        Xtr_enc = rng.binomial(N_STEPS, np.clip(Xtr_raw * INTENSITY / 1000.0 * DT, 0, 1)).astype(np.float32)
        Xte_enc = rng.binomial(N_STEPS, p_te).astype(np.float32)
        fit = lambda A, B, C, D: 100.0 * LogisticRegression(max_iter=MAX_ITER, C=1.0).fit(A, B).score(C, D)
        res["seeds"][s] = dict(encoded=fit(Xtr_enc, ytr, Xte_enc, yte), raw=fit(Xtr_raw, ytr, Xte_raw, yte),
                               n_train=len(order))
        print(f"{ds} seed {s}: encoded {res['seeds'][s]['encoded']:.1f}  raw {res['seeds'][s]['raw']:.1f}", flush=True)
    for k in ("encoded", "raw"):
        v = np.array([res["seeds"][s][k] for s in SEEDS])
        res[k] = (float(v.mean()), float(v.std(ddof=1)))
    return ds, res


def main():
    # One BLAS thread per worker, as for every probe fit (see scripts/score/_blas.py).
    out = dict(Parallel(n_jobs=8)(delayed(fixed_threads(one_dataset, 1))(ds) for ds in DATASETS))
    out["_protocol"] = __doc__.split("Output:")[0].strip()
    json.dump(out, open("results/input_baselines_probefix.json", "w"), indent=1)
    print("\ndataset | dim | test images | encoded input | raw pixels")
    for ds in DATASETS:
        r = out[ds]
        print(f"{ds} | {r['dim']} | {r['n_test']} | {r['encoded'][0]:.1f} ± {r['encoded'][1]:.1f} | "
              f"{r['raw'][0]:.1f} ± {r['raw'][1]:.1f}")


if __name__ == "__main__":
    main()
