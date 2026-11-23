"""For every probe-fix run: how many classes does the Diehl & Cook post-hoc readout
actually predict, how many classes get at least one labelled neuron, and what is the
majority-class share of the test set?  A readout that predicts one class is degenerate,
and its 'readout gap' is not under-reporting but failure (Section 3.3, PathMNIST).  Same rule as
scripts/score/probefix_aggregate.py::posthoc_dc."""
import json, numpy as np
from pathlib import Path
d = json.load(open("results/probefix_matrix_n5.json")); R = Path("experiments/runs")
rows = []
for r in d["runs"]:
    p = R / r["run"]
    zt, ze = np.load(p / "train_spike_counts.npz"), np.load(p / "test_spike_counts.npz")
    Xtr, ytr = zt["train_spike_counts"].astype(float), zt["train_labels"]
    Xte, yte = ze["test_spike_counts"].astype(float), ze["test_labels"]
    C = int(max(ytr.max(), yte.max())) + 1
    means = np.stack([Xtr[ytr == c].mean(0) for c in range(C)], 1)
    asg = means.argmax(1); asg[Xtr.sum(0) == 0] = -1
    v = np.zeros((len(yte), C))
    for c in range(C):
        m = asg == c
        if m.any(): v[:, c] = Xte[:, m].sum(1) / m.sum()
    pred = v.argmax(1)
    rows.append(dict(rule=r["rule"], ds=r["dataset"], seed=r["seed"], mode=r["mode"],
        posthoc=100 * (pred == yte).mean(), n_pred_cls=len(np.unique(pred)),
        n_lab_cls=len(set(asg[asg >= 0])), n_classes=C, n_labelled=int((asg >= 0).sum()),
        top_pred_share=100 * np.bincount(pred, minlength=C).max() / len(pred),
        majority=100 * np.bincount(yte, minlength=C).max() / len(yte),
        silent_test=100 * (Xte.sum(1) == 0).mean()))
json.dump(rows, open("results/posthoc_degeneracy.json", "w"), indent=1)
import collections
g = collections.defaultdict(list)
for x in rows: g[(x["rule"], x["ds"], x["mode"])].append(x)
print(f"{'rule':9s}{'dataset':15s}{'mode':8s} posthoc  classes-predicted  classes-labelled  top-pred-share  majority  silent-test%")
for k, v in sorted(g.items()):
    f = lambda key: f"{min(x[key] for x in v):.0f}-{max(x[key] for x in v):.0f}"
    flag = "  <-- DEGENERATE" if max(x["n_pred_cls"] for x in v) <= 2 or min(x["top_pred_share"] for x in v) > 80 else ""
    print(f"{k[0]:9s}{k[1]:15s}{k[2]:8s} {np.mean([x['posthoc'] for x in v]):6.1f}   {f('n_pred_cls'):>8s}/{v[0]['n_classes']:<3d}      {f('n_lab_cls'):>8s}     {f('top_pred_share'):>9s}   {v[0]['majority']:6.1f}   {f('silent_test'):>8s}{flag}")
