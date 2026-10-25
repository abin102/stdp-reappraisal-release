#!/usr/bin/env python3
"""Appendix J counts for the exact reimplementation of Clopath's neuron (MNIST, seeds 0-4).

Reads the runs listed in results/run_lists/jobs_port.txt (grid10k, thetatar10k, explore_k0) and
results/clopath_port_probefix_scores.json (written by scripts/score/clopath_port_probefix_score.py).
Per-network statistics come from scripts/score/clopath_port_thetatar_score.network_records:
  - a trained network "falls silent during training" if its last training image with a spike comes
    before the last training image (last_train_image_with_spike < n_train; 0 = never fired);
  - neurons still firing = neurons with at least one spike over the 10,000 test images;
  - fraction of weights at the maximum = fraction of W_final >= 3 (the weight bound).
Counts reported (text of Appendix J):
  (1) grid10k (u_ref^2 = 60 mV^2, 18 settings = 9 (g, k_inh) cells x 2 amplitudes): settings that fall
      silent in every seed, the range of the last training image with a spike over them, how many
      networks never fire, settings silent in all but one seed, and the neurons still firing in the
      trained networks that do not fall silent;
  (2) thetatar10k (u_ref^2 = 240 ... 15,360; 72 settings): settings whose total depression / total
      potentiation over training (mean over seeds) exceed those of the same setting at 60 mV^2;
      the largest fraction of weights at the maximum (mean over seeds, per setting);
  (3) single linear probe (from the probefix json): best setting at 240 mV^2, accuracy range at
      960 mV^2 and above, fits that did not converge (and how many at the largest input gain), and the
      explore_k0 (no lateral inhibition) accuracy ranges.
Run directories split for re-extraction (meta.json "split_from") are read from their parent run.
Usage (repo root): PYTHONPATH=. python scripts/score/appendix_j_counts.py [--root DIR]
  --root: directory that holds experiments/runs/clopath_port (default: the repo root).
Out: results/appendix_j_counts.json, results/appendix_j_counts.md
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from scripts.score.clopath_port_thetatar_score import load_run, network_records

JOBS = "results/run_lists/jobs_port.txt"
PROBE_JSON = "results/clopath_port_probefix_scores.json"
OUT = "results/appendix_j_counts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="directory holding experiments/runs/clopath_port")
    args = ap.parse_args()
    dirs = []
    for d in (l.strip() for l in open(JOBS) if l.strip()):
        # seed0_all_{a,b} split seed0_all for re-extraction only and carry no training statistics
        d = json.load(open(os.path.join(args.root, d, "meta.json"))).get("split_from", d)
        if d not in dirs:
            dirs.append(d)
    nets = []  # (group, record, n neurons firing on the test set)
    for d in dirs:
        group = os.path.basename(os.path.dirname(d))
        run = load_run(os.path.join(args.root, d))
        for rec, counts, _ in network_records(run, group):
            if rec["kind"] == "trained":
                nets.append((group, rec, int((counts.sum(0) > 0).sum()), counts.shape[1]))
    seeds = sorted({r["seed"] for _, r, _, _ in nets})
    by = defaultdict(list)  # (group, theta_tar, cell, amp) -> records
    for g, r, nf, nn in nets:
        by[(g, r["theta_tar"], r["cell"], r["amp"])].append((r, nf, nn))

    # (1) grid10k, Clopath's reference
    grid = {k: v for k, v in by.items() if k[0] == "grid10k"}
    all_silent = [k for k, v in grid.items() if all(r["fell_silent_during_training"] for r, _, _ in v)]
    n_silent = {k: sum(r["fell_silent_during_training"] for r, _, _ in v) for k, v in grid.items()}
    last = [r["last_train_image_with_spike"] for k in all_silent for r, _, _ in grid[k]]
    never = sum(x == 0 for x in last)
    all_but_one = [k for k, v in grid.items() if n_silent[k] == len(v) - 1]
    still = [dict(cell=k[2], amp=k[3], seed=r["seed"], neurons_firing_test=nf, n_neurons=nn)
             for k, v in grid.items() for r, nf, nn in v if not r["fell_silent_during_training"]]
    silent_per_seed = {s: sum(r["fell_silent_during_training"] for k, v in grid.items() for r, _, _ in v
                              if r["seed"] == s) for s in seeds}
    part1 = dict(n_settings=len(grid), n_seeds=len(seeds),
                 settings_silent_every_seed=len(all_silent),
                 last_train_image_with_spike_range=[int(min(last)), int(max(last))],
                 networks_never_fired=int(never), networks_in_silent_settings=len(last),
                 settings_silent_all_but_one_seed=[f"{k[2]} {k[3]}" for k in all_but_one],
                 silent_settings_per_seed=silent_per_seed,
                 networks_that_keep_firing=still)

    # (2) thetatar sweep vs the same setting at 60 mV^2 (means over seeds)
    def m(v, key):
        return float(np.mean([r[key] for r, _, _ in v]))
    ref = {(k[2], k[3]): v for k, v in grid.items()}
    sweep = {k: v for k, v in by.items() if k[0] == "thetatar10k"}
    ltd_more = sum(m(v, "ltd_total") > m(ref[(k[2], k[3])], "ltd_total") for k, v in sweep.items())
    ltp_more = sum(m(v, "ltp_total") > m(ref[(k[2], k[3])], "ltp_total") for k, v in sweep.items())
    fmax = {k: float(np.mean([r["weights_end"]["frac_wmax"] for r, _, _ in v])) for k, v in sweep.items()}
    kmax = max(fmax, key=fmax.get)
    g_of = {(k[2]): v[0][0]["g"] for k, v in by.items()}
    part2 = dict(n_settings=len(sweep), ltd_total_above_reference=int(ltd_more),
                 ltp_total_above_reference=int(ltp_more),
                 max_frac_weights_at_max=dict(value=fmax[kmax], theta_tar=kmax[1], cell=kmax[2], amp=kmax[3],
                                              g=g_of[kmax[2]]),
                 max_frac_weights_at_max_grid60=max(float(np.mean([r["weights_end"]["frac_wmax"] for r, _, _ in v]))
                                                    for v in grid.values()))

    # (3) single linear probe
    P = [n for n in json.load(open(PROBE_JSON))["networks"]]
    cell = defaultdict(list)
    for n in P:
        cell[(n["group"], n.get("theta_tar") if n["group"] == "thetatar10k" else None, n["cell"], n.get("amp"),
              n["kind"])].append(n)
    agg = {k: (float(np.mean([n["new"] for n in v])), float(np.std([n["new"] for n in v], ddof=1)))
           for k, v in cell.items()}
    t240 = {k: a for k, a in agg.items() if k[0] == "thetatar10k" and k[1] == 240.0}
    best = max(t240, key=lambda k: t240[k][0])
    rnd = agg[("grid10k", None, best[2], None, "random")]
    hi = [a[0] for k, a in agg.items() if k[0] == "thetatar10k" and k[1] >= 960.0]
    tt_nets = [n for n in P if n["group"] == "thetatar10k"]
    nc = [n for n in tt_nets if not n["converged"]]
    gmax = max(n["g"] for n in P)
    ex_tr = [a[0] for k, a in agg.items() if k[0] == "explore_k0" and k[4] == "trained"]
    ex_rd = [a[0] for k, a in agg.items() if k[0] == "explore_k0" and k[4] == "random"]
    part3 = dict(best_240=dict(cell=best[2], amp=best[3], trained=t240[best], stdp_disabled=rnd),
                 accuracy_range_960_and_above=[min(hi), max(hi)],
                 thetatar_networks=len(tt_nets), not_converged=len(nc),
                 not_converged_at_largest_gain=sum(n["g"] == gmax for n in nc),
                 explore_k0_trained_range=[min(ex_tr), max(ex_tr)],
                 explore_k0_stdp_disabled_range=[min(ex_rd), max(ex_rd)])

    out = dict(seeds=seeds, grid60=part1, thetatar=part2, probe=part3)
    json.dump(out, open(OUT + ".json", "w"), indent=1, default=float)
    p1, p2, p3 = part1, part2, part3
    L = ["# Appendix J counts (exact reimplementation of Clopath's neuron, MNIST)", "",
         "Generated by `scripts/score/appendix_j_counts.py` (definitions in its docstring). "
         f"Seeds {seeds[0]}-{seeds[-1]}.", "",
         "| quantity | value |", "|---|---|",
         f"| settings (u_ref^2 = 60) that fall silent during training in every seed | "
         f"{p1['settings_silent_every_seed']} of {p1['n_settings']} |",
         f"| last training image with a spike, over those networks | {p1['last_train_image_with_spike_range'][0]} "
         f"to {p1['last_train_image_with_spike_range'][1]:,} |",
         f"| of those networks, never fired | {p1['networks_never_fired']} of {p1['networks_in_silent_settings']} |",
         f"| settings silent in all but one seed | {', '.join(p1['settings_silent_all_but_one_seed']) or 'none'} |",
         "| silent settings per seed | " + ", ".join(f"seed {s}: {n}" for s, n in p1['silent_settings_per_seed'].items()) + " |"]
    for w in p1["networks_that_keep_firing"]:
        L.append(f"| keeps firing: {w['cell']} {w['amp']} seed {w['seed']} | {w['neurons_firing_test']} of "
                 f"{w['n_neurons']} neurons fire on the test set |")
    L += [f"| raised reference: settings with total depression above the 60 mV^2 setting | "
          f"{p2['ltd_total_above_reference']} of {p2['n_settings']} |",
          f"| raised reference: settings with total potentiation above the 60 mV^2 setting | "
          f"{p2['ltp_total_above_reference']} of {p2['n_settings']} |",
          f"| largest fraction of weights at the maximum (mean over seeds) | "
          f"{100 * p2['max_frac_weights_at_max']['value']:.1f}% (u_ref^2 = {p2['max_frac_weights_at_max']['theta_tar']:g}, "
          f"{p2['max_frac_weights_at_max']['cell']} {p2['max_frac_weights_at_max']['amp']}, g = {p2['max_frac_weights_at_max']['g']}) |",
          f"| best setting at 240 mV^2 (single probe) | {p3['best_240']['cell']} {p3['best_240']['amp']}: "
          f"{p3['best_240']['trained'][0]:.1f} ± {p3['best_240']['trained'][1]:.1f} vs STDP disabled "
          f"{p3['best_240']['stdp_disabled'][0]:.1f} ± {p3['best_240']['stdp_disabled'][1]:.1f} |",
          f"| accuracy range at 960 mV^2 and above (setting means) | {p3['accuracy_range_960_and_above'][0]:.1f} to "
          f"{p3['accuracy_range_960_and_above'][1]:.1f} |",
          f"| probe fits not converged (raised reference) | {p3['not_converged']} of {p3['thetatar_networks']}, "
          f"{p3['not_converged_at_largest_gain']} at the largest input gain |",
          f"| no lateral inhibition (explore_k0): trained / STDP disabled | {p3['explore_k0_trained_range'][0]:.1f}-"
          f"{p3['explore_k0_trained_range'][1]:.1f} / {p3['explore_k0_stdp_disabled_range'][0]:.1f}-"
          f"{p3['explore_k0_stdp_disabled_range'][1]:.1f} |", "",
          "A silent network scores the majority-class share, 11.35% (1,135 of 10,000 test images are ones); "
          "the paper writes it as 11.4."]
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
