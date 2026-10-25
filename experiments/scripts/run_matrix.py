"""Run the full factorial matrix of scaffolding-ablation experiments.

The 60-run main matrix:  3 rules × 4 scaffolds × 5 seeds × N=400
Optional scaling:        pair_exp × {S_full, S_none} × 2 seeds × N=1600  (Q10)
Supplementary:           {triplet × S_full, voltage × S_full} × 1 seed × weight-norm=OFF  (Q5)

Respects per-cell completed.flag so the matrix is RESUMABLE: a crash, kill,
or OOM can be restarted and only un-finished cells will execute.

Master log lines (per spec §7.5):
    {timestamp}\t{run_name}\t{status}\t{accuracy}

Usage:
    # Smoke test (12 cells × 1 seed × 5k samples)
    python -m experiments.scripts.run_matrix --smoke

    # Main matrix (60 runs)
    python -m experiments.scripts.run_matrix --main

    # Supplementary weight-norm=OFF runs
    python -m experiments.scripts.run_matrix --supp-wnoff

    # Everything
    python -m experiments.scripts.run_matrix --smoke --main --supp-wnoff
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

from experiments.scripts.run_single import run_cell
from experiments.scripts.config import RULES, SCAFFOLDS


MASTER_LOG = Path("experiments/runs/master_log.tsv")


def _log_master(run_name: str, status: str, accuracy: float | None,
                elapsed_min: float | None):
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    acc_s = f"{accuracy*100:.2f}" if accuracy is not None else ""
    el_s = f"{elapsed_min:.1f}" if elapsed_min is not None else ""
    with open(MASTER_LOG, "a") as f:
        f.write(f"{ts}\t{run_name}\t{status}\t{acc_s}\t{el_s}\n")


def _cells_smoke(rules=RULES):
    """Smoke: all 12 cells (or a filtered subset), 1 seed each, 5000 samples."""
    for rule, scaf in itertools.product(rules, SCAFFOLDS):
        yield dict(rule=rule, scaffold=scaf, seed=0, n_neurons=400, smoke=True)


def _cells_main(rules=RULES, seeds=range(5)):
    """Main: 5 seeds per (rule, scaffold). `rules` and `seeds` can be filtered."""
    for rule, scaf in itertools.product(rules, SCAFFOLDS):
        for seed in seeds:
            yield dict(rule=rule, scaffold=scaf, seed=seed, n_neurons=400, smoke=False)


def _cells_scale():
    """Optional: pair_exp × {full, none} × 2 seeds × N=1600 (Q10)."""
    for scaf in ("full", "none"):
        for seed in range(2):
            yield dict(rule="pair_exp", scaffold=scaf, seed=seed,
                       n_neurons=1600, smoke=False)


def _cells_supp_wnoff():
    """Supplementary: {triplet, voltage} × S_full × 1 seed × weight-norm=OFF (Q5)."""
    for rule in ("triplet", "voltage"):
        yield dict(rule=rule, scaffold="full", seed=0, n_neurons=400,
                   smoke=False, weight_norm_off=True)


def run_many(cells: list[dict], device: str, dataset: str = "mnist") -> None:
    total = len(cells)
    print(f"Running {total} cells on {device} (dataset={dataset}).")
    t_all = time.time()
    for i, c in enumerate(cells, start=1):
        ds_tag = f"_{dataset.lower()}" if dataset.lower() != "mnist" else ""
        name = f"{c['rule']}_S_{c['scaffold']}_N{c['n_neurons']}_seed{c['seed']}{ds_tag}"
        if c.get("smoke"):
            name += "_smoke"
        if c.get("weight_norm_off"):
            name += "_wnoff"
        print(f"\n[{i}/{total}] {name}  (running on {device})")
        t0 = time.time()
        try:
            metrics = run_cell(device=device, dataset=dataset, **c)
            acc = metrics["accuracy"]
            elapsed = (time.time() - t0) / 60
            _log_master(name, "ok", acc, elapsed)
            print(f"   OK  acc={acc*100:.2f}%  elapsed={elapsed:.1f}min")
        except Exception as e:
            elapsed = (time.time() - t0) / 60
            _log_master(name, f"FAILED:{type(e).__name__}", None, elapsed)
            print(f"   FAILED after {elapsed:.1f}min: {e}")
    total_min = (time.time() - t_all) / 60
    print(f"\nAll {total} cells finished in {total_min:.1f} min.")


def main():
    p = argparse.ArgumentParser(description="Run the ablation matrix.")
    p.add_argument("--smoke", action="store_true", help="12 cells, 1 seed, 5k samples.")
    p.add_argument("--main", action="store_true", help="60-run main matrix (N=400).")
    p.add_argument("--scale", action="store_true", help="N=1600 scaling (pair_exp only).")
    p.add_argument("--supp-wnoff", action="store_true", help="weight-norm OFF supplementary.")
    p.add_argument("--rules", nargs="+", choices=RULES, default=None,
                   help="Limit --smoke/--main to a subset of rules "
                        "(e.g. --rules pair_exp triplet to skip voltage).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Limit --main to a subset of seed ids (default 0-4).")
    p.add_argument("--device", default="cuda:1",
                   help="CUDA device id; gpu1 per Abin 2026-04-20")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cells that would be run and exit without "
                        "launching anything.")
    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "fmnist", "cifar10", "nmnist"],
                   help="Dataset for all cells (default: mnist).")
    args = p.parse_args()

    if not any([args.smoke, args.main, args.scale, args.supp_wnoff]):
        p.error("Specify at least one of --smoke / --main / --scale / --supp-wnoff")

    rules = args.rules if args.rules else RULES
    seeds = args.seeds if args.seeds is not None else list(range(5))

    cells = []
    if args.smoke:
        cells += list(_cells_smoke(rules=rules))
    if args.main:
        cells += list(_cells_main(rules=rules, seeds=seeds))
    if args.scale:
        cells += list(_cells_scale())
    if args.supp_wnoff:
        cells += list(_cells_supp_wnoff())

    if args.dry_run:
        print(f"Would run {len(cells)} cells on {args.device}:")
        for i, c in enumerate(cells, 1):
            print(f"  [{i:>2}/{len(cells)}] {c}")
        return

    run_many(cells, device=args.device, dataset=args.dataset)


if __name__ == "__main__":
    main()
