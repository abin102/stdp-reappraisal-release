#!/usr/bin/env python3
"""Per-cell readout table for the MNIST scaffold matrix (Section 4.1, Figure 2).

Recomputes, from the saved frozen spike counts, both readouts on IDENTICAL
features for all 60 cells (3 rules x 4 scaffolds x 5 seeds) and writes:

    results/scaffold_readout_table.csv     one row per cell
    results/scaffold_readout_summary.md    the aggregates Section 4.1 quotes

Protocol (matches scripts/compare_readout_protocols.py, the matched/train-fit
convention adopted 2026-08):
  post-hoc : the Diehl-Cook value carried over from
             results/posthoc_both_conventions.csv (`posthoc_dc`), which divides by
             class size when labelling neurons and by group size when predicting.
             This script does NOT recompute it: scripts/score/posthoc_both_conventions.py
             is the single implementation of that readout, so run it first. (Until
             2026-10-05 this column held the summed variant -- both divisions
             omitted -- which differs on 41 of the 60 cells and is NOT what Table
             A.3 or Section 4.1 report. The old values are preserved in that same
             file as `posthoc_summed`.)
  linear   : LogisticRegression(max_iter=100000, C=1.0), fit on the run's TRAIN
             features, evaluated on the FULL 10k test set. No feature scaling
             -- scaling or a lower max_iter under-converges the S_full cells and
             costs ~1.2 pp there.

Cells are enumerated explicitly (never by directory scan) so the table is
reproducible; a missing cell is an error, not a silently dropped row.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np

from threadpoolctl import threadpool_limits
threadpool_limits(1)   # single-threaded BLAS: probe fits reproduce bit for bit (scripts/score/_blas.py)
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

ROOT     = Path(__file__).resolve().parents[2]
RUNS     = ROOT / "experiments" / "runs"
RULES    = ["pair_exp", "triplet", "voltage"]
SCAFFOLDS= ["S_full", "S_inh_only", "S_thr_only", "S_none"]
SEEDS    = [0, 1, 2, 3, 4]
MAJORITY = 11.35          # MNIST test-set frequency of class 1 (1135/10000)


def _missing_runs_msg(d):
    """Run directories are gitignored, so a fresh clone has none of them."""
    return (
        f"ERROR: missing run directory {d}\n\n"
        "This script reads per-run spike counts from experiments/runs/, which is\n"
        "gitignored -- a clone of this repository contains the analysis code and the\n"
        "summary tables in results/, but not the trained networks themselves.\n\n"
        "To regenerate the runs:  experiments/scripts/run_single.py --rule <rule>\n"
        "  --scaffold <scaffold> --seed <seed>   (60 cells; see README)\n"
        "To read the already-computed numbers instead, without retraining:\n"
        "  results/scaffold_readout_table.csv   (Table A.3, Figure 6a)\n"
        "  results/posthoc_both_conventions.csv (Figure 2, Table A.2)"
    )


def cell_dir(rule, scaffold, seed):
    # Voltage uses the CORRECTED re-runs (tc_theta_decay = 1e4). The bare
    # directories are the superseded tc=1e6 cells: rule_voltage.yaml:61-65
    # records the 2026-04-28 fix and the 20 re-runs it produced, but this
    # function used to resolve to the bare names, so the re-runs were never
    # read. See results/voltage_audit_2026-09.md §5. Pair and triplet are
    # unaffected -- their tc_theta_decay was never changed.
    if rule == "voltage":
        for suffix in ("__b_fix_tc1e4", "__b_fix_tc1e4_60k"):
            d = RUNS / f"{rule}_{scaffold}_N400_seed{seed}{suffix}"
            if d.is_dir():
                return d
        sys.exit(f"ERROR: missing corrected voltage cell for {scaffold} seed {seed}")
    d = RUNS / f"{rule}_{scaffold}_N400_seed{seed}"
    if not d.is_dir():
        sys.exit(_missing_runs_msg(d))
    return d


def dc_posthoc():
    """The Diehl-Cook post-hoc accuracy per run, from the single implementation."""
    import csv
    f = ROOT / "results" / "posthoc_both_conventions.csv"
    if not f.exists():
        sys.exit(f"ERROR: {f} not found -- run scripts/score/posthoc_both_conventions.py "
                 f"first; it is the only implementation of the Diehl-Cook readout.")
    return {r["run"]: float(r["posthoc_dc"]) / 100.0
            for r in csv.DictReader(open(f))
            if r["group"] == "scaffold" and r["dataset"] == "mnist"}


def evaluate(d):
    zte = np.load(d / "test_spike_counts.npz")
    ztr = np.load(d / "train_spike_counts.npz")
    Xte, yte, asg = zte["test_spike_counts"], zte["test_labels"], zte["assignments"]
    Xtr, ytr      = ztr["train_spike_counts"], ztr["train_labels"]
    n_cls = int(yte.max()) + 1

    clf = LogisticRegression(max_iter=100000, C=1.0, n_jobs=4)
    clf.fit(Xtr, ytr)
    linear = float(clf.score(Xte, yte))

    met  = json.loads((d / "metrics.json").read_text())
    rate = Xte.mean(0)
    return dict(
        linear=linear,
        dead_frac=float((rate <= 1e-9).mean()),
        firing_rate_cv=float(met["firing_rate_cv"]),
        cosine=float(met["mean_cosine_similarity"]),
        n_asg_classes=int((np.bincount(asg[asg >= 0], minlength=n_cls) > 0).sum()),
    )


def main():
    DC = dc_posthoc()
    rows = []
    for rule in RULES:
        for sc in SCAFFOLDS:
            for seed in SEEDS:
                d = cell_dir(rule, sc, seed)
                r = evaluate(d)
                if d.name not in DC:
                    sys.exit(f"ERROR: results/posthoc_both_conventions.csv has no "
                             f"Diehl-Cook post-hoc value for {d.name} -- regenerate it "
                             f"with scripts/score/posthoc_both_conventions.py before building "
                             f"this table.")
                r["posthoc"] = DC[d.name]
                r.update(run=d.name, rule=rule, scaffold=sc, seed=seed)
                rows.append(r)
                print(f"  {d.name:34s} post={100*r['posthoc']:6.2f}%  "
                      f"linear={100*r['linear']:6.2f}%", flush=True)
    assert len(rows) == 60, len(rows)

    out_csv = ROOT / "results" / "scaffold_readout_table.csv"
    cols = ["run","rule","scaffold","seed","posthoc","linear","gap_pp",
            "dead_frac","firing_rate_cv","cosine","n_asg_classes"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            r["gap_pp"] = 100 * (r["linear"] - r["posthoc"])
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    ph = np.array([100*r["posthoc"] for r in rows])
    lg = np.array([100*r["linear"]  for r in rows])
    pt = np.array([r["rule"] in ("pair_exp","triplet") for r in rows])
    above = lg > MAJORITY + 0.05

    L = ["# Scaffold-matrix readout table (Section 4.1)", "",
         "Both readouts on identical frozen spike counts; 3 rules x 4 scaffolds",
         "x 5 seeds = 60 cells. Per-cell values in `scaffold_readout_table.csv`.",
         "", "## Per-scaffold means (%)", "",
         "| rule | scaffold | post-hoc | linear | gap |", "|---|---|--:|--:|--:|"]
    for rule in RULES:
        for sc in SCAFFOLDS:
            s = [r for r in rows if r["rule"] == rule and r["scaffold"] == sc]
            a = np.mean([100*x["posthoc"] for x in s])
            b = np.mean([100*x["linear"]  for x in s])
            sd= np.std ([100*x["linear"]  for x in s])
            L.append(f"| {rule} | {sc} | {a:.2f} | {b:.2f} ± {sd:.2f} | {b-a:+.1f} |")

    g = lg[pt] - ph[pt]
    L += ["", "## Claims quoted in Section 4.1", "",
          "| claim | recomputed |", "|---|---|",
          f"| bound holds, cells clearing majority floor | {int((ph[above]<=lg[above]).sum())}/{int(above.sum())} |",
          f"| bound holds, pair+triplet cells | {int((ph[pt]<=lg[pt]).sum())}/{int(pt.sum())} |",
          f"| median gap (pair+triplet) | {np.median(g):+.1f} pp |",
          f"| max gap (pair+triplet) | {g.max():+.1f} pp |",
          f"| lowest-scoring pair/triplet cell under linear readout | {lg[pt].min():.1f}% |",
          f"| cells at/below majority floor under linear readout | {int((lg<=MAJORITY+0.05).sum())} (all voltage) |",
          f"| worst shortfall below floor | {MAJORITY-lg.min():.2f} pp |",
          f"| majority-class floor (MNIST) | {MAJORITY}% |"]

    def m(rule, sc, k): return np.mean([100*r[k] for r in rows
                                        if r["rule"]==rule and r["scaffold"]==sc])
    rule_eff_ph = m("triplet","S_full","posthoc") - m("pair_exp","S_full","posthoc")
    rule_eff_lg = m("triplet","S_full","linear")  - m("pair_exp","S_full","linear")
    L += ["", "## Readout effect vs rule effect at S_full", "",
          f"- switching the readout: {m('pair_exp','S_full','linear')-m('pair_exp','S_full','posthoc'):+.1f} pp (pair), "
          f"{m('triplet','S_full','linear')-m('triplet','S_full','posthoc'):+.1f} pp (triplet)",
          f"- switching the rule (pair->triplet): {rule_eff_ph:+.1f} pp (post-hoc), {rule_eff_lg:+.1f} pp (linear)"]

    (ROOT / "results" / "scaffold_readout_summary.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out_csv}")
    print(f"wrote {ROOT/'results'/'scaffold_readout_summary.md'}")


if __name__ == "__main__":
    main()
