"""Seed-paired check: is the readout gap (linear - post-hoc, trained net) larger than the
rule's gain (trained - matched random, linear probe)?  Identity per seed:
    gap - gain = random_linear - trained_posthoc
so 'gap > gain' means the plasticity-off network read by a linear probe beats the trained
network read by neuron labelling (Section 3.2).  Source: results/probefix_matrix_n5.json (240 runs)."""
import json, statistics as st
from scipy.stats import ttest_rel
d = json.load(open("results/probefix_matrix_n5.json"))
R = {(r["rule"], r["dataset"], r["seed"], r["mode"]): r for r in d["runs"]}
DS = ["mnist", "emnist_letters", "fmnist", "nmnist", "organamnist", "bloodmnist", "tissuemnist", "pathmnist"]
out = {}
for rule in sorted({k[0] for k in R}):
    print("==", rule); rows = []
    for ds in DS:
        gap, gain = [], []
        for s in range(5):
            t, f = R[(rule, ds, s, "trained")], R[(rule, ds, s, "frozen")]
            gap.append(t["linear"] - t["posthoc"]); gain.append(t["linear"] - f["linear"])
        diff = [a - b for a, b in zip(gap, gain)]
        n = sum(x > 0 for x in diff); p = ttest_rel(gap, gain).pvalue
        rows.append(dict(dataset=ds, gap=st.mean(gap), gain=st.mean(gain), diff=st.mean(diff),
                         diff_sd=st.stdev(diff), seeds_gap_gt_gain=n, p=p))
        print(f"{ds:15s} gap {st.mean(gap):6.1f} gain {st.mean(gain):6.1f} diff {st.mean(diff):+6.1f}±{st.stdev(diff):.1f}"
              f"  gap>gain {n}/5  p={p:.2g}")
    out[rule] = rows
    print("  all-5-seeds gap>gain:", sum(r["seeds_gap_gt_gain"] == 5 for r in rows),
          "| all-5 gain>gap:", sum(r["seeds_gap_gt_gain"] == 0 for r in rows), "of 8")
json.dump(out, open("results/readout_vs_rule_paired.json", "w"), indent=1)
