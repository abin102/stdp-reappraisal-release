"""Rule gain (trained - same network with plasticity off) measured under each readout,
paired by seed.  Shows that the readout changes the size, and for voltage-STDP the sign,
of the rule's measured contribution (Section 3.3; Figure 3b, c).  Source: results/probefix_matrix_n5.json (240 runs).
Output: results/gain_by_readout.json"""
import json, statistics as st
d = json.load(open("results/probefix_matrix_n5.json"))
R = {(r["rule"], r["dataset"], r["seed"], r["mode"]): r for r in d["runs"]}
DS = ["mnist", "emnist_letters", "nmnist", "fmnist", "organamnist", "bloodmnist", "tissuemnist", "pathmnist"]
out = {}
for rule in ["pair_exp", "triplet", "voltage"]:
    out[rule] = []
    for ds in DS:
        g = lambda k: [R[(rule, ds, s, "trained")][k] - R[(rule, ds, s, "frozen")][k] for s in range(5)]
        lin, lab = g("linear"), g("posthoc")
        diff = [b - a for a, b in zip(lin, lab)]
        row = dict(dataset=ds, gain_linear=st.mean(lin), gain_linear_sd=st.stdev(lin),
                   gain_labelling=st.mean(lab), gain_labelling_sd=st.stdev(lab),
                   labelling_minus_linear=st.mean(diff), seeds_labelling_gt_linear=sum(x > 0 for x in diff))
        out[rule].append(row)
        print(f"{rule:9s}{ds:15s} linear {row['gain_linear']:+6.1f}  labelling {row['gain_labelling']:+6.1f}"
              f"  diff {row['labelling_minus_linear']:+5.1f}  seeds lab>lin {row['seeds_labelling_gt_linear']}/5")
json.dump(out, open("results/gain_by_readout.json", "w"), indent=1)
