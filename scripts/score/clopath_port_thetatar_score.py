#!/usr/bin/env python3
"""Score the homeostatic-target sweep of the exact Clopath port
(results/voltage_clopath_port_thetatar_PREREG.md).

In this release the module is kept for its per-network statistics: scripts/score/appendix_j_counts.py
imports load_run and network_records. The tables that main() writes (seeds 0-2, the pre-registered
strat5k probe) are not shipped and are not cited by the paper; the paper's numbers for these networks
(Table A.10, Appendix J) come from scripts/score/clopath_port_probefix_score.py and appendix_j_counts.py.

Networks:
  - STDP off: experiments/runs/clopath_port/grid10k/seed{0,1,2} (kind "random"), reused
    (they do not depend on theta_tar);
  - trained, theta_tar = 60: the same grid10k runs (kind "trained"; the first grid);
  - trained, theta_tar = 240, 960, 3840, 15360: experiments/runs/clopath_port/thetatar10k/*/.
Before pairing, every sweep run is checked against the grid10k run of its seed: initial
weights, test labels, n_train, n_test, grid file; and its meta.json must carry the
pre-registration SHA-256 at launch.

Probe (decisions, PREREG Section 5) = scripts/score/clopath_port_score.py probes(): logistic
regression (max_iter 2000, C 1.0) on the first 5,000 test images, stratified 70/30 split,
random_state 0 (strat5k); secondary perm70 on all 10,000. Gap = trained - STDP off, same seed
and cell; mean and sample SD over seeds; beats random if mean > 0 and mean > 2 SD.

Per network (PREREG Section 8): spikes per test image, silent test images, neurons active on
the test images; spikes per training image (first 100, last 1,000), last training image with
a spike (1-based; = n_train if it never fell silent), training images without a spike; weight
mean, SD, fraction at 0 and at 3, mean pairwise cosine (end; also at 1,000 and 5,000 images);
potentiation and depression totals over training and per image over the last 1,000 images;
homeostatic average theta at the end of training and theta / theta_tar.

Usage: python scripts/score/clopath_port_thetatar_score.py
Out: results/voltage_clopath_port_thetatar.json, results/voltage_clopath_port_thetatar_tables.md
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scripts.score._blas import fixed_threads  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
from scripts.score import clopath_port_score as _score  # noqa: E402
probes, ms = _score.probes, _score.ms

PREREG = "results/voltage_clopath_port_thetatar_PREREG.md"
PREREG_SHA_AT_LAUNCH = "7dd1c61571c2dc21d87adad156ea518e26275f9b45d8681502b42e0845f83ded"
FIRST_PORT_JSON = "results/voltage_clopath_neuron_port.json"
TTS = [60.0, 240.0, 960.0, 3840.0, 15360.0]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None


def cosine(w):
    wn = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
    cs = wn.T.astype(np.float64) @ wn.astype(np.float64)
    return float((cs.sum() - np.trace(cs)) / (cs.shape[0] * (cs.shape[0] - 1)))


def wstats(w):
    return dict(mean=float(w.mean()), sd=float(w.std()), frac_w0=float((w <= 0).mean()),
                frac_wmax=float((w >= 3.0).mean()), mean_pairwise_cosine=cosine(w))


def load_run(d):
    if not os.path.isfile(os.path.join(d, "completed.flag")):
        sys.exit(f"not completed: {d}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    zf = np.load(os.path.join(d, "test_spike_counts.npz"))      # npz members decompress on every access:
    z = dict(counts=zf["counts"], labels=zf["labels"])          # read each one once
    specs = json.loads(str(zf["specs"]))
    assert specs == meta["specs"], f"spec mismatch in {d}"
    sf = np.load(os.path.join(d, "train_stats.npz")); st = {k: sf[k] for k in sf.files}
    wf = np.load(os.path.join(d, "weights.npz")); W = {k: wf[k] for k in wf.files}
    curve = list(csv.DictReader(open(os.path.join(d, "train_curve.csv"))))
    theta_end = {}
    last_img = max(int(r["image"]) for r in curve)
    for r in curve:
        if int(r["image"]) == last_img:
            theta_end[int(r["net"])] = float(r["theta_mean"])
    return dict(dir=d, meta=meta, z=z, specs=specs, st=st, W=W, theta_end=theta_end)


def network_records(run, base_label):
    """(record, counts, labels) for every network of a run."""
    meta, st, W, specs = run["meta"], run["st"], run["W"], run["specs"]
    ntr = st["spikes"].shape[0]
    last = slice(max(ntr - 1000, 0), ntr)
    tr_index = {j: t for t, j in enumerate([j for j, s in enumerate(specs) if s["kind"] == "trained"])}
    out = []
    for j, s in enumerate(specs):
        sp = st["spikes"][:, j]
        nz = np.nonzero(sp > 0)[0]
        rec = dict(seed=meta["seed"], run=run["dir"], source=base_label, idx=j, cell=s["cell"], g=s["g"],
                   k_inh=s["k_inh"], amp=s["amp"], kind=s["kind"],
                   theta_tar=(float(s.get("theta_tar", 60.0)) if s["kind"] == "trained" else None),
                   n_train=int(ntr),
                   spikes_per_train_image_first100=float(sp[:100].mean()),
                   spikes_per_train_image_last1000=float(sp[last].mean()),
                   active_fraction_train_last1000=float(st["active"][last, j].mean()),
                   last_train_image_with_spike=int(nz[-1] + 1) if len(nz) else 0,
                   train_images_without_spike=int((sp == 0).sum()),
                   theta_mean_end=run["theta_end"].get(j))
        rec["fell_silent_during_training"] = bool(rec["last_train_image_with_spike"] < ntr)
        if s["kind"] == "trained":
            t = tr_index[j]
            rec["weights_end"] = wstats(W["W_final"][t])
            for k in ("W_1000", "W_5000"):
                if k in W:
                    rec[f"weights_{k[2:]}"] = wstats(W[k][t])
            rec.update(ltp_total=float(st["ltp"][:, t].sum()), ltd_total=float(st["ltd"][:, t].sum()),
                       ltp_per_image_last1000=float(st["ltp"][last, t].mean()),
                       ltd_per_image_last1000=float(st["ltd"][last, t].mean()))
            if rec["theta_mean_end"] is not None:
                rec["theta_over_target_end"] = rec["theta_mean_end"] / rec["theta_tar"]
        out.append((rec, run["z"]["counts"][j], run["z"]["labels"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="experiments/runs/clopath_port/grid10k")
    ap.add_argument("--sweep", default="experiments/runs/clopath_port/thetatar10k")
    ap.add_argument("--out", default="results/voltage_clopath_port_thetatar")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--allow_partial", action="store_true", help="score whatever sweep runs are complete")
    args = ap.parse_args()

    base = {}
    for d in sorted(glob.glob(os.path.join(args.base, "seed*"))):
        if (not os.path.isfile(os.path.join(d, "train_stats.npz"))
                and os.path.isfile(os.path.join(d, "train_spike_counts_pf.npz"))):
            continue   # re-extraction-only half of another run (seed0_all_a/_b of seed0_all): no training outputs
        r = load_run(d); base[r["meta"]["seed"]] = r
    checks = []
    sweep = []
    for d in sorted(glob.glob(os.path.join(args.sweep, "seed*"))):
        if (not os.path.isfile(os.path.join(d, "train_stats.npz"))
                and os.path.isfile(os.path.join(d, "train_spike_counts_pf.npz"))):
            continue   # re-extraction-only half of another run (seed0_all_a/_b of seed0_all): no training outputs
        if not os.path.isfile(os.path.join(d, "completed.flag")):
            if args.allow_partial:
                continue
            sys.exit(f"not completed: {d}")
        r = load_run(d); m = r["meta"]; b = base[m["seed"]]
        c = dict(run=d, seed=m["seed"], theta_tar=m["theta_tar"], n_networks=len(r["specs"]),
                 prereg_sha256_ok=m.get("prereg_sha256") == PREREG_SHA_AT_LAUNCH,
                 W0_equal_base=bool(np.array_equal(r["W"]["W0"], b["W"]["W0"])),
                 test_labels_equal_base=bool(np.array_equal(r["z"]["labels"], b["z"]["labels"])),
                 n_train_equal=m["n_train"] == b["meta"]["n_train"], n_test_equal=m["n_test"] == b["meta"]["n_test"],
                 grid_equal=m["grid_sha256"] == b["meta"]["grid_sha256"],
                 only_trained=all(s["kind"] == "trained" for s in r["specs"]),
                 started=m["started"], finished=m.get("finished"), train_minutes=m.get("train_minutes"),
                 test_minutes=m.get("test_minutes"), device_note=m.get("device"))
        c["ok"] = all(c[k] for k in ("prereg_sha256_ok", "W0_equal_base", "test_labels_equal_base", "n_train_equal",
                                     "n_test_equal", "grid_equal", "only_trained"))
        checks.append(c)
        if not c["ok"]:
            sys.exit(f"pairing check failed: {json.dumps(c, indent=1)}")
        sweep.append(r)
    nets = []
    for s, r in sorted(base.items()):
        nets += network_records(r, "grid10k (first grid)")
    for r in sweep:
        nets += network_records(r, "thetatar10k (sweep)")
    print(f"{len(nets)} networks ({len(base)} base seeds, {len(sweep)} sweep runs); fitting probes ...", flush=True)
    res = Parallel(n_jobs=args.jobs)(delayed(fixed_threads(probes, 1))(c, y) for _, c, y in nets)
    # the lbfgs fit is not bit-reproducible across runs (BLAS summation order); fit every probe a second
    # time and report the largest difference (decisions use the first fit)
    res2 = Parallel(n_jobs=args.jobs)(delayed(fixed_threads(probes, 1))(c, y) for _, c, y in nets)
    rows = []
    for (rec, _, _), p, p2 in zip(nets, res, res2):
        rec.update(p)
        rec["probe_refit_abs_diff"] = {k: abs(p[k] - p2[k]) for k in ("strat5k", "perm70")}
        rows.append(rec)
    refit = dict(max_abs_diff_strat5k=max(r["probe_refit_abs_diff"]["strat5k"] for r in rows),
                 max_abs_diff_perm70=max(r["probe_refit_abs_diff"]["perm70"] for r in rows),
                 n_networks_differing=sum(any(v > 0 for v in r["probe_refit_abs_diff"].values()) for r in rows))

    # consistency with the first port's scored JSON (same networks: STDP off and theta_tar = 60)
    cons = None
    if os.path.isfile(FIRST_PORT_JSON):
        old = {(n["seed"], n["cell"], n["amp"], n["kind"]): n for n in json.load(open(FIRST_PORT_JSON))["networks"]}
        diffs = [abs(r[p] - old[(r["seed"], r["cell"], r["amp"], r["kind"])][p]) for r in rows
                 if r["source"].startswith("grid10k") for p in ("strat5k", "perm70")]
        cons = dict(file=FIRST_PORT_JSON, n_values=len(diffs), max_abs_diff_points=float(max(diffs)))

    cells = sorted({r["cell"] for r in rows}); amps = ["10x", "1x"]
    seeds_all = sorted(base)
    summ = []
    for c in cells:
        rnd = {r["seed"]: r for r in rows if r["cell"] == c and r["kind"] == "random"}
        for a in amps:
            for tt in TTS:
                trn = {r["seed"]: r for r in rows if r["cell"] == c and r["amp"] == a and r["kind"] == "trained"
                       and r["theta_tar"] == tt}
                ss = sorted(set(trn) & set(rnd))
                if not ss:
                    continue
                gaps = [trn[s]["strat5k"] - rnd[s]["strat5k"] for s in ss]
                m, sd, n = ms(gaps)
                beats = bool(n >= 2 and m > 0 and m > 2 * sd)
                gp = [trn[s]["perm70"] - rnd[s]["perm70"] for s in ss]
                margin = (m - max(0.0, 2 * sd)) if m is not None else None   # > 0 means beats random
                row = dict(cell=c, amp=a, theta_tar=tt, g=trn[ss[0]]["g"], k_inh=trn[ss[0]]["k_inh"], seeds=ss,
                           decision_margin=margin,
                           source="first grid" if tt == 60.0 else "sweep",
                           trained=[trn[s]["strat5k"] for s in ss], random=[rnd[s]["strat5k"] for s in ss],
                           gaps=gaps, gap_mean=m, gap_sd=sd, n=n, beats_random=beats,
                           trained_perm70=[trn[s]["perm70"] for s in ss], random_perm70=[rnd[s]["perm70"] for s in ss],
                           gaps_perm70=gp, gap_perm70_mean=ms(gp)[0], gap_perm70_sd=ms(gp)[1])
                for key in ("spikes_per_test_image", "silent_test_images", "neurons_ever_active_test",
                            "spikes_per_train_image_first100", "spikes_per_train_image_last1000",
                            "last_train_image_with_spike", "train_images_without_spike",
                            "ltp_total", "ltd_total", "ltp_per_image_last1000", "ltd_per_image_last1000",
                            "theta_mean_end", "theta_over_target_end"):
                    row[f"trained_{key}"] = [trn[s].get(key) for s in ss]
                row["trained_fell_silent_during_training"] = [trn[s]["fell_silent_during_training"] for s in ss]
                for key in ("mean", "sd", "frac_w0", "frac_wmax", "mean_pairwise_cosine"):
                    row[f"trained_w_{key}"] = [trn[s]["weights_end"][key] for s in ss]
                for key in ("spikes_per_test_image", "theta_mean_end"):
                    row[f"random_{key}"] = [rnd[s].get(key) for s in ss]
                summ.append(row)
    new = [r for r in summ if r["theta_tar"] != 60.0]
    beats_new = [r for r in new if r["beats_random"]]
    complete = len(new) == 9 * 2 * 4 and all(r["n"] == len(seeds_all) for r in summ)
    if not complete:
        verdict = "INCOMPLETE: not every cell has all seeds at every target; no verdict."
    elif not beats_new:
        verdict = ("No. No homeostatic target between 60 and 15,360 mV^2 lets the rule beat random initialisation "
                   "on Clopath's own neuron in this MNIST setting: no cell beats its STDP-off network.")
    else:
        verdict = (f"{len(beats_new)} cell(s) beat random at 10,000 images; confirmation at 60,000 images "
                   "(PREREG Section 5) pending.")
    def rng(rows_, key):
        v = [r[key] for r in rows_]
        return [float(min(v)), float(max(v))] if v else None

    def silent_all(r):
        return all(x == 1.0 for x in r["trained_silent_test_images"])

    def numbers(rows_):
        return dict(n_cells=len(rows_), trained_mean_range=rng([dict(v=np.mean(r["trained"])) for r in rows_], "v"),
                    stdp_off_mean_range=rng([dict(v=np.mean(r["random"])) for r in rows_], "v"),
                    gap_mean_range=rng(rows_, "gap_mean"), n_beats_random=sum(r["beats_random"] for r in rows_),
                    n_silent_on_test_every_seed=sum(silent_all(r) for r in rows_),
                    n_fell_silent_during_training_every_seed=sum(all(r["trained_fell_silent_during_training"])
                                                                 for r in rows_),
                    best_cell=(lambda r: dict(cell=r["cell"], amp=r["amp"], theta_tar=r["theta_tar"],
                                              trained=ms(r["trained"])[:2], stdp_off=ms(r["random"])[:2],
                                              gap=[r["gap_mean"], r["gap_sd"]]))(max(rows_, key=lambda r: r["gap_mean"]))
                    if rows_ else None)

    table3_numbers = dict(all_targets=numbers(summ), new_targets_only=numbers(new),
                          by_target={f"{tt:g}": numbers([r for r in summ if r["theta_tar"] == tt]) for tt in TTS})
    out = dict(prereg=dict(file=PREREG, sha256_at_launch=PREREG_SHA_AT_LAUNCH, sha256_now=sha(PREREG)),
               probe="strat5k (decisions); perm70 reported", seeds=seeds_all, theta_tar=TTS,
               complete=complete, sweep_run_checks=checks, consistency_with_first_port=cons,
               probe_refit=refit,
               closest_to_decision=sorted([dict(cell=r["cell"], amp=r["amp"], theta_tar=r["theta_tar"],
                                                margin=r["decision_margin"]) for r in summ],
                                          key=lambda d: -d["margin"])[:5],
               check_column_path=(json.load(open("results/clopath_port_thetatar/check_column_path.json"))
                                  if os.path.isfile("results/clopath_port_thetatar/check_column_path.json") else None),
               check_default_reproduction=(
                   {k: v for k, v in json.load(open("results/clopath_port_thetatar/check_default_reproduction.json")).items()
                    if k != "networks"}
                   if os.path.isfile("results/clopath_port_thetatar/check_default_reproduction.json") else None),
               cells=summ, networks=rows,
               any_new_cell_beats_random=bool(beats_new), cells_beating_random=[
                   dict(cell=r["cell"], amp=r["amp"], theta_tar=r["theta_tar"], gap_mean=r["gap_mean"],
                        gap_sd=r["gap_sd"]) for r in beats_new],
               table3_numbers=table3_numbers, verdict=verdict)
    json.dump(out, open(args.out + ".json", "w"), indent=2, default=float)

    def f(v, d=1):
        m, sd, n = ms(v)
        return "–" if m is None else (f"{m:.{d}f} ± {sd:.{d}f}" if n > 1 else f"{m:.{d}f}")

    def fg(r):
        return f"{r['gap_mean']:+.1f} ± {r['gap_sd']:.1f}" + (" **beats**" if r["beats_random"] else "")

    L = [f"Probe for decisions: strat5k (first 5,000 test images, stratified 70/30, random_state 0). "
         f"Mean ± sample SD over seeds {seeds_all}; gap = trained - STDP off, paired by seed, in points. "
         f"theta_tar = 60 is the first grid (not rerun). 11.4 % = score of a network without spikes.", ""]
    for a in amps:
        L += [f"### Amplitudes {a}: trained accuracy (%) and gap, by theta_tar (mV^2)", "",
              "| cell | g | k_inh | STDP off | " + " | ".join(f"{tt:g}: trained | gap" for tt in TTS) + " |",
              "|---|--:|--:|--:|" + "--:|--:|" * len(TTS)]
        for c in cells:
            rs = {r["theta_tar"]: r for r in summ if r["cell"] == c and r["amp"] == a}
            r0 = next(iter(rs.values()))
            L.append(f"| {c} | {r0['g']:g} | {r0['k_inh']:g} | {f(r0['random'])} | " +
                     " | ".join((f"{f(rs[tt]['trained'])} | {fg(rs[tt])}" if tt in rs else "– | –") for tt in TTS) + " |")
        L.append("")
    L += ["### Activity: spikes per test image (trained; STDP off in the first column) and last training image "
          "with a spike (median over seeds; 10,000 = did not fall silent)", "",
          "| cell | amp | STDP off spikes/test img | " + " | ".join(f"{tt:g}: spikes/test img | last spike" for tt in TTS) + " |",
          "|---|---|--:|" + "--:|--:|" * len(TTS)]
    for c in cells:
        for a in amps:
            rs = {r["theta_tar"]: r for r in summ if r["cell"] == c and r["amp"] == a}
            r0 = next(iter(rs.values()))
            L.append(f"| {c} | {a} | {f(r0['random_spikes_per_test_image'], 0)} | " + " | ".join(
                (f"{f(rs[tt]['trained_spikes_per_test_image'], 0)} | {int(np.median(rs[tt]['trained_last_train_image_with_spike']))}"
                 if tt in rs else "– | –") for tt in TTS) + " |")
    L += ["", "### Weights at the end of training (trained): mean / fraction at 0 / fraction at 3 / mean pairwise "
          "cosine (initial U(0.5, 2): 1.25 / 0 / 0 / about 0.89)", "",
          "| cell | amp | " + " | ".join(f"{tt:g}" for tt in TTS) + " |", "|---|---|" + "---|" * len(TTS)]
    for c in cells:
        for a in amps:
            rs = {r["theta_tar"]: r for r in summ if r["cell"] == c and r["amp"] == a}
            L.append(f"| {c} | {a} | " + " | ".join(
                (f"{np.mean(rs[tt]['trained_w_mean']):.2f} / {np.mean(rs[tt]['trained_w_frac_w0']):.2f} / "
                 f"{np.mean(rs[tt]['trained_w_frac_wmax']):.2f} / {np.mean(rs[tt]['trained_w_mean_pairwise_cosine']):.2f}"
                 if tt in rs else "–") for tt in TTS) + " |")
    L += ["", "### Potentiation and depression totals over training (summed over synapses, before clipping; "
          "mean over seeds) and theta / theta_tar at the end of training", "",
          "| cell | amp | " + " | ".join(f"{tt:g}: LTP total | LTD total | theta/target" for tt in TTS) + " |",
          "|---|---|" + "--:|--:|--:|" * len(TTS)]
    for c in cells:
        for a in amps:
            rs = {r["theta_tar"]: r for r in summ if r["cell"] == c and r["amp"] == a}
            L.append(f"| {c} | {a} | " + " | ".join(
                (f"{np.mean(rs[tt]['trained_ltp_total']):.3g} | {np.mean(rs[tt]['trained_ltd_total']):.3g} | "
                 f"{np.mean([v for v in rs[tt]['trained_theta_over_target_end'] if v is not None]):.3g}"
                 if tt in rs else "– | – | –") for tt in TTS) + " |")
    L += ["", f"Any new cell beats random: **{'yes' if beats_new else 'no'}**. {verdict}"]
    open(args.out + "_tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:2 * (len(cells) + 5)]))
    print(verdict)
    print(f"probe refit: max |diff| strat5k {refit['max_abs_diff_strat5k']:.3f}, perm70 "
          f"{refit['max_abs_diff_perm70']:.3f} points ({refit['n_networks_differing']} networks differ); "
          f"closest cell to the decision boundary: {out['closest_to_decision'][0]}")
    if cons:
        print(f"consistency with the first port's JSON: max |diff| {cons['max_abs_diff_points']:.3f} points "
              f"over {cons['n_values']} values")


if __name__ == "__main__":
    main()
