#!/usr/bin/env python3
"""Total potentiation and depression produced by the production voltage-STDP step,
with or without the Clopath spike plateau (stdp.spike_plateau).

scripts/score/voltage_gate_saturation_probe.py recomputes the gates itself and does not know
the plateau option, so it cannot measure the plateau runs. This probe instead calls the
production step function (models/fc_stdp_layer._make_adex_voltage_step) three times per
time step on the same state: once as the network runs it, once with the depression
amplitudes set to zero (giving that step's potentiation) and once from a second layer
built with A_plus = 0 (giving its depression). The step is pure within an image (the
weights change only at the end of the image), so the two extra calls do not alter the
network's trajectory. Totals are summed over all synapses, neurons and steps.

Setup as in voltage_gate_saturation_probe.py: production loader, seed-dependent image
order, torch.compile off. With the plateau off and the tuned configuration this should
reproduce the published ratio of about 1 : 3,442 (results/voltage_ltp_gate_probe.md).

Run: CUDA_VISIBLE_DEVICES=1 python scripts/score/voltage_plateau_gate_probe.py \
         --config-file experiments/runs/<run>/config.yaml --out results/<name>.json
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

from models.fc_stdp_layer import FCStdpLayer          # noqa: E402
from experiments.scripts.run_single import load_dataset                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--plateau", choices=["config", "on", "off"], default="config",
                    help="override stdp.spike_plateau (and gate_cap_mv) from the config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = yaml.safe_load(Path(args.config_file).read_text())
    cfg.get("smoke", {}).pop("n_train_override", None)
    cfg["experiment"]["seed"] = args.seed
    cfg["training"]["n_train"] = args.n_train
    if args.plateau == "on":
        cfg["stdp"]["spike_plateau"] = True; cfg["stdp"]["gate_cap_mv"] = 1000
    elif args.plateau == "off":
        cfg["stdp"]["spike_plateau"] = False; cfg["stdp"].pop("gate_cap_mv", None)
    device = torch.device(args.device)

    net = FCStdpLayer(cfg, device=device, use_compile=False)
    cfg_ltd = copy.deepcopy(cfg); cfg_ltd["stdp"]["A_plus"] = 0.0
    ltd_step = FCStdpLayer(cfg_ltd, device=device, use_compile=False)._step_fn
    main_step = net._step_fn
    # accumulate on the device; reading a value back every step forces a GPU sync
    tot = {"ltp": torch.zeros((), device=device), "ltd": torch.zeros((), device=device),
           "spikes": torch.zeros((), device=device)}

    def wrapped(v, refrac, theta, w_adapt, z_after, x_pre_bar, u_plus_bar, u_minus_bar,
                u_homeo_bar, dw, spike_count, W, pre, train, a_minus_vec):
        out = main_step(v, refrac, theta, w_adapt, z_after, x_pre_bar, u_plus_bar,
                        u_minus_bar, u_homeo_bar, dw, spike_count, W, pre, train, a_minus_vec)
        if train:
            z = torch.zeros_like(dw)
            ltp = main_step(v, refrac, theta, w_adapt, z_after, x_pre_bar, u_plus_bar,
                            u_minus_bar, u_homeo_bar, z, spike_count, W, pre, train,
                            torch.zeros_like(a_minus_vec))[9]
            ltd = ltd_step(v, refrac, theta, w_adapt, z_after, x_pre_bar, u_plus_bar,
                           u_minus_bar, u_homeo_bar, z, spike_count, W, pre, train, a_minus_vec)[9]
            tot["ltp"] += ltp.sum(); tot["ltd"] -= ltd.sum()
            tot["spikes"] += (out[10] - spike_count).sum()
        return out

    net._step_fn = wrapped
    train_set, _ = load_dataset(cfg["data"]["data_dir"], cfg["data"]["dataset"])
    threshold = cfg.get("data", {}).get("threshold", None)
    g = torch.Generator().manual_seed(args.seed + 1_000)
    order = torch.randperm(len(train_set), generator=g).tolist()[: args.n_train]
    for k, idx in enumerate(order):
        image, _ = train_set[idx]
        if threshold is not None:
            image = (image > threshold).float()
        net.run_sample(image, train=True)
        if (k + 1) % 200 == 0:
            r = float(tot["ltp"]) / max(float(tot["ltd"]), 1e-30)
            print(f"  [{k+1}/{args.n_train}] potentiation/depression = {r:.3g}", flush=True)

    P, D, N = float(tot["ltp"]), float(tot["ltd"]), float(tot["spikes"])
    out = dict(potentiation_total=P, depression_total=D,
               potentiation_over_depression=P / max(D, 1e-30),
               spikes_per_image=N / args.n_train,
               config_file=args.config_file, seed=args.seed, n_train=args.n_train,
               spike_plateau=bool(cfg["stdp"].get("spike_plateau", False)),
               gate_cap_mv=cfg["stdp"].get("gate_cap_mv"),
               A_plus=cfg["stdp"]["A_plus"], A_minus=cfg["stdp"]["A_minus"],
               theta_ltd=cfg["stdp"]["theta_ltd"], v_rest=cfg["neuron"]["v_rest"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
