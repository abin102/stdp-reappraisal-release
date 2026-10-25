#!/usr/bin/env python3
"""Clopath's neuron and rule (models/clopath_port.py) on MNIST, 400 neurons per network.

Unchanged from Clopath's code (aEIF.m, RFdevelop.m): neuron, per-ms weight update, bounds
[0, 3], initialisation U(0.5, 2), homeostasis (1.2-s average of (u - E_L)^2, target 60).
Taken from the fully connected network (experiments/scripts/run_single.py): MNIST loader,
Poisson encoding (pixel x 63.75 Hz, 1-ms steps, 350 ms per image, then 150 ms without
input), training order randperm(seed + 1000), first n_train images, first n_test test images.
Choices beyond Clopath's code (results/voltage_clopath_neuron_port_PREREG.md):
  - input current I = g * C * W' * spikes (Clopath: g = 4), g a free parameter;
  - lateral inhibition as a current: a spike registered in step t-1 gives every other
    neuron of the network -k_inh * C pA during step t (a k_inh-mV step), k_inh free;
    the voltage is floored at -100 mV (Diehl & Cook's inhibitory reversal potential), see
    models/clopath_port.py;
  - state carries over between images (continuous simulation, as RFdevelop.m, whose
    patterns change every 100 ms without a reset); the 150-ms silent gap separates images;
  - spike counts per image = spikes registered during the image's 500-ms slot;
  - STDP is off while the test images are presented (weights frozen); the simulation
    continues from the state reached at the end of training;
  - all networks in one process share the seed: same image order, same Poisson spike
    trains, same initial weights; they differ only in g, k_inh, amplitudes, STDP on/off.

Modes:
  calibrate  STDP off, a (g x k_inh) grid, n images of the training order, no labels.
  train      the pre-registered cells: trained (both amplitude settings) and STDP-off.
             --theta_tar v1,v2,... (default 60 = RFdevelop.m) trains every cell at each
             homeostatic target (specs: theta_tar x amplitude x cell); --no_random leaves
             out the STDP-off networks (they do not depend on theta_tar);
             --prereg names the pre-registration whose SHA-256 goes into meta.json
             (results/voltage_clopath_port_thetatar_PREREG.md for the theta_tar sweep).
--dataset bloodmnist (both modes) replaces MNIST by BloodMNIST, loaded and converted to
grayscale by run_single.py's loader (results/voltage_clopath_port_blood_PREREG.md); the
encoding, training order randperm(len(train split), seed + 1000) and test order are unchanged.

Each 500-step image is captured once as a CUDA graph and replayed (the same kernels as the
eager path, so the arithmetic is the one tested in scripts/checks/clopath_port_equivalence.py;
--check-graph compares graph and eager on two images). float64 throughout.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
from models import clopath_port as cp  # noqa: E402

torch.set_grad_enabled(False)
N_IMG, N_QUIET = 350, 150
N_STEPS = N_IMG + N_QUIET
INTENSITY = 63.75
AMPS = {"10x": (cp.A3p_RF, cp.A2d_RF), "1x": (cp.A3p_PUB, cp.A2d_PUB)}


# ----------------------------------------------------------------------
def load_mnist():
    from experiments.scripts.run_single import load_dataset
    tr, te = load_dataset("./data", "mnist")
    Xtr = tr.data.reshape(len(tr), -1).float().div(255)
    Xte = te.data.reshape(len(te), -1).float().div(255)
    for ds, X in ((tr, Xtr), (te, Xte)):          # identical to the FC network's ToTensor path
        for i in (0, 1, 777, len(ds) - 1):
            assert torch.equal(ds[i][0], X[i]), "MNIST tensor path differs from run_single"
    return Xtr, tr.targets.clone(), Xte, te.targets.clone()


def load_medmnist(name):
    """BloodMNIST through the FC network's loader (run_single.load_dataset): grayscale 28x28
    (PIL convert("L")), ToTensor, flattened to 784 values in [0, 1]."""
    from experiments.scripts.run_single import load_dataset
    tr, te = load_dataset("./data", name)
    out = []
    for ds in (tr, te):
        X = torch.stack([ds[i][0] for i in range(len(ds))]).float()
        y = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.int64)
        assert X.shape[1] == 784 and float(X.min()) >= 0.0 and float(X.max()) <= 1.0
        for i in (0, 1, 777, len(ds) - 1):
            assert torch.equal(ds[i][0], X[i]) and ds[i][1] == int(y[i])
        out += [X, y]
    return tuple(out)


def load_data(dataset):
    """MNIST: the original code path (load_mnist). Other datasets: load_medmnist."""
    return load_mnist() if dataset == "mnist" else load_medmnist(dataset)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else None


class ImageRunner:
    """Runs one 500-step image through a list of ClopathNet objects that share the input,
    accumulating spike counts (whole 500-ms slot and the 350-ms presentation) and the total
    potentiation/depression per network. Optionally captured as a CUDA graph."""

    def __init__(self, nets, Ni, device, use_graph=True):
        self.nets, self.device = nets, device
        self.inp = torch.zeros(N_STEPS, Ni, dtype=torch.float64, device=device)
        self.counts = [torch.zeros(n.B, n.N, dtype=torch.float64, device=device) for n in nets]
        self.counts_img = [torch.zeros(n.B, n.N, dtype=torch.float64, device=device) for n in nets]
        self.ltp = [torch.zeros(n.B, dtype=torch.float64, device=device) for n in nets]
        self.ltd = [torch.zeros(n.B, dtype=torch.float64, device=device) for n in nets]
        self.use_graph = use_graph and device.type == "cuda"
        self.graph = None

    def _body(self):
        for i, n in enumerate(self.nets):
            self.counts[i].zero_(); self.counts_img[i].zero_(); self.ltp[i].zero_(); self.ltd[i].zero_()
        for t in range(N_STEPS):
            x = self.inp[t]
            for i, n in enumerate(self.nets):
                s = n.step(x, diag=n.plastic).to(torch.float64)
                self.counts[i].add_(s)
                if t < N_IMG:
                    self.counts_img[i].add_(s)
                if n.plastic:
                    self.ltp[i].add_(n.last_ltp); self.ltd[i].add_(n.last_ltd)

    def _capture(self):
        S0 = [{k: getattr(n, k) for k in n.STATE} for n in self.nets]
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                     # warm-up (results discarded)
            self._body()
        torch.cuda.current_stream().wait_stream(side)
        for n, s0 in zip(self.nets, S0):
            for k, v in s0.items():
                setattr(n, k, v)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._body()
            for n, s0 in zip(self.nets, S0):
                for k, v in s0.items():
                    v.copy_(getattr(n, k))
        for n, s0 in zip(self.nets, S0):                  # state lives in the static buffers
            for k, v in s0.items():
                setattr(n, k, v)
        self.graph = g

    def run(self, spikes_img):
        """spikes_img: (350, Ni) 0/1 tensor on device."""
        self.inp[:N_IMG].copy_(spikes_img)
        if not self.use_graph:
            self._body()
            return
        if self.graph is None:
            self._capture()
        self.graph.replay()


def encode(img, gen):
    """FC network's Poisson encoder: rates = pixel * 63.75 Hz * 1 ms, 350 Bernoulli draws."""
    rates = (img * INTENSITY / 1000.0 * 1.0).clamp_(0.0, 1.0)
    return torch.bernoulli(rates.unsqueeze(0).expand(N_IMG, -1), generator=gen)


def init_weights(seed, Ni=784, N=400):
    g = torch.Generator().manual_seed(int(seed))
    return torch.rand(Ni, N, generator=g, dtype=torch.float64) * 1.5 + 0.5   # rand(Ni,n)*1.5+0.5


# ----------------------------------------------------------------------
def calibrate(args):
    dev = torch.device(args.device)
    Xtr, _, _, _ = load_data(args.dataset)
    order = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(args.seed + 1000)).tolist()
    gs =[float(v) for v in args.g.split(",")]; ks = [float(v) for v in args.k.split(",")]
    cells = [(g, k) for g in gs for k in ks]
    W0 = init_weights(args.seed)
    net = cp.ClopathNet(W0.unsqueeze(0).expand(len(cells), -1, -1), g=[c[0] for c in cells],
                        k_inh=[c[1] for c in cells], plastic=False, dtype=torch.float64, device=dev, u_floor=cp.U_FLOOR_INH)
    runner = ImageRunner([net], 784, dev, use_graph=not args.no_graph)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    Xd = Xtr.to(dev)
    n = args.n_images
    spk_img = np.zeros((n, len(cells)), np.float32); act = np.zeros((n, len(cells)), np.float32)
    spk_slot = np.zeros((n, len(cells)), np.float32)
    ever = torch.zeros(len(cells), 400, dtype=torch.bool, device=dev)
    t0 = time.time()
    for i in range(n):
        runner.run(encode(Xd[order[i]], gen))
        c_img = runner.counts_img[0]
        spk_img[i] = c_img.sum(1).cpu().numpy()
        act[i] = (c_img > 0).double().mean(1).cpu().numpy()
        spk_slot[i] = runner.counts[0].sum(1).cpu().numpy()
        ever |= runner.counts[0] > 0
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n}] {time.time() - t0:.0f} s", flush=True)
    rate = spk_img.mean(0) / 400 / (N_IMG / 1000.0)
    out = dict(seed=args.seed, dataset=args.dataset, n_images=n, g=gs, k=ks, wall_s=time.time() - t0,
               definition=dict(rate_hz="mean over images and the 400 neurons of spikes in the 350-ms "
                                       "presentation / 0.35 s",
                               active_fraction="mean over images of the fraction of the 400 neurons with "
                                               ">= 1 spike in the 350-ms presentation"),
               cells=[dict(g=c[0], k_inh=c[1], rate_hz=float(rate[j]), active_fraction=float(act[:, j].mean()),
                           spikes_per_image_presentation=float(spk_img[:, j].mean()),
                           spikes_per_image_slot=float(spk_slot[:, j].mean()),
                           frac_images_silent=float((spk_img[:, j] == 0).mean()),
                           neurons_ever_active=int(ever[j].sum()))
                      for j, c in enumerate(cells)])
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "calibration.json"), "w"), indent=2)
    print(f"wrote {args.out}/calibration.json ({out['wall_s']:.0f} s)")


# ----------------------------------------------------------------------
def train(args):
    dev = torch.device(args.device)
    grid = json.load(open(args.grid))
    cells = grid["cells"]                                   # [{"id":..., "g":..., "k_inh":...}]
    amps = grid.get("amplitudes", ["10x", "1x"])
    tts = [float(v) for v in args.theta_tar.split(",")]
    out = args.out; os.makedirs(out, exist_ok=True)
    if os.path.isfile(os.path.join(out, "completed.flag")):
        print("completed.flag present, nothing to do"); return
    tr_specs = [dict(cell=c["id"], g=c["g"], k_inh=c["k_inh"], amp=a, kind="trained", theta_tar=tt)
                for tt in tts for a in amps for c in cells]
    if args.only:       # keep only the listed trained settings "cell/amp/theta_tar" (and their cells' STDP-off nets)
        keep = [tuple(v.split("/")) for v in args.only.split(",")]
        keep = {(c, a, float(t)) for c, a, t in keep}
        tr_specs = [s for s in tr_specs if (s["cell"], s["amp"], s["theta_tar"]) in keep]
        assert len(tr_specs) == len(keep), "--only names a setting that is not in the grid"
        cells = [c for c in cells if c["id"] in {s["cell"] for s in tr_specs}]
    rd_specs = [] if args.no_random else \
        [dict(cell=c["id"], g=c["g"], k_inh=c["k_inh"], amp=None, kind="random") for c in cells]
    specs = tr_specs + rd_specs
    assert specs, "nothing to run"
    W0 = init_weights(args.seed)
    nets = []
    if tr_specs:
        nets.append(cp.ClopathNet(W0.unsqueeze(0).expand(len(tr_specs), -1, -1),
                                  g=[s["g"] for s in tr_specs], k_inh=[s["k_inh"] for s in tr_specs],
                                  A3p=[AMPS[s["amp"]][0] for s in tr_specs],
                                  A2d=[AMPS[s["amp"]][1] for s in tr_specs],
                                  plastic=True, dtype=torch.float64, device=dev, fused=not args.no_fused,
                                  u_floor=cp.U_FLOOR_INH, theta_tar=[s["theta_tar"] for s in tr_specs]))
    if rd_specs:
        nets.append(cp.ClopathNet(W0.unsqueeze(0).expand(len(rd_specs), -1, -1),
                                  g=[s["g"] for s in rd_specs], k_inh=[s["k_inh"] for s in rd_specs],
                                  plastic=False, dtype=torch.float64, device=dev, u_floor=cp.U_FLOOR_INH))
    Xtr, ytr, Xte, yte = load_data(args.dataset)
    order = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(args.seed + 1000)).tolist()
    for p in range(1, args.n_passes):   # pass p: its own order, randperm(seed + 1000 + 100 p); pass 0 as before
        order += torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(args.seed + 1000 + 100 * p)).tolist()
    order = order[:args.n_train]
    assert len(order) == args.n_train, f"n_train {args.n_train} > {args.n_passes} pass(es) of {len(Xtr)} images"
    Xd, Xted = Xtr.to(dev), Xte.to(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    B = len(specs)
    ckpt = os.path.join(out, "progress.pt")
    i0 = 0
    stats = dict(spikes=np.zeros((args.n_train, B), np.float32), active=np.zeros((args.n_train, B), np.float32),
                 ltp=np.zeros((args.n_train, len(tr_specs)), np.float64),
                 ltd=np.zeros((args.n_train, len(tr_specs)), np.float64))
    if os.path.isfile(ckpt):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        for n, sd in zip(nets, ck["nets"]):
            n.load_state_dict(sd)
        gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        i0 = ck["i"]; stats = ck["stats"]
        print(f"RESUME at training image {i0}", flush=True)
    meta = dict(seed=args.seed, dataset=args.dataset, n_train=args.n_train, n_test=args.n_test, specs=specs, grid_file=args.grid,
                grid_sha256=sha256(args.grid), prereg_file=args.prereg, prereg_sha256=sha256(args.prereg),
                theta_tar=tts, no_random=args.no_random, only=args.only, n_passes=args.n_passes,
                code_sha256={f: sha256(f) for f in ("models/clopath_port.py", "scripts/train/clopath_port_mnist.py")},
                git_head=git_head(), torch=torch.__version__, device=str(dev), dtype="float64",
                fused_update=not args.no_fused, cuda_graph=not args.no_graph, u_floor=cp.U_FLOOR_INH,
                started=time.strftime("%Y-%m-%d %H:%M:%S"), resumed_at=i0)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    runner = ImageRunner(nets, 784, dev, use_graph=not args.no_graph)
    curve = open(os.path.join(out, "train_curve.csv"), "a", newline=""); cw = csv.writer(curve)
    if i0 == 0:
        cw.writerow(["image", "net", "kind", "cell", "amp", "spikes_per_img", "active_frac", "w_mean", "w_sd",
                     "frac_w0", "frac_wmax", "theta_mean", "ltp_per_img", "ltd_per_img"])
    snaps = {}
    t0 = time.time()
    for i in range(i0, args.n_train):
        runner.run(encode(Xd[order[i]], gen))
        cnt = torch.cat(runner.counts, 0)
        stats["spikes"][i] = cnt.sum(1).cpu().numpy()
        stats["active"][i] = (cnt > 0).double().mean(1).cpu().numpy()
        if tr_specs:
            stats["ltp"][i] = runner.ltp[0].cpu().numpy(); stats["ltd"][i] = runner.ltd[0].cpu().numpy()
        if (i + 1) in (1000, 5000) or ((i + 1) % 10000 == 0 and args.n_train > 10000 and (i + 1) < args.n_train):
            if tr_specs:
                snaps[f"W_{i + 1}"] = nets[0].W.float().cpu().numpy()
        if (i + 1) % args.log_every == 0:
            lo = i + 1 - args.log_every
            Wt = nets[0].W if tr_specs else None
            th = torch.cat([n.theta for n in nets], 0).mean(1).cpu().numpy()
            for j, s in enumerate(specs):
                if s["kind"] == "trained":
                    W = Wt[j]
                    wst = [float(W.mean()), float(W.std()), float((W <= 0).double().mean()),
                           float((W >= cp.wmax).double().mean())]
                    lt = [float(stats["ltp"][lo:i + 1, j].mean()), float(stats["ltd"][lo:i + 1, j].mean())]
                else:
                    wst = [float(W0.mean()), float(W0.std()), 0.0, 0.0]; lt = [0.0, 0.0]
                cw.writerow([i + 1, j, s["kind"], s["cell"], s["amp"], float(stats["spikes"][lo:i + 1, j].mean()),
                             float(stats["active"][lo:i + 1, j].mean())] + wst + [float(th[j])] + lt)
            curve.flush()
            el = time.time() - t0
            msg = " ".join(f"{s['cell']}/{s['amp'] or 'rnd'}{'/%g' % s['theta_tar'] if len(tts) > 1 and 'theta_tar' in s else ''}"
                           f":{stats['spikes'][lo:i + 1, j].mean():.0f}" for j, s in enumerate(specs))
            print(f"[{i + 1}/{args.n_train}] {el / (i + 1 - i0):.3f} s/img | spikes/img {msg}", flush=True)
        if (i + 1) % args.ckpt_every == 0 and (i + 1) < args.n_train:
            torch.save(dict(nets=[n.state_dict() for n in nets], gen=gen.get_state(), i=i + 1, stats=stats),
                       ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    curve.close()
    t_train = time.time() - t0
    np.savez_compressed(os.path.join(out, "train_stats.npz"), **stats)
    if tr_specs:
        np.savez_compressed(os.path.join(out, "weights.npz"), W_final=nets[0].W.float().cpu().numpy(),
                            W0=W0.float().numpy(), **snaps)
    # ---- test: STDP off for every network, simulation continues ----------
    # (all networks in one ClopathNet, state carried over; same order as specs)
    test_net = cp.ClopathNet.concat(nets, plastic=False)
    del nets, runner
    torch.cuda.empty_cache()
    runner = ImageRunner([test_net], 784, dev, use_graph=not args.no_graph)
    nt = min(args.n_test, len(Xte))
    counts = np.zeros((B, nt, 400), np.uint16)
    t1 = time.time()
    for i in range(nt):
        runner.run(encode(Xted[i], gen))
        counts[:, i] = torch.cat(runner.counts, 0).cpu().numpy().astype(np.uint16)
        if (i + 1) % 1000 == 0:
            print(f"  test [{i + 1}/{nt}] {(time.time() - t1) / (i + 1):.3f} s/img", flush=True)
    np.savez_compressed(os.path.join(out, "test_spike_counts.npz"), counts=counts,
                        labels=yte[:nt].numpy(), specs=json.dumps(specs))
    meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), train_minutes=t_train / 60,
                test_minutes=(time.time() - t1) / 60, max_count=int(counts.max()))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    if os.path.isfile(ckpt):
        os.remove(ckpt)
    open(os.path.join(out, "completed.flag"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"DONE train {t_train / 60:.1f} min, test {(time.time() - t1) / 60:.1f} min")


def check_graph(args):
    """Graph replay vs eager on two images: must be bitwise identical."""
    dev = torch.device(args.device)
    Xtr, _, _, _ = load_mnist()
    W0 = init_weights(0)
    res = []
    for use_graph in (False, True):
        nets = [cp.ClopathNet(W0.unsqueeze(0).expand(3, -1, -1), g=[0.2, 0.4, 0.8], k_inh=[0.0, 5.0, 20.0],
                              A3p=cp.A3p_RF, A2d=cp.A2d_RF, plastic=True, device=dev),
                cp.ClopathNet(W0.unsqueeze(0).expand(2, -1, -1), g=[0.4, 0.8], k_inh=[5.0, 20.0], plastic=False, device=dev)]
        r = ImageRunner(nets, 784, dev, use_graph=use_graph)
        gen = torch.Generator(device=dev).manual_seed(0)
        cs = []
        for i in range(3):
            r.run(encode(Xtr[i].to(dev), gen)); cs.append(torch.cat(r.counts, 0).cpu().numpy().copy())
        res.append(dict(W=nets[0].W.cpu().numpy(), u=nets[0].u.cpu().numpy(), counts=np.array(cs),
                        ltp=r.ltp[0].cpu().numpy()))
    same = all(np.array_equal(res[0][k], res[1][k]) for k in ("W", "u", "counts", "ltp"))
    print(f"graph vs eager bitwise identical: {same}; spikes per image {res[0]['counts'].sum(2).tolist()}")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "train", "check-graph"])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "bloodmnist"],
                    help="mnist (default, the original code path) or bloodmnist (run_single.py's loader, grayscale)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--g", default="0.1,0.2,0.4")
    ap.add_argument("--k", default="0,5,20")
    ap.add_argument("--n_images", type=int, default=1000)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--no_graph", action="store_true")
    ap.add_argument("--no_fused", action="store_true", help="unfused weight update (bitwise = eager reference path)")
    ap.add_argument("--theta_tar", default="60", help="homeostatic target(s) in mV^2, comma-separated "
                                                      "(RFdevelop.m: 60); train mode")
    ap.add_argument("--no_random", action="store_true", help="train mode: leave out the STDP-off networks")
    ap.add_argument("--only", default=None, help="train mode: comma-separated trained settings cell/amp/theta_tar "
                                                 "(e.g. c32/1x/240) to keep; STDP-off nets only for their cells")
    ap.add_argument("--n_passes", type=int, default=1, help="train mode: passes over the training split, pass p in "
                                                            "the order randperm(seed + 1000 + 100 p); default 1")
    ap.add_argument("--prereg", default="results/voltage_clopath_neuron_port_PREREG.md",
                    help="pre-registration whose SHA-256 is stored in meta.json")
    args = ap.parse_args()
    if args.mode == "calibrate":
        calibrate(args)
    elif args.mode == "train":
        train(args)
    else:
        sys.exit(0 if check_graph(args) else 1)


if __name__ == "__main__":
    main()
