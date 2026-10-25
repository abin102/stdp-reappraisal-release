#!/usr/bin/env python3
"""Train and evaluate one seed of a network family from its saved configuration.

run_configs/<family>.yaml is the config.yaml that experiments/scripts/run_single.py saved for the
family's first seed, with the machine-specific `experiment.device` line removed. This command runs
run_single.train_and_eval on exactly that configuration, changing only experiment.seed,
experiment.name (_seed<base> -> _seed<S>) and experiment.device, so the run directory
experiments/runs/<name> is the one the scorers in scripts/score/ read. A run that already has
completed.flag is skipped. See run_configs/INDEX.md for which family feeds which paper element.

Usage (repo root):
    PYTHONPATH=. python scripts/train/run_config.py run_configs/<family>.yaml --seed S [--device cuda:0]
    PYTHONPATH=. python scripts/train/run_config.py run_configs/<family>.yaml --seed S --dry-run
"""
import argparse
import json
import re
from pathlib import Path

import yaml


def resolve(cfg_path, seed, device):
    cfg = yaml.safe_load(open(cfg_path))
    exp = cfg["experiment"]
    base = int(exp["seed"])
    name, n = re.subn(rf"_seed{base}(?=_|$)", f"_seed{seed}", exp["name"], count=1)
    if n != 1:
        raise SystemExit(f"ERROR: run name {exp['name']!r} has no _seed{base} to replace")
    exp.update(seed=seed, name=name, device=device)
    return cfg, Path(exp["save_dir"]) / name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="run_configs/<family>.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true", help="print the run directory and exit")
    a = ap.parse_args()
    cfg, run_dir = resolve(a.config, a.seed, a.device)
    if a.dry_run:
        print(run_dir)
        return
    if (run_dir / "completed.flag").exists():
        print(f"SKIP (completed): {run_dir.name}")
        return

    from experiments.scripts.run_single import set_global_seeds, setup_logger, train_and_eval

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    log = setup_logger(run_dir / "log.txt", cfg["experiment"]["name"])
    log.info(f"RUN {run_dir.name} from {a.config}, seed {a.seed} (scripts/train/run_config.py)")
    set_global_seeds(a.seed)
    try:
        m = train_and_eval(cfg, run_dir, log)
    except Exception as e:
        log.exception(f"FAILED: {e}")
        (run_dir / "FAILED.flag").write_text(f"{type(e).__name__}: {e}\n")
        raise
    print(f"OK: {run_dir.name} {json.dumps({'accuracy': m.get('accuracy')})}")


if __name__ == "__main__":
    main()
