"""Config merger for the STDP scaffolding ablation.

Layers three YAML files into one resolved config:
    final = deep_merge(base, rule_override, scaffold_override)

Then applies scaffold-toggle semantics:
    lateral_inhibition=false  -> scaffold.inh_weight    := 0.0
    adaptive_threshold=false  -> scaffold.theta_plus    := 0.0
                                 scaffold.tc_theta_decay := inf
                                 (decay_theta computed from tc_theta_decay
                                  will be 1.0, freezing theta at theta_init)

Usage:
    python -m experiments.scripts.config resolve \
        --rule triplet --scaffold full --out /tmp/resolved.yaml

    python -m experiments.scripts.config dump-all \
        --out-dir /tmp/resolved_configs

The CLI is only for inspection / sanity checking. run_single.py imports
`resolve()` and `apply_scaffold_toggles()` directly.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import yaml


RULES = ("pair_exp", "triplet", "voltage")
SCAFFOLDS = ("full", "inh_only", "thr_only", "none")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins on scalar keys. Does NOT merge lists."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_scaffold_toggles(cfg: dict) -> dict:
    """Translate scaffold.lateral_inhibition/adaptive_threshold bools into
    concrete numerical overrides on the scaffold parameters."""
    cfg = copy.deepcopy(cfg)
    scaf = cfg.setdefault("scaffold", {})
    if not scaf.get("lateral_inhibition", True):
        scaf["inh_weight"] = 0.0
    if not scaf.get("adaptive_threshold", True):
        scaf["theta_plus"] = 0.0
        scaf["tc_theta_decay"] = math.inf   # -> decay_theta = exp(-dt/inf) = 1.0
    return cfg


def resolve(rule: str, scaffold: str, config_dir: Path | None = None) -> dict:
    """Resolve a (rule, scaffold) pair into a concrete config dict."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if scaffold not in SCAFFOLDS:
        raise ValueError(f"scaffold must be one of {SCAFFOLDS}, got {scaffold!r}")
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    base = _load(config_dir / "base_config.yaml")
    rule_ov = _load(config_dir / f"rule_{rule}.yaml")
    scaf_ov = _load(config_dir / f"scaffold_{scaffold}.yaml")
    merged = deep_merge(deep_merge(base, rule_ov), scaf_ov)
    return apply_scaffold_toggles(merged)


def _cli_resolve(args) -> None:
    cfg = resolve(args.rule, args.scaffold)
    out = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
    if args.out:
        Path(args.out).write_text(out)
        print(f"Wrote resolved config -> {args.out}")
    else:
        print(out)


def _cli_dump_all(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for rule in RULES:
        for scaf in SCAFFOLDS:
            cfg = resolve(rule, scaf)
            path = out_dir / f"{rule}_S_{scaf}.yaml"
            path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
            inh = cfg["scaffold"].get("inh_weight")
            theta_plus = cfg["scaffold"].get("theta_plus")
            tc_decay = cfg["scaffold"].get("tc_theta_decay")
            rule_name = cfg["rule"]["name"]
            neuron_model = cfg["neuron"].get("model", "?")
            lines.append(
                f"{rule:>8} x S_{scaf:<8} | rule={rule_name:<10} "
                f"neuron={neuron_model:<4} inh={inh} theta+={theta_plus} "
                f"tc_theta_decay={tc_decay}"
            )
    print("Wrote 12 resolved configs ->", out_dir)
    print("")
    print("Sanity:")
    for l in lines:
        print("  " + l)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_resolve = sub.add_parser("resolve", help="Resolve one (rule, scaffold) cell.")
    p_resolve.add_argument("--rule", required=True, choices=RULES)
    p_resolve.add_argument("--scaffold", required=True, choices=SCAFFOLDS)
    p_resolve.add_argument("--out", default=None)
    p_resolve.set_defaults(fn=_cli_resolve)

    p_dump = sub.add_parser("dump-all", help="Resolve all 12 cells to a directory.")
    p_dump.add_argument("--out-dir", required=True)
    p_dump.set_defaults(fn=_cli_dump_all)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
