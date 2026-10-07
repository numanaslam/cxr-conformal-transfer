"""Config loading: YAML -> nested SimpleNamespace, with device resolution.

Imports of yaml/torch are lazy so config parsing (and the stdlib smoke test) works
in environments without the heavy stack installed.
"""
from __future__ import annotations
import argparse
from types import SimpleNamespace


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(v) for v in d]
    return d


def load_config(path: str) -> SimpleNamespace:
    import yaml  # lazy
    with open(path) as f:
        cfg = _to_ns(yaml.safe_load(f))
    # resolve device (cuda only if actually available)
    want = getattr(getattr(cfg, "compute", SimpleNamespace()), "device", "cuda")
    resolved = "cpu"
    if want == "cuda":
        try:
            import torch  # lazy
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved = "cpu"
    if not hasattr(cfg, "compute"):
        cfg.compute = SimpleNamespace()
    cfg.compute.device = resolved
    return cfg


def parse_config_arg() -> SimpleNamespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args, _ = ap.parse_known_args()
    return load_config(args.config)
