"""Small helpers: config loading, seeding, and saving run outputs."""
from __future__ import annotations

import copy
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RUNS_DIR = Path("runs")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Read a YAML config and apply overrides like {"features.use_market": True}."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return cfg


def set_seed(seed: int) -> None:
    """Same seed -> same weights and shuffling -> same results (on the same machine)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def make_run_dir(name: str | None, prefix: str) -> Path:
    name = name or f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Re-using a run name overwrites it, so start predictions.csv fresh.
    (run_dir / "predictions.csv").unlink(missing_ok=True)
    return run_dir


def save_json(obj, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def save_predictions(meta: pd.DataFrame, probs, split: str, path: Path) -> None:
    """Append one split's predictions to predictions.csv (used by the dashboard)."""
    out = meta.copy()
    out["prob_up"] = np.asarray(probs, dtype=float)
    out["pred"] = (out["prob_up"] >= 0.5).astype(int)
    out["split"] = split
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
