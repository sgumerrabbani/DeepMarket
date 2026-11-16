"""Chronological splits, training-only scaling, and 30-day windows.

The big idea: the model must never see the future. So we
  1. split by TARGET date (the day we are predicting), oldest 70% -> train,
     next 15% -> validation, newest 15% -> test;
  2. throw away windows whose 30 input days start in an earlier split
     (so no window straddles a boundary);
  3. fit the feature scaler (mean / std) on training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import load_prices, load_yfinance
from .features import build_feature_table

SPLITS = ("train", "val", "test")


@dataclass
class SplitData:
    X: np.ndarray        # (n_samples, seq_len, n_features), float32
    y: np.ndarray        # (n_samples,), float32 with values 0/1
    meta: pd.DataFrame   # one row per sample: ticker, date, target_date, label


def split_boundaries(target_dates: pd.Series, train_frac: float, val_frac: float):
    """Return (train_end, val_end): the last target date in train and in val."""
    dates = np.sort(target_dates.dropna().unique())
    train_end = dates[int(len(dates) * train_frac) - 1]
    val_end = dates[int(len(dates) * (train_frac + val_frac)) - 1]
    return pd.Timestamp(train_end), pd.Timestamp(val_end)


def assign_split(target_dates: pd.Series, train_end, val_end) -> pd.Series:
    s = pd.Series("test", index=target_dates.index)
    s[target_dates <= val_end] = "val"
    s[target_dates <= train_end] = "train"
    s[target_dates.isna()] = "none"   # last day of data: no tomorrow to predict
    return s


def make_windows(df: pd.DataFrame, feature_cols: list[str], seq_len: int) -> dict[str, SplitData]:
    """Slide a seq_len-day window over each ticker and file it under its split."""
    buckets = {s: {"X": [], "y": [], "meta": []} for s in SPLITS}
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        values = g[feature_cols].to_numpy(np.float32)
        split = g["split"].to_numpy()
        for end in range(seq_len - 1, len(g)):
            start = end - seq_len + 1
            s = split[end]
            # Skip the final day (no label) and windows that cross a boundary.
            if s == "none" or split[start] != s:
                continue
            b = buckets[s]
            b["X"].append(values[start:end + 1])
            b["y"].append(g.at[end, "label"])
            b["meta"].append((ticker, g.at[end, "date"], g.at[end, "target_date"], g.at[end, "label"]))

    out = {}
    for s, b in buckets.items():
        X = np.stack(b["X"]) if b["X"] else np.zeros((0, seq_len, len(feature_cols)), np.float32)
        meta = pd.DataFrame(b["meta"], columns=["ticker", "date", "target_date", "label"])
        out[s] = SplitData(X=X.astype(np.float32), y=np.asarray(b["y"], np.float32), meta=meta)
    return out


def build_dataset(cfg: dict) -> tuple[dict[str, SplitData], dict]:
    """Full data pipeline: load -> features -> split -> scale -> windows.

    Returns (splits, info). `info` records what was done so it can be
    saved alongside the results.
    """
    d, f, sp = cfg["data"], cfg["features"], cfg["split"]

    stock = load_prices(cfg, d["tickers"])
    # SPY/QQQ come from Yahoo even when the stocks come from a CSV,
    # unless we're running offline with synthetic data.
    if d["source"] == "synthetic":
        market = load_prices(cfg, d["market_tickers"])
    elif d["source"] == "csv":
        market = load_yfinance(d["market_tickers"], d["start"], d["end"])
    else:
        market = load_prices(cfg, d["market_tickers"])

    df, stock_cols, market_cols = build_feature_table(stock, market)
    feature_cols = stock_cols + (market_cols if f["use_market"] else [])

    train_end, val_end = split_boundaries(df["target_date"], sp["train_frac"], sp["val_frac"])
    df["split"] = assign_split(df["target_date"], train_end, val_end)

    # Standardise using TRAINING rows only (otherwise test statistics leak in).
    train_rows = df["split"] == "train"
    mean = df.loc[train_rows, feature_cols].mean()
    std = df.loc[train_rows, feature_cols].std().replace(0, 1.0)
    df[feature_cols] = (df[feature_cols] - mean) / std

    splits = make_windows(df, feature_cols, f["seq_len"])
    info = {
        "feature_cols": feature_cols,
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "first_date": str(df["date"].min().date()),
        "last_date": str(df["date"].max().date()),
        "n_samples": {s: int(len(splits[s].y)) for s in SPLITS},
        "up_rate": {s: float(splits[s].y.mean()) if len(splits[s].y) else None for s in SPLITS},
        "scaler_mean": mean.round(6).to_dict(),
        "scaler_std": std.round(6).to_dict(),
    }
    return splits, info
