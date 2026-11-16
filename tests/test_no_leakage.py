"""
Test checks data pipeline for look-ahead leakage, and that the train/val/test splits are chronological and not overlapping. 
Using synthetic prices here so its running offline
"""
import numpy as np
import pandas as pd

from deepmarket.dataset import build_dataset
from deepmarket.features import add_stock_features
from deepmarket.utils import load_config


def _cfg(use_market=False):
    return load_config("configs/default.yaml", {
        "data.source": "synthetic", "data.start": "2021-01-01", "data.end": "2023-12-31",
        "data.tickers": ["AAA", "BBB"], "features.use_market": use_market,
    })


def test_label_is_next_day_direction():
    prices = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=4), "ticker": "X",
        "open": 1.0, "high": 2.0, "low": 0.5, "close": [1.0, 1.1, 1.05, 1.2], "volume": 100.0,
    })
    df = add_stock_features(prices)
    assert df["label"].tolist()[:3] == [1.0, 0.0, 1.0]   # up, down, up
    assert np.isnan(df["label"].iloc[-1])                 # no tomorrow for the last day


def test_splits_are_chronological_and_do_not_overlap():
    splits, info = build_dataset(_cfg())
    tr, va, te = (splits[s].meta for s in ("train", "val", "test"))
    train_end, val_end = pd.Timestamp(info["train_end"]), pd.Timestamp(info["val_end"])
    assert tr["target_date"].max() <= train_end < va["target_date"].min()
    assert va["target_date"].max() <= val_end < te["target_date"].min()
    
    # No validation window may include days whose target was in training:
    # the earliest input day of any val window must be on/after train_end.
    seq_len = splits["val"].X.shape[1]
    for _, g in va.groupby("ticker"):
        first_end_day = g["date"].min()
        assert (first_end_day - train_end).days >= seq_len - 1


def test_scaler_uses_training_data_only():
    splits, _ = build_dataset(_cfg())
    last_day = splits["train"].X[:, -1, :]          # each training row appears once as a last day
    assert np.allclose(last_day.mean(axis=0), 0, atol=0.1)
    assert np.allclose(last_day.std(axis=0), 1, atol=0.1)


def test_stock_and_market_runs_use_identical_samples():
    a, _ = build_dataset(_cfg(use_market=False))
    b, _ = build_dataset(_cfg(use_market=True))
    for s in ("train", "val", "test"):
        pd.testing.assert_frame_equal(a[s].meta, b[s].meta)
        assert b[s].X.shape[2] > a[s].X.shape[2]
