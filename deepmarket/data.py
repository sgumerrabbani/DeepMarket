"""Loading the  OHLCV prices into one tidy table.


Every loader output is a datframe with hese columns:
--> date (datetime64), ticker (str), open, high, low, close, volume

Using three sources as our premise allows us to run the same code on a laptop, a server (yfinance), or a cluster (csv) without changing the code:

yfinance  - downloads from Yahoo Finance and caches to data/raw/
    csv       - reads a local CSV (e.g. the Aethereal S&P 500 dataset)
    synthetic - random prices, only for testing the code offline

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
RAW_DIR = Path("data/raw")

# Different datasets name their columns differently; map them to ours.
_ALIASES = {
    "date": "date", "datetime": "date", "timestamp": "date",
    "ticker": "ticker", "symbol": "ticker", "name": "ticker",
    "open": "open", "high": "high", "low": "low",
    "close": "close", "volume": "volume",
}


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and invalid rows, sort by ticker then date."""
    df = df[COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna()
    # Invalid prices: non-positive values or a high below the low.
    valid = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["high"] >= df["low"]) & (df["volume"] >= 0)
    df = df[valid]
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_csv(path: str | Path, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: _ALIASES[c.lower()] for c in df.columns if c.lower() in _ALIASES})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} is missing columns {missing}; found {list(df.columns)}")
    df = df[df["ticker"].isin(tickers)]
    df = clean_prices(df)
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def load_yfinance(tickers: list[str], start: str, end: str, cache_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Download each ticker once and cache it as data/raw/<TICKER>.csv."""
    import yfinance as yf  # imported here so other sources don't need it

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        cache = cache_dir / f"{t}.csv"
        if cache.exists():
            df = pd.read_csv(cache, parse_dates=["date"])
        else:
            print(f"  downloading {t} from Yahoo Finance...")
            # auto_adjust=True gives split/dividend adjusted prices.
            raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                raise RuntimeError(f"No data returned for {t}")
            if isinstance(raw.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
                raw.columns = raw.columns.get_level_values(0)
            df = raw.reset_index().rename(columns=str.lower)
            df["ticker"] = t
            df = df[COLUMNS]
            df.to_csv(cache, index=False)
        frames.append(df)
    df = clean_prices(pd.concat(frames, ignore_index=True))
    return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def load_synthetic(tickers: list[str], start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Random-walk prices. There is no real signal here, so models should score about 0.5."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for t in tickers:
        ret = rng.normal(0.0003, 0.015, len(dates))
        close = 100 * np.exp(np.cumsum(ret))
        open_ = close * np.exp(rng.normal(0, 0.005, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, len(dates))))
        vol = rng.lognormal(15, 0.3, len(dates))
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "open": open_, "high": high,
                                    "low": low, "close": close, "volume": vol}))
    return clean_prices(pd.concat(frames, ignore_index=True))


def load_prices(cfg: dict, tickers: list[str]) -> pd.DataFrame:
    """Dispatch to the loader named in cfg['data']['source']."""
    d = cfg["data"]
    source = d["source"]
    if source == "yfinance":
        return load_yfinance(tickers, d["start"], d["end"])
    if source == "csv":
        return load_csv(d["csv_path"], tickers, d["start"], d["end"])
    if source == "synthetic":
        return load_synthetic(tickers, d["start"], d["end"], seed=cfg["train"]["seed"])
    raise ValueError(f"Unknown data source: {source}")
