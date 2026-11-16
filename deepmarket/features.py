"""
Turning these raw prices into model features and the UP / NOT-UP label.

Every feature on day t is computed from the stock's own prices on days t, t-1, t-2, ... it
never looks prices in the future. Only the label looks ahead, because it is the thing we are trying to predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Method A: five daily stock features.
STOCK_FEATURES = ["ret_1", "hl_range", "oc_ret", "vol_chg", "mom_5"]


def add_stock_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker features. `prices` is the tidy table from data.py."""
    out = []
    for _, g in prices.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]
        g["ret_1"] = np.log(close / close.shift(1))            # lagged daily log return
        g["hl_range"] = (g["high"] - g["low"]) / close           # intraday price range
        g["oc_ret"] = close / g["open"] - 1                      # open-to-close move
        g["vol_chg"] = np.log((g["volume"] + 1) / (g["volume"].shift(1) + 1))  # volume change
        g["mom_5"] = close / close.shift(5) - 1                  # 5-day momentum
        
        # Label: 1 if TOMORROW's close is above today's close. shift(-1) looks
        # one day ahead, which is fine because this is the target, not an input.
        g["next_close"] = close.shift(-1)
        g["target_date"] = g["date"].shift(-1)
        g["label"] = (g["next_close"] > close).astype(float)
        g.loc[g["next_close"].isna(), "label"] = np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True)


def market_features(market_prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """SPY/QQQ return, momentum and rolling volatility, one row per date.

    Returns (wide table indexed by date, list of feature column names).
    """
    cols = {}
    for t, g in market_prices.groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        ret = np.log(g["close"] / g["close"].shift(1))
        key = t.lower()
        cols[f"{key}_ret_1"] = ret
        cols[f"{key}_mom_5"] = g["close"] / g["close"].shift(5) - 1
        # rolling(10) on day t only uses days t-9..t, so no look-ahead.
        cols[f"{key}_vol_10"] = ret.rolling(10).std()
    wide = pd.DataFrame(cols)
    return wide, list(wide.columns)


def build_feature_table(stock_prices: pd.DataFrame,
                        market_prices: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join stock features with market features on the same date.

    We always build BOTH feature sets and drop rows where either is missing.
    That way the stock-only run (Method A) and the stock+market run
    (Method B) see exactly the same samples and dates, so the only thing
    that changes between them is the extra columns. That is what makes the
    comparison a controlled experiment.

    Returns (table, stock feature names, market feature names).
    """
    df = add_stock_features(stock_prices)
    market_cols: list[str] = []
    if market_prices is not None and not market_prices.empty:
        wide, market_cols = market_features(market_prices)
        # Left join on the stock's own trading date: each stock row gets the
        # market values from that same day (never a later day).
        df = df.merge(wide, left_on="date", right_index=True, how="left")
    # Drop warm-up rows where rolling features are not defined yet.
    df = df.dropna(subset=STOCK_FEATURES + market_cols).reset_index(drop=True)
    return df, list(STOCK_FEATURES), market_cols
