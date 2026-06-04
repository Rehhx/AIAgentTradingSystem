"""
ml/features.py
--------------
Trailing-only feature engineering — every column is computed from data at or
before its own timestamp (rolling windows, lagged returns), so the feature matrix
carries no look-ahead. These are the same families of signal the rule-based book
uses (momentum, trend distance, mean-reversion oscillators, realized vol), handed
to the model so the honest question is: does ML combine them better than equal
weight? (Spoiler in runners/ml_signal.py: not meaningfully, out-of-sample.)
"""
from __future__ import annotations

import pandas as pd

from agents.daily_strategies import _rsi


def make_features(close: pd.Series) -> pd.DataFrame:
    c = close.astype(float)
    ret = c.pct_change()
    f = pd.DataFrame(index=c.index)

    # momentum / trailing returns over several horizons
    f["ret_1"] = ret
    f["ret_5"] = c.pct_change(5)
    f["ret_21"] = c.pct_change(21)
    f["ret_63"] = c.pct_change(63)
    f["ret_126"] = c.pct_change(126)
    f["mom_12_1"] = c.shift(21) / c.shift(21 + 126) - 1     # 12-1 momentum (skip 1m)

    # realized volatility
    f["vol_20"] = ret.rolling(20).std()
    f["vol_60"] = ret.rolling(60).std()

    # trend distance
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["px_sma50"] = c / sma50 - 1
    f["px_sma200"] = c / sma200 - 1
    f["sma50_200"] = sma50 / sma200 - 1

    # mean-reversion oscillators
    f["rsi_2"] = _rsi(c, 2)
    f["rsi_14"] = _rsi(c, 14)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_z"] = (c - mid) / sd

    return f
