"""
tsmom_cross_asset_etf
=====================

Single-asset adaptation of Moskowitz/Ooi/Pedersen (2012) time-series momentum.

Mechanism
---------
On each monthly rebalance date (first available bar of a new calendar month):
  * Compute the trailing ``lookback_months`` total return of close-to-close.
  * If trailing return > entry_band : target position = +1 (long).
  * If trailing return < -entry_band : target position = -1 (short).
  * Otherwise: flat (0).
The target position is then HELD constant until the next monthly rebalance, so
the strategy trades at most once per calendar month (~12 round-trips/yr).

A realized-volatility gate optionally suppresses signals when annualized vol
exceeds ``vol_cap`` — this proxies the vol-targeting step (very high vol would
shrink notional toward zero in the original paper).

This is distinct from the 50/200 SMA crossover (``daily_trend_5020``), the
Donchian channel breakout, and the breadth/regime strategies already in the
book: the signal here is purely the sign of an N-month excess return at a
monthly cadence, not a moving-average state nor a range break.
"""

import numpy as np
import pandas as pd


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    lookback_months = int(params.get("lookback_months", 12))
    bars_per_month = int(params.get("bars_per_month", 21))
    vol_lookback_days = int(params.get("vol_lookback_days", 60))
    vol_target = float(params.get("vol_target", 0.10))
    vol_cap = float(params.get("vol_cap", 0.60))
    entry_band = float(params.get("entry_band", 0.0))
    annualization = float(params.get("annualization_factor", 252.0))

    close = df["close"].astype(float)
    n = len(close)

    lookback_bars = max(1, lookback_months * bars_per_month)

    # Trailing total return over the lookback window (e.g. ~252 bars for 12m).
    trailing_ret = close.pct_change(lookback_bars)

    # Realized annualized vol from daily log returns.
    log_ret = np.log(close).diff()
    realized_vol = log_ret.rolling(
        window=vol_lookback_days,
        min_periods=vol_lookback_days,
    ).std() * np.sqrt(annualization)

    # Raw monthly-rebalanced target: sign of trailing return, gated by band.
    raw = pd.Series(0, index=close.index, dtype="int64")
    raw = raw.where(~(trailing_ret > entry_band), 1)
    raw = raw.where(~(trailing_ret < -entry_band), -1)

    # Suppress when vol is extreme (vol target would set size ~ 0) or when
    # the lookback window is not yet filled.
    valid = trailing_ret.notna() & realized_vol.notna() & (realized_vol <= vol_cap)
    raw = raw.where(valid, 0)

    # Only refresh the held position on the first bar of each new calendar
    # month — that's the "rebalance monthly" rule. Between rebalances we
    # carry the prior month's target. This keeps round-trips at ~12/yr.
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)

    month_key = idx.year.astype("int64") * 12 + idx.month.astype("int64")
    is_new_month = pd.Series(
        np.concatenate([[True], month_key[1:] != month_key[:-1]]),
        index=df.index,
    )

    held = raw.where(is_new_month, other=np.nan).ffill()
    held = held.fillna(0).astype(int)

    # Vol-target proportionality: if realized vol drops well below target,
    # we'd lever up, but our output is constrained to {-1, 0, 1}. If realized
    # vol is so high that the targeted notional would be below ``min_notional``
    # of full size, drop the signal to 0 to avoid noisy near-zero positions.
    min_notional = float(params.get("min_notional_frac", 0.25))
    if min_notional > 0:
        sized = (vol_target / realized_vol).clip(upper=1.0)
        sized = sized.reindex(df.index).ffill()
        too_small = sized < min_notional
        held = held.where(~too_small.fillna(False), 0)

    held = held.clip(-1, 1).astype(int)
    held.iloc[:max(lookback_bars, vol_lookback_days)] = 0
    return held
