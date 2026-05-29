"""
capitulation_close_reversal_swing
---------------------------------

Long-only swing strategy that fires on a daily "distribution close" pattern:
    - Close in the bottom N% of the day's range (close_location <= close_location_max_pct)
    - Volume >= volume_multiple * trailing volume_lookback-day average
    - Day's range (high - low) >= range_atr_mult_min * ATR(atr_period)

When all three conditions hit on bar t, we enter long at bar t (signal=1) and
hold for hold_days subsequent bars or until a profit target / stop loss based
on ATR is touched (using subsequent highs/lows). Persistence is enforced by
the holding window — no flipping every bar — and re-entries while a position
is open are suppressed, keeping turnover to roughly tens of round-trips per
year on a typical equity index.
"""

import numpy as np
import pandas as pd


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    close_loc_max = float(params.get("close_location_max_pct", 0.10))
    vol_mult = float(params.get("volume_multiple", 1.7))
    vol_lookback = int(params.get("volume_lookback", 20))
    range_atr_mult = float(params.get("range_atr_mult_min", 1.5))
    atr_period = int(params.get("atr_period", 14))
    hold_days = int(params.get("hold_days", 5))
    pt_atr = float(params.get("profit_target_atr", 2.0))
    sl_atr = float(params.get("stop_loss_atr", 1.5))

    if df.empty:
        return pd.Series(0, index=df.index, dtype=int)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    day_range = (high - low).replace(0.0, np.nan)
    close_location = (close - low) / day_range  # 0 = at low, 1 = at high
    avg_vol = volume.rolling(window=vol_lookback, min_periods=vol_lookback).mean()
    vol_ratio = volume / avg_vol.replace(0.0, np.nan)
    atr = _atr(high, low, close, atr_period)
    range_atr_ratio = (high - low) / atr.replace(0.0, np.nan)

    trigger = (
        (close_location <= close_loc_max)
        & (vol_ratio >= vol_mult)
        & (range_atr_ratio >= range_atr_mult)
        & atr.notna()
        & avg_vol.notna()
    ).fillna(False)

    sig = np.zeros(len(df), dtype=int)
    n = len(df)
    highs = high.to_numpy()
    lows = low.to_numpy()
    closes = close.to_numpy()
    atrs = atr.to_numpy()
    trig = trigger.to_numpy()

    in_position_until = -1
    entry_idx = -1
    entry_price = np.nan
    pt_level = np.nan
    sl_level = np.nan

    for i in range(n):
        if i <= in_position_until:
            # Currently holding — check exit conditions for THIS bar
            exited = False
            if not np.isnan(pt_level) and highs[i] >= pt_level:
                exited = True
            elif not np.isnan(sl_level) and lows[i] <= sl_level:
                exited = True

            if exited:
                # Close position on this bar (signal becomes 0 from next bar)
                sig[i] = 0
                in_position_until = -1
                entry_idx = -1
                entry_price = np.nan
                pt_level = np.nan
                sl_level = np.nan
                continue

            # Otherwise stay long
            sig[i] = 1
            # End-of-window forced exit handled by loop boundary
            continue

        # Not in position — look for fresh entry
        if trig[i] and not np.isnan(atrs[i]):
            entry_idx = i
            entry_price = closes[i]
            pt_level = entry_price + pt_atr * atrs[i]
            sl_level = entry_price - sl_atr * atrs[i]
            in_position_until = min(n - 1, i + hold_days)
            sig[i] = 1

    out = pd.Series(sig, index=df.index, dtype=int)
    return out
