"""breadth_thrust_persistence_long

Constructs a within-session breadth proxy from intraday OHLCV bars: the
share of volume traded on up-ticking bars (close > open) versus the sum of
up-volume and down-volume each session. Sessions where this up-volume share
exceeds a high threshold (default 0.88) are labelled 'breadth thrusts',
intended as a proxy for advancing-volume dominance in the underlying basket.

Entry: on the open of the session AFTER a thrust day, go long.
Hold: position is carried for ``hold_days`` sessions (default 15). A fresh
thrust observed while already long resets the hold clock (re-leverage on
double-thrust within ``lookback_for_double_thrust_days``).
Exit: either (a) hold-period expires, or (b) the session low pierces an
ATR-based trailing stop set at entry (entry_close - stop_atr_mult * ATR14).
After exiting, no new entry is taken for ``min_days_between_signals``
sessions, which keeps round-trip count modest and avoids churn.

Long-only. Signal is held flat across all intraday bars of a held session
so the engine does not over-trade.
"""

import numpy as np
import pandas as pd


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    up_vol_thresh = float(params.get("up_volume_ratio_threshold", 0.88))
    double_lookback = int(params.get("lookback_for_double_thrust_days", 10))
    hold_days = int(params.get("hold_days", 15))
    stop_atr_mult = float(params.get("stop_atr_mult", 2.5))
    min_gap_days = int(params.get("min_days_between_signals", 20))
    atr_period = int(params.get("atr_period", 14))

    if df is None or len(df) == 0:
        return pd.Series([], dtype=int)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float).fillna(0.0)

    # Per-bar directional volume
    up_bar = (close > open_).astype(float)
    down_bar = (close < open_).astype(float)
    up_vol_bar = vol * up_bar
    down_vol_bar = vol * down_bar

    # Aggregate to daily sessions
    day_key = pd.Series(df.index.normalize(), index=df.index)
    daily = pd.DataFrame({
        "up_vol": up_vol_bar.groupby(day_key).sum(),
        "down_vol": down_vol_bar.groupby(day_key).sum(),
        "high": high.groupby(day_key).max(),
        "low": low.groupby(day_key).min(),
        "close": close.groupby(day_key).last(),
        "open": open_.groupby(day_key).first(),
    }).sort_index()

    if len(daily) < max(atr_period + 2, double_lookback + 2):
        return pd.Series(0, index=df.index, dtype=int)

    denom = (daily["up_vol"] + daily["down_vol"]).replace(0.0, np.nan)
    up_ratio = (daily["up_vol"] / denom).fillna(0.0)

    # Daily ATR (Wilder-style simple mean of true range)
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()

    # Thrust events
    is_thrust = (up_ratio >= up_vol_thresh).fillna(False).values
    thrust_count_window = (
        pd.Series(is_thrust.astype(int), index=daily.index)
        .rolling(double_lookback, min_periods=1)
        .sum()
        .values
    )
    is_double = (thrust_count_window >= 2) & is_thrust

    daily_close = daily["close"].values
    daily_low = daily["low"].values
    daily_atr = atr.values

    n_days = len(daily)
    daily_pos = np.zeros(n_days, dtype=np.int8)

    in_pos = False
    entry_price = np.nan
    entry_atr = np.nan
    days_held = 0
    last_exit_idx = -(10 ** 9)

    for i in range(n_days):
        if in_pos:
            # Re-thrust extends the hold clock
            if is_thrust[i] or is_double[i]:
                days_held = 0
            else:
                days_held += 1

            stop_level = entry_price - stop_atr_mult * entry_atr
            stop_hit = (
                not np.isnan(stop_level)
                and not np.isnan(daily_low[i])
                and daily_low[i] <= stop_level
            )

            if stop_hit or days_held >= hold_days:
                in_pos = False
                entry_price = np.nan
                entry_atr = np.nan
                days_held = 0
                last_exit_idx = i
                daily_pos[i] = 1  # mark current day as still held; exit applies at close
            else:
                daily_pos[i] = 1
        else:
            cooled_down = (i - last_exit_idx) >= min_gap_days
            atr_ok = not np.isnan(daily_atr[i]) and daily_atr[i] > 0
            if is_thrust[i] and cooled_down and atr_ok:
                in_pos = True
                entry_price = daily_close[i]
                entry_atr = daily_atr[i]
                days_held = 0
                # Entry is taken from the NEXT session; do not set today.
                daily_pos[i] = 0
            else:
                daily_pos[i] = 0

    daily_signal = pd.Series(daily_pos, index=daily.index, dtype=int)
    # Shift by 1 session so we act on the day after the thrust observation
    daily_signal = daily_signal.shift(1).fillna(0).astype(int)

    # Broadcast the daily position to every intraday bar of that session
    out = daily_signal.reindex(df.index.normalize(), fill_value=0)
    out.index = df.index
    return out.astype(int)
