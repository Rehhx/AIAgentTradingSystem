import numpy as np
import pandas as pd


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Volatility regime normalization swing long (realized-vol proxy for VIX term structure).

    Entry rule (daily, long-only):
      1. Short-window annualized realized vol (VIX proxy) has touched at least
         `vix_min_recent_high` (in vol points, e.g. 22 -> 0.22) at some point
         within the past `vix_high_lookback_days` sessions.
      2. The ratio of short-window vol to long-window vol (VIX/VIX3M proxy)
         has been below `ts_ratio_threshold` for `ts_cross_confirmation_days`
         consecutive sessions (contango regime).
      3. The same ratio was at or above the threshold (backwardation) at some
         point in the prior `vix_high_lookback_days` window, confirming a flip.
      Trigger fires only on the first bar that all three conditions become
      true (de-duped via persistence diff) to avoid retriggering.

    Exit rule: position is held for `hold_days` daily bars after the trigger,
    or exits early if cumulative return from entry close falls to
    -`stop_loss_pct`. Intraday bars inherit the daily position via
    date-alignment, producing a piecewise-constant 0/1 series.
    """
    vix_min_pct = float(params.get('vix_min_recent_high', 22)) / 100.0
    vix_lookback = int(params.get('vix_high_lookback_days', 10))
    ts_threshold = float(params.get('ts_ratio_threshold', 1.0))
    ts_confirm = int(params.get('ts_cross_confirmation_days', 2))
    hold_days = int(params.get('hold_days', 8))
    stop_pct = float(params.get('stop_loss_pct', 0.04))
    short_w = int(params.get('short_vol_window', 20))
    long_w = int(params.get('long_vol_window', 63))

    if len(df) == 0:
        return pd.Series(0, index=df.index, dtype=int)

    # Collapse to one observation per session
    date_idx = df.index.normalize()
    daily_close = df['close'].groupby(date_idx).last().sort_index()

    min_needed = max(long_w, vix_lookback) + ts_confirm + 2
    if len(daily_close) < min_needed:
        return pd.Series(0, index=df.index, dtype=int)

    daily_ret = np.log(daily_close / daily_close.shift(1))

    short_mp = max(5, short_w // 2)
    long_mp = max(10, long_w // 2)
    short_vol = daily_ret.rolling(short_w, min_periods=short_mp).std() * np.sqrt(252.0)
    long_vol = daily_ret.rolling(long_w, min_periods=long_mp).std() * np.sqrt(252.0)

    ts_ratio = short_vol / long_vol.replace(0, np.nan)

    # (1) Short-vol elevated at some point in recent window
    elevated_recent = (short_vol >= vix_min_pct).fillna(False).rolling(
        vix_lookback, min_periods=1
    ).max().fillna(0).astype(bool)

    # (2) Contango persistence
    in_contango = (ts_ratio < ts_threshold).fillna(False)
    contango_streak = in_contango.rolling(ts_confirm, min_periods=ts_confirm).sum() >= ts_confirm
    contango_streak = contango_streak.fillna(False)

    # (3) Prior backwardation (look earlier than current contango window)
    prior_backwardation = (ts_ratio.shift(ts_confirm) >= ts_threshold).fillna(False).rolling(
        vix_lookback, min_periods=1
    ).max().fillna(0).astype(bool)

    trigger = (elevated_recent & contango_streak & prior_backwardation).fillna(False)
    new_trigger = trigger & ~trigger.shift(1).fillna(False)

    # Daily position with hold + stop-loss
    n = len(daily_close)
    pos_arr = np.zeros(n, dtype=int)
    trig_arr = new_trigger.values
    close_arr = daily_close.values

    days_remaining = 0
    entry_price = np.nan
    for i in range(n):
        if days_remaining > 0:
            cur = close_arr[i]
            if not np.isnan(entry_price) and not np.isnan(cur) and \
               (cur / entry_price - 1.0) <= -stop_pct:
                pos_arr[i] = 0
                days_remaining = 0
                entry_price = np.nan
            else:
                pos_arr[i] = 1
                days_remaining -= 1
                if days_remaining == 0:
                    entry_price = np.nan
        elif trig_arr[i]:
            pos_arr[i] = 1
            days_remaining = max(0, hold_days - 1)
            entry_price = close_arr[i]
            if days_remaining == 0:
                entry_price = np.nan

    daily_pos = pd.Series(pos_arr, index=daily_close.index, dtype=int)

    # Broadcast daily positions back to original (possibly intraday) index
    mapped = daily_pos.reindex(date_idx).fillna(0).astype(int)
    sig = pd.Series(mapped.values, index=df.index, dtype=int)
    return sig
