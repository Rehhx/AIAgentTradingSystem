Looking at this spec, I need to implement a PEAD-style strategy. Since the function only receives OHLCV (no earnings calendar), I'll use a proxy: abnormally large standardized one-day returns confirmed by volume, then a fixed multi-day hold — a mechanism distinct from momentum (multi-bar return), gap-fade strategies (opposite side), and trend_ride (EMA breakout). The fixed hold provides hysteresis to control turnover.import pandas as pd
import numpy as np


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Post-Earnings Announcement Drift (PEAD) swing proxy.

    Entry rule: on a daily timeframe, look for a 'surprise' bar where the
    one-day return is large relative to its trailing rolling std (a SUE
    proxy because we have no fundamentals here) AND the daily volume is
    abnormally high (z-score > vol_z_min vs. its trailing window). The
    direction of the surprise (sign of the standardized return) sets the
    side: positive surprise -> long, negative surprise -> short.

    Hold rule: fixed hold of `hold_days` trading days after entry,
    providing hysteresis so noise doesn't churn the position. An ATR-based
    catastrophic stop (`stop_atr_mult` * ATR vs. entry close) exits early
    on large adverse moves. Only one position at a time; re-entries are
    allowed once flat.

    All signals are computed on a daily resample of the input frame and
    then forward-filled back to the original frequency with a one-bar
    lag, so intraday bars never see same-day daily-close data.
    """
    sue_entry = float(params.get('sue_entry', 1.5))
    hold_days = int(params.get('hold_days', 12))
    sigma_lookback = int(params.get('sigma_lookback', 60))
    vol_lookback = int(params.get('vol_lookback', 60))
    vol_z_min = float(params.get('vol_z_min', 1.0))
    atr_period = int(params.get('atr_period', 14))
    stop_atr_mult = float(params.get('stop_atr_mult', 3.0))
    side = str(params.get('side', 'both')).lower()

    if len(df) == 0:
        return pd.Series([], index=df.index, dtype=int)

    # Resample to daily; for already-daily data this is effectively a passthrough.
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['close'])

    warmup = max(sigma_lookback, vol_lookback, atr_period) + 2
    if len(daily) < warmup:
        return pd.Series(0, index=df.index, dtype=int)

    close = daily['close']
    high = daily['high']
    low = daily['low']
    vol = daily['volume']

    ret = close.pct_change()

    mp_sig = min(max(20, sigma_lookback // 2), sigma_lookback)
    mp_vol = min(max(20, vol_lookback // 2), vol_lookback)

    ret_std = ret.rolling(sigma_lookback, min_periods=mp_sig).std()
    sue = ret / ret_std.replace(0, np.nan)

    vol_mean = vol.rolling(vol_lookback, min_periods=mp_vol).mean()
    vol_std = vol.rolling(vol_lookback, min_periods=mp_vol).std()
    vol_z = (vol - vol_mean) / vol_std.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()

    allow_long = side in ('both', 'long')
    allow_short = side in ('both', 'short')

    long_trigger = (sue > sue_entry) & (vol_z > vol_z_min)
    short_trigger = (sue < -sue_entry) & (vol_z > vol_z_min)
    if not allow_long:
        long_trigger = pd.Series(False, index=daily.index)
    if not allow_short:
        short_trigger = pd.Series(False, index=daily.index)

    # Entry trigger detected at close of day t executes next session (t+1).
    long_entry = long_trigger.shift(1).fillna(False).to_numpy()
    short_entry = short_trigger.shift(1).fillna(False).to_numpy()

    close_arr = close.to_numpy()
    atr_arr = atr.to_numpy()
    n = len(daily)
    pos = np.zeros(n, dtype=np.int8)

    direction = 0
    bars_remaining = 0
    entry_price = np.nan

    for i in range(n):
        # Stop / time exit for currently held position
        if direction != 0:
            stopped = False
            if not np.isnan(entry_price) and not np.isnan(atr_arr[i]):
                if direction > 0 and close_arr[i] <= entry_price - stop_atr_mult * atr_arr[i]:
                    stopped = True
                elif direction < 0 and close_arr[i] >= entry_price + stop_atr_mult * atr_arr[i]:
                    stopped = True
            if stopped or bars_remaining <= 0:
                direction = 0
                bars_remaining = 0
                entry_price = np.nan

        # Fresh entry only when flat
        if direction == 0:
            if long_entry[i] and not np.isnan(atr_arr[i]):
                direction = 1
                bars_remaining = hold_days
                entry_price = close_arr[i]
            elif short_entry[i] and not np.isnan(atr_arr[i]):
                direction = -1
                bars_remaining = hold_days
                entry_price = close_arr[i]

        pos[i] = direction
        if direction != 0:
            bars_remaining -= 1

    daily_pos = pd.Series(pos, index=daily.index, dtype=int)

    # Shift one bar forward so a daily position computed using close[t]
    # is only applied from bar t+1 onward — eliminates lookahead at the
    # intraday level when ffilled.
    daily_pos_safe = daily_pos.shift(1).fillna(0).astype(int)

    out = daily_pos_safe.reindex(df.index, method='ffill').fillna(0).astype(int)
    return out
