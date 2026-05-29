"""
vix_term_structure_equity_carry
-------------------------------

Long-only regime filter that mimics a VIX term-structure carry switch using
realized-volatility surrogates derived from the price series itself.

Mechanism (different from VRP short-vol, momentum, or trend strategies):
  * Compute three realized-vol horizons on close-to-close log returns:
      - short  : ~9 bars  (VIX9D analog)
      - mid    : ~21 bars (VIX analog)
      - long   : ~63 bars (VIX3M analog)
  * "Contango" proxy: short/mid and mid/long ratios are BOTH below a
    threshold (default 0.95), meaning the near-term realized vol is
    materially calmer than the longer-term envelope.
  * When the regime is "calm contango", hold a long position; otherwise flat.
  * A multi-bar smoothing of the raw regime flag plus a minimum-hold lock
    prevent whipsaw flips, so the position persists across noise and
    realizes the equity-premium drift conditional on the calm vol regime.

Entry  : smoothed regime flag = 1
Exit   : smoothed regime flag = 0 after minimum hold has elapsed
Output : pd.Series of int in {0, 1} aligned to df.index (long-only)
"""

import numpy as np
import pandas as pd


def signals(df: pd.DataFrame, params: dict) -> pd.Series:
    short_win = int(params.get("short_window", 9))
    mid_win = int(params.get("mid_window", 21))
    long_win = int(params.get("long_window", 63))
    ratio_short = float(params.get("ratio_short", 0.95))
    ratio_long = float(params.get("ratio_long", 0.95))
    smoothing = int(params.get("regime_smoothing_days", 2))
    min_hold = int(params.get("min_hold_days", 5))
    vol_floor = float(params.get("vol_floor", 1e-6))

    close = df["close"].astype(float)

    # Log returns; first value is NaN.
    log_ret = np.log(close).diff()

    # Realized-vol surrogates at three horizons. The annualization constant
    # cancels in the ratios, so we leave it out.
    rv_short = log_ret.rolling(window=short_win, min_periods=short_win).std()
    rv_mid = log_ret.rolling(window=mid_win, min_periods=mid_win).std()
    rv_long = log_ret.rolling(window=long_win, min_periods=long_win).std()

    rv_short = rv_short.clip(lower=vol_floor)
    rv_mid = rv_mid.clip(lower=vol_floor)
    rv_long = rv_long.clip(lower=vol_floor)

    ratio_sm = rv_short / rv_mid
    ratio_ml = rv_mid / rv_long

    # Raw regime: both ratios indicate calm/contango.
    raw_regime = ((ratio_sm < ratio_short) & (ratio_ml < ratio_long)).astype(float)
    raw_regime = raw_regime.fillna(0.0)

    # Smoothing: require the regime to hold for `smoothing` consecutive bars
    # before we honor it.
    if smoothing > 1:
        smoothed = (
            raw_regime.rolling(window=smoothing, min_periods=smoothing)
            .min()
            .fillna(0.0)
        )
    else:
        smoothed = raw_regime

    flag_arr = smoothed.fillna(0.0).astype(int).to_numpy()

    # Minimum-hold hysteresis: once long, hold for at least min_hold bars
    # even if the smoothed flag dips briefly. Refresh hold while regime
    # persists. This keeps trades in the tens-to-low-hundreds range.
    n = len(flag_arr)
    pos = np.zeros(n, dtype=int)
    hold_left = 0
    warmup = max(short_win, mid_win, long_win)
    for i in range(n):
        if i < warmup:
            pos[i] = 0
            continue
        if hold_left > 0:
            pos[i] = 1
            hold_left -= 1
            if flag_arr[i] == 1:
                hold_left = max(hold_left, min_hold - 1)
            continue
        if flag_arr[i] == 1:
            pos[i] = 1
            hold_left = max(min_hold - 1, 0)
        else:
            pos[i] = 0

    return pd.Series(pos, index=df.index, dtype=int)
