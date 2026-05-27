"""
agents/backtesting_agent.py
----------------------------
real backtesting agent. loads 1m parquet data, runs strategy signals,
computes actual performance metrics.

strategies implemented:
  - RSI mean reversion
  - VWAP reversion
  - Opening range breakout (ORB)
  - Momentum / price continuation
  - Bollinger band squeeze
  - EMA crossover

each backtest also segments performance by market regime
using the vector_stores regime store.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_ticker, available_tickers, DATA_DIR

log = logging.getLogger("backtesting_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/backtesting.log"),
    ],
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
COMMISSION  = 0.0001   # 0.01% per trade — Alpaca is commission-free; this
                       # captures FINRA TAF + SEC fees + residual market impact
                       # at retail size. round-trip with slippage = ~6 bps total.
SLIPPAGE    = 0.0002   # 0.02% slippage per fill
INITIAL_CAP = 100_000  # $100k starting capital (matches alpaca paper account)

# risk / execution controls (apply to every strategy via the backtest engine)
ATR_STOP_MULT  = 1.5   # exit when adverse move >= 1.5 * 14-bar ATR
ATR_PERIOD     = 14    # bars for the ATR used by the stop
REENTRY_COOLDOWN_BARS = 5   # min bars between exit and next entry

# regime classifier — mirrors RegimeStore._label_regime exactly, vectorized so
# we can label every bar's trailing 60-bar window without OpenAI API calls.
REGIME_WINDOW_BARS = 60


# ---------------------------------------------------------------------------
# indicator library
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def vwap(df: pd.DataFrame) -> pd.Series:
    """intraday VWAP — resets each trading day"""
    df = df.copy()
    df["date"]      = df.index.normalize()
    df["typical"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"]    = df["typical"] * df["volume"]
    df["cum_tpvol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"]   = df.groupby("date")["volume"].cumsum()
    return df["cum_tpvol"] / df["cum_vol"]


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0):
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    return mid - std * sigma, mid, mid + std * sigma


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    mean = volume.rolling(period).mean()
    std  = volume.rolling(period).std()
    return (volume - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# regime classifier (local, vectorized — same thresholds as RegimeStore)
# ---------------------------------------------------------------------------

def regime_label_series(df: pd.DataFrame, window: int = REGIME_WINDOW_BARS) -> pd.Series:
    """
    label each bar by the regime of its trailing `window` bars.
    thresholds mirror RegimeStore._label_regime so labels are consistent
    with whatever is stored in the vector DB.
    """
    close   = df["close"]
    returns = close.pct_change()

    trend = (close - close.shift(window)) / close.shift(window)
    vol   = returns.rolling(window).std()

    labels = pd.Series("unknown", index=df.index)
    abs_t  = trend.abs()
    labels[(abs_t > 0.005) & (vol < 0.002)] = "trending"
    labels[(abs_t < 0.001) & (vol < 0.001)] = "chop"
    labels[(vol > 0.003)]                    = "breakout"
    # anything still "unknown" after the cascade is mean-reversion
    labels[labels == "unknown"] = "mean_reversion"
    # bars before the window is full have no defined regime
    labels.iloc[:window] = "unknown"
    return labels


# ---------------------------------------------------------------------------
# market-wide regime classifier (bull / bear / high_vol / neutral)
# ---------------------------------------------------------------------------
# this is a coarser-grained classifier than regime_label_series above.
# regime_label_series tells us what the *intraday window* looks like for a
# given ticker; compute_market_regime_series tells us what the *broad market
# environment* is on a given calendar day. strategies can opt into either or
# both via STRATEGY_REGIME_AFFINITY (intraday) and STRATEGY_MARKET_AFFINITY
# (market-wide).

def compute_market_regime_series(
    spy_df: pd.DataFrame,
    trend_lookback_days: int  = 50,
    vol_lookback_days:   int  = 20,
    bull_trend_threshold: float = 0.05,
    bear_trend_threshold: float = -0.05,
    high_vol_threshold:   float = 0.25,
) -> pd.Series:
    """
    classify each trading day as bull / bear / high_vol / neutral using SPY:
      - bull       : 50-day return > +5% AND realized vol <= 25%
      - bear       : 50-day return < -5% AND realized vol <= 25%
      - high_vol   : 20-day realized vol > 25%
      - neutral    : everything else

    high_vol overrides direction — chaotic markets don't behave like trends.
    returns a daily Series of strings indexed by calendar date.
    """
    daily = spy_df["close"].resample("1D").last().dropna()
    if len(daily) < trend_lookback_days + 1:
        return pd.Series("neutral", index=daily.index, dtype=object)

    daily_returns = daily.pct_change()
    trend         = (daily - daily.shift(trend_lookback_days)) / daily.shift(trend_lookback_days)
    realized_vol  = daily_returns.rolling(vol_lookback_days).std() * np.sqrt(252)

    regime = pd.Series("neutral", index=daily.index, dtype=object)
    bull_mask = (trend >  bull_trend_threshold) & (realized_vol <= high_vol_threshold)
    bear_mask = (trend <  bear_trend_threshold) & (realized_vol <= high_vol_threshold)
    vol_mask  = realized_vol > high_vol_threshold
    regime[bull_mask] = "bull"
    regime[bear_mask] = "bear"
    regime[vol_mask]  = "high_vol"   # vol overrides direction
    return regime


def market_regime_for_df(spy_df: pd.DataFrame, target_df: pd.DataFrame,
                         **classifier_kwargs) -> pd.Series:
    """
    classify the market on the *days* spanned by spy_df, then forward-fill the
    label onto every 1m bar of target_df. result is a Series aligned to
    target_df.index ready for run_backtest's market_regime_series parameter.
    """
    daily_regime = compute_market_regime_series(spy_df, **classifier_kwargs)
    return daily_regime.reindex(target_df.index, method="ffill").fillna("neutral")


# which market-wide regimes each strategy is allowed to fire in. opt-in:
# strategies not listed here have no market-regime filter applied.
STRATEGY_MARKET_AFFINITY = {
    # squeeze breakouts work best in trending markets; high vol noise tends to
    # produce false breakouts that whipsaw
    "bb_squeeze":             {"bull", "bear", "neutral"},
    # gap fades depend on retail panic — bigger gaps and bigger fades in vol
    "overnight_gap_fade":     {"bull", "bear", "neutral", "high_vol"},
    # extreme bar fades capture liquidity vacuums — these happen in any regime
    "extreme_bar_fade":       {"bull", "bear", "neutral", "high_vol"},
    # momentum strategies are pure trend plays — needs direction
    "momentum":               {"bull", "bear"},
    "ema_crossover":          {"bull", "bear"},
    # bollinger band touch reversion needs chop or normal vol — fails in trends
    "bb_band_touch_revert":   {"neutral", "high_vol"},
    "bb_band_touch_revert_v2":{"neutral", "high_vol"},
    # ORB and VWAP slope work when there's directional flow
    "orb":                    {"bull", "bear", "high_vol"},
    "vwap_slope_break":       {"bull", "bear", "high_vol"},
    # intraday seasonality should work in any regime if mechanism is real
    "half_hour_continuation": {"bull", "bear", "neutral", "high_vol"},
    # zarattini noise-area breakout — paper tests on SPY in all market modes
    "noise_area_breakout":    {"bull", "bear", "neutral", "high_vol"},
    # custom trend-ride is designed specifically for directional regimes
    "trend_ride":             {"bull", "bear"},
}


# which regimes each strategy is allowed to fire in. derived from the
# strategy's structural assumption (trend-following vs mean-reverting).
# RSI/VWAP narrowed to "chop" only — the "mean_reversion" bucket is the
# heuristic's fallback class and still contains too much directional drift
# for a snapback trade to work reliably.
STRATEGY_REGIME_AFFINITY = {
    "rsi_reversion":     {"chop"},
    "vwap_reversion":    {"chop"},
    "orb":               {"breakout", "trending"},
    "momentum":          {"trending", "breakout"},
    "bb_squeeze":        {"breakout"},
    "ema_crossover":     {"trending"},
    "overnight_gap_fade":{"mean_reversion", "chop"},
    "extreme_bar_fade":  {"mean_reversion", "chop", "breakout"},
    "vwap_slope_break":  {"trending", "breakout"},
    "bb_band_touch_revert": {"chop", "mean_reversion"},
    "bb_band_touch_revert_v2": {"chop", "mean_reversion"},
    # half-hour continuation operates on intraday seasonality across all regimes
    "half_hour_continuation": {"trending", "chop", "mean_reversion", "breakout"},
    "noise_area_breakout":   {"breakout", "trending", "mean_reversion"},
    "trend_ride":            {"trending", "breakout"},
}


# ---------------------------------------------------------------------------
# signal generators
# ---------------------------------------------------------------------------

def signals_rsi_reversion(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    long when RSI < oversold, exit when RSI > 50
    short when RSI > overbought, exit when RSI < 50
    returns: series of {1=long, -1=short, 0=flat}
    """
    period     = params.get("rsi_period", 14)
    oversold   = params.get("oversold", 30)
    overbought = params.get("overbought", 70)

    r       = rsi(df["close"], period)
    signal  = pd.Series(0, index=df.index)
    pos     = 0

    for i in range(period, len(df)):
        if pos == 0:
            if r.iloc[i] < oversold:
                pos = 1
            elif r.iloc[i] > overbought:
                pos = -1
        elif pos == 1 and r.iloc[i] > 50:
            pos = 0
        elif pos == -1 and r.iloc[i] < 50:
            pos = 0
        signal.iloc[i] = pos

    return signal


def signals_vwap_reversion(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    long when price > threshold% below VWAP
    short when price > threshold% above VWAP
    exit at VWAP touch or end of day
    """
    threshold = params.get("threshold_pct", 0.003)   # 0.3%
    v         = vwap(df)
    signal    = pd.Series(0, index=df.index)
    pos       = 0

    dates = df.index.normalize().unique()
    for date in dates:
        day_mask = df.index.normalize() == date
        day_idx  = df.index[day_mask]

        for i, ts in enumerate(day_idx):
            c  = df.loc[ts, "close"]
            vw = v.loc[ts]
            if pd.isna(vw) or vw == 0:
                continue
            pct_from_vwap = (c - vw) / vw

            if pos == 0:
                if pct_from_vwap < -threshold:
                    pos = 1
                elif pct_from_vwap > threshold:
                    pos = -1
            elif pos == 1:
                if pct_from_vwap >= 0:   # touched VWAP
                    pos = 0
            elif pos == -1:
                if pct_from_vwap <= 0:
                    pos = 0

            # close all at end of day
            if i == len(day_idx) - 1:
                pos = 0

            signal.loc[ts] = pos

    return signal


def signals_orb(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    opening range breakout — first N minutes define the range
    long on break above range high, short on break below range low
    target: 1x range size, stop: opposite side of range
    """
    orb_minutes = params.get("orb_minutes", 15)
    signal      = pd.Series(0, index=df.index)
    pos         = 0

    dates = df.index.tz_convert("America/New_York").normalize().unique()

    for date in dates:
        day_et  = df.index.tz_convert("America/New_York")
        day_mask = day_et.normalize() == date
        day_df   = df[day_mask]

        if len(day_df) < orb_minutes + 1:
            continue

        # define opening range
        orb_df   = day_df.iloc[:orb_minutes]
        orb_high = orb_df["high"].max()
        orb_low  = orb_df["low"].min()
        orb_size = orb_high - orb_low

        if orb_size <= 0:
            continue

        target_long  = orb_high + orb_size
        target_short = orb_low  - orb_size
        pos          = 0
        # at most one long and one short break per day — otherwise the strategy
        # re-fires every bar that price stays above orb_high after the target hits
        traded_long  = False
        traded_short = False

        intraday   = day_df.iloc[orb_minutes:]
        last_index = len(intraday) - 1

        for i, (ts, row) in enumerate(intraday.iterrows()):
            c = row["close"]

            if pos == 0:
                if c > orb_high and not traded_long:
                    pos = 1
                    traded_long = True
                elif c < orb_low and not traded_short:
                    pos = -1
                    traded_short = True
            elif pos == 1:
                if c >= target_long or c < orb_low:
                    pos = 0
            elif pos == -1:
                if c <= target_short or c > orb_high:
                    pos = 0

            # close EOD
            if i == last_index:
                pos = 0

            signal.loc[ts] = pos

            if traded_long and traded_short and pos == 0:
                break

    return signal


def signals_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    long when close > N-bar high with volume confirmation
    short when close < N-bar low with volume confirmation
    """
    lookback  = params.get("lookback_bars", 20)
    vol_z_min = params.get("volume_zscore_min", 1.0)

    roll_high = df["close"].rolling(lookback).max().shift(1)
    roll_low  = df["close"].rolling(lookback).min().shift(1)
    vol_z     = volume_zscore(df["volume"])
    signal    = pd.Series(0, index=df.index)
    pos       = 0

    for i in range(lookback, len(df)):
        c  = df["close"].iloc[i]
        vh = roll_high.iloc[i]
        vl = roll_low.iloc[i]
        vz = vol_z.iloc[i]

        if pd.isna(vh) or pd.isna(vl):
            continue

        if pos == 0:
            if c > vh and vz > vol_z_min:
                pos = 1
            elif c < vl and vz > vol_z_min:
                pos = -1
        elif pos == 1 and c < df["close"].iloc[i - lookback // 2]:
            pos = 0
        elif pos == -1 and c > df["close"].iloc[i - lookback // 2]:
            pos = 0

        signal.iloc[i] = pos

    return signal


def signals_bb_squeeze(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    bollinger band squeeze — trade breakout when bands expand after tight squeeze
    """
    bb_period = params.get("bb_period", 20)
    bb_std    = params.get("bb_std", 2.0)
    kc_mult   = params.get("kc_mult", 1.5)

    lower, mid, upper = bollinger_bands(df["close"], bb_period, bb_std)
    atr_val           = atr(df, bb_period)
    kc_upper          = mid + kc_mult * atr_val
    kc_lower          = mid - kc_mult * atr_val

    # squeeze = BB inside KC
    squeeze  = (upper < kc_upper) & (lower > kc_lower)
    bb_width = upper - lower

    signal = pd.Series(0, index=df.index)
    pos    = 0

    for i in range(bb_period + 1, len(df)):
        in_squeeze_prev = squeeze.iloc[i - 1]
        in_squeeze_now  = squeeze.iloc[i]
        momentum        = df["close"].iloc[i] - df["close"].iloc[i - bb_period // 2]

        # breakout = was in squeeze, now expanding
        if in_squeeze_prev and not in_squeeze_now:
            pos = 1 if momentum > 0 else -1

        # exit when width contracts again
        if pos != 0 and bb_width.iloc[i] < bb_width.iloc[i - 1]:
            pos = 0

        signal.iloc[i] = pos

    return signal


def signals_vwap_slope_break(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    high-frequency intraday strategy: trade in the direction of the VWAP slope
    when it crosses zero. enters when VWAP slope flips positive (long) or
    negative (short); exits when slope flips back. fires often — most days
    produce multiple entries.

    structural mechanism: intraday VWAP is the reference price for institutional
    execution. when VWAP slope changes sign, large orders are being filled
    against the recent direction. trading with the new slope captures the
    continuation while those orders complete.
    """
    slope_lookback = params.get("slope_lookback_bars", 10)
    min_atr_pct    = params.get("min_atr_pct", 0.0005)

    v   = vwap(df)
    a   = atr(df, 14) / df["close"]   # normalized volatility filter
    sl  = v.diff(slope_lookback)
    sl_sign = np.sign(sl)
    sl_prev = sl_sign.shift(1)

    signal = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    for i in range(slope_lookback, len(df)):
        s_now  = sl_sign.iloc[i]
        s_prev = sl_prev.iloc[i]
        atr_ok = bool(a.iloc[i] >= min_atr_pct) if not pd.isna(a.iloc[i]) else False
        if pos == 0:
            if s_now > 0 and s_prev <= 0 and atr_ok:
                pos = 1
            elif s_now < 0 and s_prev >= 0 and atr_ok:
                pos = -1
        elif pos == 1 and s_now < 0:
            pos = 0
        elif pos == -1 and s_now > 0:
            pos = 0
        signal.iloc[i] = pos
    return signal


def signals_bb_band_touch_revert(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    high-frequency mean reversion: fade touches of the upper/lower bollinger
    band. enters short on upper touch, long on lower touch, exits at the mid
    band. fires daily, often multiple times.

    structural mechanism: at 2-sigma band touch on 1m, price has overextended
    relative to its recent distribution. without a fundamental driver, the
    median outcome is reversion toward the rolling mean.
    """
    period = params.get("bb_period", 20)
    std    = params.get("bb_std", 2.0)
    min_atr_pct = params.get("min_atr_pct", 0.0003)

    lower, mid, upper = bollinger_bands(df["close"], period, std)
    a = atr(df, 14) / df["close"]

    signal = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    for i in range(period, len(df)):
        c   = df["close"].iloc[i]
        lo  = lower.iloc[i]
        hi  = upper.iloc[i]
        mi  = mid.iloc[i]
        atr_ok = bool(a.iloc[i] >= min_atr_pct) if not pd.isna(a.iloc[i]) else False
        if pd.isna(lo) or pd.isna(hi) or pd.isna(mi):
            continue
        if pos == 0 and atr_ok:
            if c <= lo:
                pos = 1
            elif c >= hi:
                pos = -1
        elif pos == 1 and c >= mi:
            pos = 0
        elif pos == -1 and c <= mi:
            pos = 0
        signal.iloc[i] = pos
    return signal


def signals_bb_band_touch_revert_v2(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    v2 of bb_band_touch_revert with RSI confluence and stricter entry:
      - require RSI <= rsi_oversold (default 30) AND close <= lower band for long
      - require RSI >= rsi_overbought (default 70) AND close >= upper band for short
      - require close *outside* the band by min_breach_pct (not just touch)

    v1 had 49% win rate but Sharpe -11.68 — losses were larger than wins because
    band-touches alone catch too many noise dips that continue. requiring RSI
    confirmation filters to genuinely overextended moves where mean-reversion is
    more likely; the stricter entry trades volume for hit-rate quality.
    """
    period          = params.get("bb_period", 20)
    std             = params.get("bb_std", 2.0)
    rsi_period      = params.get("rsi_period", 14)
    rsi_overbought  = params.get("rsi_overbought", 70)
    rsi_oversold    = params.get("rsi_oversold", 30)
    min_breach_pct  = params.get("min_breach_pct", 0.0005)
    min_atr_pct     = params.get("min_atr_pct", 0.0003)

    lower, mid, upper = bollinger_bands(df["close"], period, std)
    a = atr(df, 14) / df["close"]
    r = rsi(df["close"], rsi_period)

    signal = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    for i in range(period, len(df)):
        c   = df["close"].iloc[i]
        lo  = lower.iloc[i]
        hi  = upper.iloc[i]
        mi  = mid.iloc[i]
        ri  = r.iloc[i]
        atr_ok = bool(a.iloc[i] >= min_atr_pct) if not pd.isna(a.iloc[i]) else False
        if pd.isna(lo) or pd.isna(hi) or pd.isna(mi) or pd.isna(ri):
            continue
        if pos == 0 and atr_ok:
            below_band = c <= lo * (1.0 - min_breach_pct)
            above_band = c >= hi * (1.0 + min_breach_pct)
            if below_band and ri <= rsi_oversold:
                pos = 1
            elif above_band and ri >= rsi_overbought:
                pos = -1
        elif pos == 1 and c >= mi:
            pos = 0
        elif pos == -1 and c <= mi:
            pos = 0
        signal.iloc[i] = pos
    return signal


def signals_half_hour_continuation(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    intraday periodicity: trade the sign of each 30-min bucket's prior-N-day
    average return. e.g. if 10:00-10:30 has averaged +5 bps over the past 40
    days, take a long during 10:00-10:30 today.

    structural mechanism: Heston, Korajczyk & Sadka (JF 2010) — institutional
    order-splitting on fixed schedules (TWAP slicers, fund-flow rebalances,
    options-hedging cadences) repeats at the same intraday clock-time, producing
    return persistence in each half-hour bucket up to ~40 trading days out.

    single-ticker time-series variant of the paper's cross-sectional design.
    """
    lookback_days  = params.get("lookback_days", 40)
    bucket_min     = params.get("bucket_size_min", 30)
    threshold_bps  = params.get("threshold_bps", 3)
    threshold      = threshold_bps / 10000.0

    # build 30-min bars from the 1-min source
    bars = df.resample(f"{bucket_min}min", label="left", closed="left").agg(
        open=("open", "first"),
        close=("close", "last"),
    ).dropna()
    if bars.empty:
        return pd.Series(0, index=df.index, dtype=int)

    # keep only buckets that begin during regular trading hours
    et_idx = bars.index.tz_convert("America/New_York")
    in_session = (
        (et_idx.time >= pd.Timestamp("2000-01-01 09:30").time()) &
        (et_idx.time <  pd.Timestamp("2000-01-01 16:00").time())
    )
    bars = bars[in_session]
    if bars.empty:
        return pd.Series(0, index=df.index, dtype=int)

    bars["ret"]    = bars["close"] / bars["open"] - 1.0
    et_idx         = bars.index.tz_convert("America/New_York")
    bars["bucket"] = (et_idx.hour * 60 + et_idx.minute).astype(int)

    # per-bucket trailing avg of the prior N occurrences (one occurrence = one day)
    bars["avg_ret"] = (
        bars.groupby("bucket")["ret"]
            .transform(lambda x: x.shift(1).rolling(lookback_days, min_periods=max(10, lookback_days // 4)).mean())
    )

    sig_30m = pd.Series(0, index=bars.index, dtype=int)
    sig_30m[bars["avg_ret"] >  threshold] =  1
    sig_30m[bars["avg_ret"] < -threshold] = -1

    # project the 30-min signal onto every 1-min bar via forward-fill
    sig_1m = sig_30m.reindex(df.index, method="ffill").fillna(0).astype(int)

    # force flat at the last 1-min bar of each trading day so we never carry
    # overnight (the resample step doesn't reset on session breaks)
    et_idx_1m = df.index.tz_convert("America/New_York")
    date_1m   = pd.Series(et_idx_1m.normalize(), index=df.index)
    is_eod    = date_1m != date_1m.shift(-1)
    sig_1m[is_eod] = 0
    return sig_1m


def signals_noise_area_breakout(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    intraday noise area breakout — Zarattini, Aziz & Barbon (2024).
    SSRN 4824172. claimed SPY Sharpe ~1.33.

    mechanism:
      - each trading day, the "noise area" is the range we expect the
        underlying to move purely from microstructure noise.
      - measured as a multiple sigma_mult of the average absolute move-from-
        open across the prior `lookback_days` sessions.
      - when intraday close pierces the upper (lower) bound, take it as a
        sign of NON-noise directional flow and go long (short).
      - exit on VWAP cross against position OR end of session.

    paper's structural justification: institutional order flow that exceeds
    the typical noise envelope reflects new information. exiting at VWAP
    cross protects against early-day false breaks that retrace.
    """
    sigma_mult       = params.get("sigma_mult", 1.0)
    lookback_days    = params.get("lookback_days", 14)
    one_entry_per_day = bool(params.get("one_entry_per_day", True))

    # 1. daily open + average absolute move-from-open over prior N sessions.
    # use the calendar-date STRING as the bucket key — avoids tz-aware/naive
    # reindex bugs where DatetimeIndex with ET tz fails to match values from
    # date_series.values (which strips the tz).
    et_idx       = df.index.tz_convert("America/New_York")
    date_keys    = pd.Series(et_idx.strftime("%Y-%m-%d"), index=df.index)

    daily_open    = df["open"].groupby(date_keys).first()
    daily_high    = df["high"].groupby(date_keys).max()
    daily_low     = df["low"].groupby(date_keys).min()
    daily_range   = (daily_high - daily_low) / daily_open
    avg_move      = daily_range.rolling(lookback_days).mean().shift(1)

    upper_per_day = (daily_open * (1 + sigma_mult * avg_move)).to_dict()
    lower_per_day = (daily_open * (1 - sigma_mult * avg_move)).to_dict()

    # 2. intraday VWAP (resets each session)
    v = vwap(df)

    # 3. walk bar-by-bar — dict lookup keyed by date string
    signal = pd.Series(0, index=df.index, dtype=int)
    pos    = 0
    traded_long_today  = False
    traded_short_today = False
    prev_date          = None

    dates_arr = date_keys.values
    close_arr = df["close"].values
    vwap_arr  = v.values

    last_idx = len(df) - 1
    for i in range(len(df)):
        d  = dates_arr[i]
        if d != prev_date:
            # session boundary
            traded_long_today  = False
            traded_short_today = False
            prev_date = d
            # close any overnight position (we never carry across days)
            pos = 0

        c   = close_arr[i]
        up  = upper_per_day.get(d)
        lo  = lower_per_day.get(d)
        vw  = vwap_arr[i]
        if up is None or lo is None or pd.isna(up) or pd.isna(lo) or pd.isna(vw):
            signal.iloc[i] = pos
            continue

        # entry
        if pos == 0:
            if c > up and (not traded_long_today or not one_entry_per_day):
                pos = 1
                traded_long_today = True
            elif c < lo and (not traded_short_today or not one_entry_per_day):
                pos = -1
                traded_short_today = True

        # exit on VWAP cross against position
        elif pos == 1 and c < vw:
            pos = 0
        elif pos == -1 and c > vw:
            pos = 0

        # force flat at last bar of the day (next iter's date check would do it
        # too, but this records the flat in the signal for cleanliness)
        next_idx = i + 1
        if next_idx > last_idx or dates_arr[next_idx] != d:
            pos = 0

        signal.iloc[i] = pos
    return signal


def signals_trend_ride(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    custom strategy designed from prior backtest findings:

      - bb_squeeze had episodic positive Sharpe — momentum breakout has SOMETIMES
        edge, often in trending regimes
      - mean-reversion strategies had high WR but bad payoff
      - market regime classifier shows ~37% of days are bull/bear (real trend)

    design: slow, picky trend-follower with wide stops + long max-hold.

      entry  : close > 50-bar EMA AND close > 30-bar high AND vol_z > 1
               (mirror: close < 50-bar EMA AND close < 30-bar low AND vol_z > 1)
      exit   : close crosses back through the 50-bar EMA
      filter : intended to pair with apply_market_regime=True so only fires
               in bull or bear regimes (handled at run_backtest level via
               STRATEGY_MARKET_AFFINITY['trend_ride'] = {'bull','bear'})

    the goal is "hold a lot to make profit": few trades, wide stops, long hold.
    """
    ema_period        = params.get("ema_period", 50)
    breakout_lookback = params.get("breakout_lookback", 30)
    vol_z_min         = params.get("vol_z_min", 1.0)

    e         = ema(df["close"], ema_period)
    roll_high = df["close"].rolling(breakout_lookback).max().shift(1)
    roll_low  = df["close"].rolling(breakout_lookback).min().shift(1)
    vol_z     = volume_zscore(df["volume"])

    signal = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    for i in range(ema_period + 1, len(df)):
        c  = df["close"].iloc[i]
        em = e.iloc[i]
        vh = roll_high.iloc[i]
        vl = roll_low.iloc[i]
        vz = vol_z.iloc[i]

        if pd.isna(em) or pd.isna(vh) or pd.isna(vl) or pd.isna(vz):
            continue

        if pos == 0:
            if c > em and c > vh and vz > vol_z_min:
                pos = 1
            elif c < em and c < vl and vz > vol_z_min:
                pos = -1
        elif pos == 1 and c < em:
            pos = 0
        elif pos == -1 and c > em:
            pos = 0

        signal.iloc[i] = pos
    return signal


def signals_overnight_gap_fade(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    fade large overnight gaps: long after a sharp gap-down open, short after a
    sharp gap-up open. exits at exit_time_et (default 12:00 ET) — the fade is
    a morning-only effect once VWAP funds finish their forced execution.

    structural mechanism: retail panic on overnight news compounds with
    market-on-open fills; the residual pressure exhausts in the first 1-2
    hours and price drifts back toward prior-day close.
    """
    gap_threshold = params.get("gap_threshold_pct", 0.010)
    exit_time_et  = params.get("exit_time_et", "12:00")

    et_idx = df.index.tz_convert("America/New_York")
    dates  = pd.Series(et_idx.normalize(), index=df.index)

    daily_open  = df["open"].groupby(dates).first()
    daily_close = df["close"].groupby(dates).last()
    daily_gap   = (daily_open - daily_close.shift(1)) / daily_close.shift(1)

    direction_per_day = pd.Series(0, index=daily_gap.index, dtype=int)
    direction_per_day[daily_gap >  gap_threshold] = -1   # fade gap-up
    direction_per_day[daily_gap < -gap_threshold] =  1   # fade gap-down

    bar_direction = direction_per_day.reindex(dates).values

    exit_h, exit_m = [int(x) for x in exit_time_et.split(":")]
    after_exit = (et_idx.hour > exit_h) | ((et_idx.hour == exit_h) & (et_idx.minute >= exit_m))

    signal = pd.Series(bar_direction, index=df.index)
    signal[after_exit] = 0
    return signal.astype(int)


def signals_extreme_bar_fade(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    fade single-bar moves whose body exceeds threshold * ATR: long the bar
    after a big down-bar, short the bar after a big up-bar. holds for
    hold_bars then exits.

    structural mechanism: 1-bar spikes are usually liquidity vacuums (one
    aggressive order, no offsetting flow). once the next minute's resting
    orders refresh, price tends to revert at least partially.
    """
    atr_period = params.get("atr_period", 14)
    threshold  = params.get("body_atr_mult", 3.0)
    hold_bars  = params.get("hold_bars", 10)

    body    = (df["close"] - df["open"]).abs()
    atr_v   = atr(df, atr_period)
    extreme = body > (threshold * atr_v)
    bar_dir = np.sign(df["close"] - df["open"])   # +1 up bar, -1 down bar

    signal = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    held = 0
    for i in range(atr_period, len(df)):
        if pos != 0:
            held += 1
            if held >= hold_bars:
                pos = 0
                held = 0
        if pos == 0 and bool(extreme.iloc[i]):
            d = int(-bar_dir.iloc[i])
            if d != 0:
                pos  = d
                held = 0
        signal.iloc[i] = pos
    return signal


def signals_qqq_spy_dispersion_multi(dfs: dict, params: dict) -> pd.Series:
    """multi-asset adapter wrapping signals_qqq_spy_dispersion for the engine."""
    return signals_qqq_spy_dispersion(dfs["QQQ"], dfs["SPY"], params)


def signals_qqq_spy_dispersion(qqq_df: pd.DataFrame, spy_df: pd.DataFrame, params: dict) -> pd.Series:
    """
    cross-asset: short-horizon dislocations between QQQ and SPY beyond their
    typical beta-adjusted co-movement should mean-revert. trades QQQ.

      - rolling beta of QQQ-vs-SPY log returns (beta_lookback_min)
      - residual = qqq_ret - beta * spy_ret
      - z-score of residual (zscore_lookback_min)
      - enter SHORT QQQ when z > entry_z (QQQ outperformed, expect snapback)
      - enter LONG  QQQ when z < -entry_z
      - exit when |z| <= exit_z

    returns signal series aligned to qqq_df.index (single-ticker shape so the
    standard run_backtest can consume it without changes).
    """
    beta_lb = params.get("beta_lookback_min", 60)
    z_lb    = params.get("zscore_lookback_min", 30)
    entry_z = params.get("entry_z", 2.0)
    exit_z  = params.get("exit_z", 0.3)

    common = qqq_df.index.intersection(spy_df.index)
    q = qqq_df["close"].reindex(common)
    s = spy_df["close"].reindex(common)

    q_ret = np.log(q / q.shift(1))
    s_ret = np.log(s / s.shift(1))

    cov  = q_ret.rolling(beta_lb).cov(s_ret)
    var  = s_ret.rolling(beta_lb).var().replace(0, np.nan)
    beta = cov / var
    resid = q_ret - beta * s_ret
    z = (resid - resid.rolling(z_lb).mean()) / resid.rolling(z_lb).std().replace(0, np.nan)

    sig = pd.Series(0, index=common)
    pos = 0
    for i in range(len(common)):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue
        if pos == 0:
            if zi > entry_z:
                pos = -1
            elif zi < -entry_z:
                pos = 1
        elif pos == 1 and zi >= -exit_z:
            pos = 0
        elif pos == -1 and zi <= exit_z:
            pos = 0
        sig.iloc[i] = pos

    return sig.reindex(qqq_df.index, fill_value=0)


def signals_ema_crossover(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    fast EMA crosses above slow EMA = long
    fast EMA crosses below slow EMA = short
    """
    fast = params.get("fast_period", 9)
    slow = params.get("slow_period", 21)

    fast_ema = ema(df["close"], fast)
    slow_ema = ema(df["close"], slow)
    cross    = fast_ema - slow_ema
    signal   = pd.Series(0, index=df.index)

    for i in range(slow, len(df)):
        if cross.iloc[i] > 0 and cross.iloc[i - 1] <= 0:
            signal.iloc[i] = 1
        elif cross.iloc[i] < 0 and cross.iloc[i - 1] >= 0:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


# strategy registry — maps name to signal function + default params.
# stop_atr_mult is read by run_backtest: tight for trend, wide for mean-reversion.
# "active" is consumed by the standalone runner; orchestrator can still dispatch
# an inactive strategy explicitly by name (useful for revisiting after signal work).
#
# the registry is mutable at runtime — code_agent registers freshly-generated
# strategies via register_strategy() so the next backtest can find them.
STRATEGIES = {
    "rsi_reversion":  (signals_rsi_reversion,  {"rsi_period": 14, "oversold": 30, "overbought": 70, "stop_atr_mult": 3.5, "active": False}),
    "vwap_reversion": (signals_vwap_reversion,  {"threshold_pct": 0.003, "stop_atr_mult": 3.5,                         "active": False}),
    "orb":            (signals_orb,             {"orb_minutes": 15, "stop_atr_mult": 1.5,                              "active": True}),
    "momentum":       (signals_momentum,        {"lookback_bars": 20, "volume_zscore_min": 1.0, "stop_atr_mult": 1.5,  "active": True}),
    "bb_squeeze":     (signals_bb_squeeze,      {"bb_period": 20, "bb_std": 2.0, "kc_mult": 1.5, "stop_atr_mult": 2.0, "active": True}),
    "ema_crossover":  (signals_ema_crossover,   {"fast_period": 9, "slow_period": 21, "stop_atr_mult": 1.5,            "active": True}),

    # high-frequency intraday strategies — fire many times per day
    "vwap_slope_break": (
        signals_vwap_slope_break,
        {"slope_lookback_bars": 10, "min_atr_pct": 0.0005,
         "stop_atr_mult": 1.5, "max_hold_bars": 30, "active": True},
    ),
    "bb_band_touch_revert": (
        signals_bb_band_touch_revert,
        {"bb_period": 20, "bb_std": 2.0, "min_atr_pct": 0.0003,
         "stop_atr_mult": 2.5, "max_hold_bars": 30, "active": True},
    ),
    # v2: stricter entry (RSI confluence + breach beyond band) and tighter stop.
    # v1 had 49% WR but Sharpe -11.68 — payoff asymmetry, fixed here by being
    # pickier about entries rather than letting any band touch fire.
    "bb_band_touch_revert_v2": (
        signals_bb_band_touch_revert_v2,
        {"bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
         "rsi_overbought": 70, "rsi_oversold": 30,
         "min_breach_pct": 0.0005, "min_atr_pct": 0.0003,
         "stop_atr_mult": 1.5, "max_hold_bars": 30, "active": True},
    ),
    # intraday periodicity per Heston/Korajczyk/Sadka 2010 — single-ticker
    # time-series variant. half-hour bucket return sign over prior N days.
    "half_hour_continuation": (
        signals_half_hour_continuation,
        {"lookback_days": 40, "bucket_size_min": 30, "threshold_bps": 3,
         "stop_atr_mult": 2.0, "max_hold_bars": 30, "active": True},
    ),

    # Zarattini, Aziz & Barbon 2024 — intraday noise-area breakout.
    # SSRN 4824172. paper claims SPY Sharpe ~1.33 with sigma_mult=1.0.
    "noise_area_breakout": (
        signals_noise_area_breakout,
        {"sigma_mult": 1.0, "lookback_days": 14, "one_entry_per_day": True,
         "stop_atr_mult": 2.0, "max_hold_bars": 390, "active": True},
    ),

    # custom: slow trend-ride. wide stops, long hold, regime-gated.
    "trend_ride": (
        signals_trend_ride,
        {"ema_period": 50, "breakout_lookback": 30, "vol_z_min": 1.0,
         "stop_atr_mult": 2.5, "max_hold_bars": 200, "active": True},
    ),

    # structural-edge strategies — fading liquidity dislocations
    "overnight_gap_fade": (
        signals_overnight_gap_fade,
        {"gap_threshold_pct": 0.010, "exit_time_et": "12:00",
         "stop_atr_mult": 2.5, "max_hold_bars": 180, "active": True},
    ),
    "extreme_bar_fade": (
        signals_extreme_bar_fade,
        {"atr_period": 14, "body_atr_mult": 3.0, "hold_bars": 10,
         "stop_atr_mult": 2.0, "active": True},
    ),

    # cross-asset (3-tuple) — engine dispatches via meta["kind"] == "multi"
    # ATR stop disabled, 20-bar time exit per autonomous_agent's original spec —
    # the residual needs room to mean-revert; ATR stop was killing it inside noise.
    "qqq_spy_dispersion": (
        signals_qqq_spy_dispersion_multi,
        {"beta_lookback_min": 60, "zscore_lookback_min": 30,
         "entry_z": 2.0, "exit_z": 0.3,
         "disable_atr_stop": True, "max_hold_bars": 20, "active": True},
        {"kind": "multi", "data_tickers": ["QQQ", "SPY"], "tradeable_ticker": "QQQ"},
    ),
}


# ---------------------------------------------------------------------------
# runtime registration — code_agent uses these to add generated strategies
# ---------------------------------------------------------------------------

def is_registered(name: str) -> bool:
    """true if the strategy name has a signal function in STRATEGIES."""
    if not name:
        return False
    norm = name.lower().replace(" ", "_")
    return any(key in norm or norm in key for key in STRATEGIES)


def get_strategy_meta(strategy_key: str) -> dict:
    """returns the 3rd-tuple meta dict for a strategy; defaults to single-asset."""
    entry = STRATEGIES.get(strategy_key)
    if not entry:
        return {"kind": "single"}
    if len(entry) >= 3 and isinstance(entry[2], dict):
        meta = dict(entry[2])
        meta.setdefault("kind", "single")
        return meta
    return {"kind": "single"}


def load_generated_strategies(strategies_dir: Path = Path("strategies")) -> int:
    """
    scan strategies/ for modules with a signals() function and register each
    one into STRATEGIES. called on orchestrator startup so generated code
    survives process restarts. returns the count registered.

    file naming convention: <strategy_id>_<strategy_name>.py — name is the
    portion after the first underscore.
    """
    if not strategies_dir.exists():
        return 0

    import importlib.util
    loaded = 0
    for path in sorted(strategies_dir.glob("*.py")):
        if path.name.startswith("_") or path.stem == "__init__":
            continue
        # name = filename minus the strategy_id prefix
        parts = path.stem.split("_", 1)
        strategy_name = parts[1] if len(parts) == 2 else path.stem

        if is_registered(strategy_name):
            continue  # already registered (either built-in or previously loaded)

        try:
            spec = importlib.util.spec_from_file_location(f"gen_{path.stem}", path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            log.warning(f"failed to import {path.name}: {e}")
            continue

        if not hasattr(mod, "signals"):
            log.warning(f"{path.name}: no signals() function — skipping")
            continue

        try:
            register_strategy(strategy_name, mod.signals, default_params={}, overwrite=False)
            loaded += 1
        except Exception as e:
            log.warning(f"register_strategy failed for {strategy_name}: {e}")

    if loaded:
        log.info(f"loaded {loaded} generated strategies from {strategies_dir}/")
    return loaded


def register_strategy(name: str, signal_fn, default_params: dict,
                      regime_affinity: Optional[set] = None,
                      stop_atr_mult: float = 2.0,
                      overwrite: bool = False,
                      meta: Optional[dict] = None) -> str:
    """
    add a strategy to the runtime registry.

    single-asset signal: signal_fn(df, params) -> Series of {-1,0,1} on df.index
                         meta=None (default).
    cross-asset signal:  signal_fn(dfs_dict, params) -> Series on
                         dfs_dict[meta["tradeable_ticker"]].index
                         meta={"kind": "multi", "data_tickers": [...],
                               "tradeable_ticker": "..."}.

    returns the normalized key used in STRATEGIES.
    """
    if signal_fn is None:
        raise ValueError("signal_fn required")

    key    = name.lower().replace(" ", "_")
    params = {**(default_params or {})}
    params.setdefault("stop_atr_mult", stop_atr_mult)
    params.setdefault("active", True)

    if key in STRATEGIES and not overwrite:
        raise ValueError(f"strategy '{key}' already registered (pass overwrite=True to replace)")

    if meta and meta.get("kind") == "multi":
        if not meta.get("data_tickers"):
            raise ValueError("multi-asset strategy requires meta['data_tickers']")
        if not meta.get("tradeable_ticker"):
            raise ValueError("multi-asset strategy requires meta['tradeable_ticker']")
        if meta["tradeable_ticker"] not in meta["data_tickers"]:
            raise ValueError("tradeable_ticker must be in data_tickers")
        STRATEGIES[key] = (signal_fn, params, meta)
    else:
        STRATEGIES[key] = (signal_fn, params)

    if regime_affinity is not None:
        STRATEGY_REGIME_AFFINITY[key] = set(regime_affinity)
    log.info(f"registered strategy '{key}' | meta={meta or 'single-asset'} | affinity={regime_affinity}")
    return key


# ---------------------------------------------------------------------------
# backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    position_size_pct: float = 0.10,   # 10% of capital per trade
    stop_atr_mult: float = ATR_STOP_MULT,
    regime_series: Optional[pd.Series] = None,
    allowed_regimes: Optional[set] = None,
    quality_series: Optional[pd.DataFrame] = None,
    quality_min_pct_pos: float = 0.55,
    disable_atr_stop: bool = False,    # opt-out for slow mean-reversion
    max_hold_bars: Optional[int] = None,  # force exit after N bars (None = no cap)
    market_regime_series: Optional[pd.Series] = None,    # bar-aligned bull/bear/high_vol/neutral
    allowed_market_regimes: Optional[set] = None,        # filter entries by market regime
) -> dict:
    """
    simulates trades from a signal series on OHLCV data.
    returns full performance metrics dict.

    signal: series of {1=long, -1=short, 0=flat} aligned to df.index

    execution model:
      - signal generated from close[i-1] is acted on at close[i] (1-bar shift)
      - equity is marked-to-market each bar (capital + unrealized PnL)
      - Sharpe is computed on daily-resampled returns and annualized by √252
    """
    # shift signal by one bar to remove same-bar lookahead: a signal generated
    # from bar i-1's close is filled at bar i's close.
    signal = signal.shift(1).fillna(0).astype(int)

    # ATR series for the universal stop loss
    atr_series = atr(df, period=ATR_PERIOD)

    capital  = INITIAL_CAP
    equity   = [capital]
    trades   = []
    pos      = 0
    entry_px = 0.0
    entry_ts = None
    entry_bar = -1
    shares   = 0
    last_exit_bar = -REENTRY_COOLDOWN_BARS   # allow first entry immediately

    close = df["close"]

    for i in range(1, len(df)):
        ts  = df.index[i]
        c   = close.iloc[i]
        sig = int(signal.iloc[i])
        prev_sig = int(signal.iloc[i - 1])

        # ATR stop (skippable per strategy): override sig to flat if the open
        # position has moved adversely by >= stop_atr_mult * ATR.
        if pos != 0 and not disable_atr_stop:
            atr_val = atr_series.iloc[i]
            if not pd.isna(atr_val) and atr_val > 0:
                adverse = (entry_px - c) * pos   # > 0 when losing
                if adverse >= stop_atr_mult * atr_val:
                    sig = 0

        # time-based stop: force flat after max_hold_bars regardless of signal
        if pos != 0 and max_hold_bars is not None and (i - entry_bar) >= max_hold_bars:
            sig = 0

        # entry — only if cooldown elapsed AND regime is compatible AND the
        # embedding-based quality gate (if provided) confirms the direction.
        # open positions are not force-exited on regime/quality change.
        regime_ok = (
            allowed_regimes is None
            or regime_series is None
            or regime_series.iloc[i] in allowed_regimes
        )
        market_ok = (
            allowed_market_regimes is None
            or market_regime_series is None
            or market_regime_series.iloc[i] in allowed_market_regimes
        )
        quality_ok = True
        if quality_series is not None and sig != 0 and i < len(quality_series):
            pct_pos = quality_series["fwd_pct_positive"].iloc[i]
            if not pd.isna(pct_pos):
                if sig == 1:
                    quality_ok = pct_pos >= quality_min_pct_pos
                else:  # sig == -1
                    quality_ok = pct_pos <= (1.0 - quality_min_pct_pos)
        if pos == 0 and sig != 0 and regime_ok and market_ok and quality_ok and (i - last_exit_bar) >= REENTRY_COOLDOWN_BARS:
            pos      = sig
            entry_px = c * (1 + SLIPPAGE * sig)   # slip in direction of trade
            shares   = int((capital * position_size_pct) / entry_px)
            entry_ts = ts
            entry_bar = i
            commission_cost = shares * entry_px * COMMISSION
            capital -= commission_cost

        # exit
        elif pos != 0 and (sig == 0 or sig != prev_sig):
            exit_px  = c * (1 - SLIPPAGE * pos)
            pnl      = shares * (exit_px - entry_px) * pos
            commission_cost = shares * exit_px * COMMISSION
            capital += pnl - commission_cost

            trades.append({
                "entry_ts":  entry_ts,
                "exit_ts":   ts,
                "side":      "long" if pos == 1 else "short",
                "entry_px":  round(entry_px, 4),
                "exit_px":   round(exit_px, 4),
                "shares":    shares,
                "pnl":       round(pnl, 2),
                "pct_return": round(pnl / (shares * entry_px), 6) if shares > 0 else 0,
                "bars_held": i - df.index.get_loc(entry_ts),
            })

            pos           = 0
            entry_px      = 0.0
            shares        = 0
            last_exit_bar = i
            # no same-bar flip — cooldown will release the next entry naturally

        # mark-to-market equity: cash + unrealized PnL on any open position
        mtm = capital + (shares * (c - entry_px) * pos if pos != 0 else 0.0)
        equity.append(mtm)

    # compute metrics
    equity_s    = pd.Series(equity, index=df.index[:len(equity)])

    total_return = (equity_s.iloc[-1] - INITIAL_CAP) / INITIAL_CAP

    # sharpe on daily-resampled equity (proper annualization by √252)
    daily_equity  = equity_s.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # max drawdown on mark-to-market equity
    roll_max    = equity_s.cummax()
    drawdown    = (equity_s - roll_max) / roll_max
    max_dd      = float(drawdown.min())

    # calmar
    calmar = float(total_return / abs(max_dd)) if max_dd < 0 else 0.0

    # trade stats
    if trades:
        trade_df   = pd.DataFrame(trades)
        win_rate   = float((trade_df["pnl"] > 0).mean())
        avg_win    = float(trade_df[trade_df["pnl"] > 0]["pnl"].mean()) if (trade_df["pnl"] > 0).any() else 0.0
        avg_loss   = float(trade_df[trade_df["pnl"] < 0]["pnl"].mean()) if (trade_df["pnl"] < 0).any() else 0.0
        avg_bars   = float(trade_df["bars_held"].mean())
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    else:
        win_rate = avg_win = avg_loss = avg_bars = profit_factor = 0.0
        trade_df = pd.DataFrame()

    return {
        "total_return":   round(total_return, 6),
        "final_capital":  round(capital, 2),
        "sharpe":         round(sharpe, 4),
        "max_drawdown":   round(max_dd, 6),
        "calmar":         round(calmar, 4),
        "win_rate":       round(win_rate, 4),
        "total_trades":   len(trades),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "avg_bars_held":  round(avg_bars, 2),
        "profit_factor":  round(profit_factor, 4),
        "trades":         trades,
        "equity_curve":   equity_s.resample("1h").last().tolist(),   # hourly for storage
    }


# ---------------------------------------------------------------------------
# regime-segmented backtest
# ---------------------------------------------------------------------------

def backtest_by_regime(results: dict, df: pd.DataFrame, signal: pd.Series,
                       stop_atr_mult: float = ATR_STOP_MULT,
                       regime_series: Optional[pd.Series] = None,
                       allowed_regimes: Optional[set] = None) -> dict:
    """
    segments backtest results by time-of-day as a proxy for regime
    until regime store is fully wired in.
    open (9:30-11:00), midday (11:00-14:00), close (14:00-16:00)
    """
    sessions = {
        "open":   ("09:30", "11:00"),
        "midday": ("11:00", "14:00"),
        "close":  ("14:00", "16:00"),
    }

    regime_results = {}
    idx_et = df.index.tz_convert("America/New_York")

    for session, (start, end) in sessions.items():
        mask       = (idx_et.time >= pd.Timestamp(f"2000-01-01 {start}").time()) & \
                     (idx_et.time <= pd.Timestamp(f"2000-01-01 {end}").time())
        session_df = df[mask]
        session_sig = signal[mask]

        if len(session_df) < 10:
            continue

        try:
            session_regime = regime_series[mask] if regime_series is not None else None
            r = run_backtest(
                session_df,
                session_sig,
                stop_atr_mult=stop_atr_mult,
                regime_series=session_regime,
                allowed_regimes=allowed_regimes,
            )
            regime_results[session] = {
                "sharpe":       r["sharpe"],
                "win_rate":     r["win_rate"],
                "total_trades": r["total_trades"],
                "max_drawdown": r["max_drawdown"],
            }
        except Exception:
            pass

    return regime_results


# ---------------------------------------------------------------------------
# walk-forward parameter optimization
# ---------------------------------------------------------------------------

def walk_forward_optimize(
    strategy_name: str,
    param_grid: dict,
    tickers: Optional[list] = None,
    start: str = "2022-01-01",
    end: str = "2025-01-01",
    train_pct: float = 0.7,
    data_dir: Path = DATA_DIR,
) -> dict:
    """
    grid-search params on the first train_pct of each ticker's data, then
    evaluate the best params out-of-sample on the remaining test split.
    a positive train_sharpe with a similar test_sharpe is what we want — a
    big gap means the grid overfit the train period.
    """
    import itertools

    if strategy_name not in STRATEGIES:
        return {"success": False, "reason": f"unknown strategy {strategy_name}"}

    tickers = tickers or ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"]
    fn, default_params = STRATEGIES[strategy_name][0], STRATEGIES[strategy_name][1]
    allowed   = STRATEGY_REGIME_AFFINITY.get(strategy_name)

    # preload + split each ticker
    splits = {}
    for t in tickers:
        try:
            df = load_ticker(t, data_dir=data_dir, start=start, end=end, session="regular")
            cut = int(len(df) * train_pct)
            splits[t] = (df.iloc[:cut], df.iloc[cut:])
        except FileNotFoundError:
            continue
    if not splits:
        return {"success": False, "reason": "no ticker data"}

    keys   = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[param_grid[k] for k in keys])]

    def _avg_sharpe(combo: dict, split_idx: int) -> float:
        merged    = {**default_params, **combo}
        stop_mult = merged.get("stop_atr_mult", ATR_STOP_MULT)
        sharpes   = []
        for t, (train_df, test_df) in splits.items():
            piece    = train_df if split_idx == 0 else test_df
            if len(piece) < 100:
                continue
            signal   = fn(piece, merged)
            regime_s = regime_label_series(piece)
            r = run_backtest(piece, signal, stop_atr_mult=stop_mult,
                             regime_series=regime_s, allowed_regimes=allowed)
            sharpes.append(r["sharpe"])
        return float(np.mean(sharpes)) if sharpes else float("-inf")

    train_results = []
    for combo in combos:
        s = _avg_sharpe(combo, split_idx=0)
        train_results.append({"params": combo, "train_sharpe": round(s, 4)})
        log.info(f"  WF {strategy_name} {combo} train_sharpe={s:.3f}")

    train_results.sort(key=lambda r: r["train_sharpe"], reverse=True)
    best = train_results[0]
    test_sharpe = _avg_sharpe(best["params"], split_idx=1)

    return {
        "success":       True,
        "strategy":      strategy_name,
        "best_params":   best["params"],
        "train_sharpe":  best["train_sharpe"],
        "test_sharpe":   round(test_sharpe, 4),
        "overfit_gap":   round(best["train_sharpe"] - test_sharpe, 4),
        "all_results":   train_results,
    }


def walk_forward_with_gate(
    strategy_name: str,
    ticker: str,
    quality_min_pct_pos_grid: Optional[list] = None,
    start: str = "2022-01-01",
    end: str = "2025-01-01",
    train_pct: float = 0.7,
    data_dir: Path = DATA_DIR,
    params_override: Optional[dict] = None,
) -> dict:
    """
    walk-forward a single (strategy, ticker) pair with the embedding gate.
    sweeps quality thresholds on the train split, picks the best, then
    measures it on the test split. catches threshold overfit explicitly.

    requires a cached quality parquet for the ticker. raises if missing.
    """
    if strategy_name not in STRATEGIES:
        return {"success": False, "reason": f"unknown strategy {strategy_name}"}

    quality_min_pct_pos_grid = quality_min_pct_pos_grid or [0.50, 0.52, 0.55, 0.58]

    fn, default_params = STRATEGIES[strategy_name][0], STRATEGIES[strategy_name][1]
    merged_params = {**default_params, **(params_override or {})}
    stop_mult     = merged_params.get("stop_atr_mult", ATR_STOP_MULT)
    allowed       = STRATEGY_REGIME_AFFINITY.get(strategy_name)

    df = load_ticker(ticker, data_dir=data_dir, start=start, end=end, session="regular")
    cut       = int(len(df) * train_pct)
    train_df  = df.iloc[:cut]
    test_df   = df.iloc[cut:]

    quality_df  = precompute_regime_quality(ticker, df, step=60)
    quality_all = quality_df.reindex(df.index, method="ffill")
    quality_tr  = quality_all.iloc[:cut]
    quality_te  = quality_all.iloc[cut:]

    train_signal = fn(train_df, merged_params)
    test_signal  = fn(test_df,  merged_params)
    train_regime = regime_label_series(train_df)
    test_regime  = regime_label_series(test_df)

    def _bt(piece_df, piece_sig, piece_regime, piece_quality, threshold):
        return run_backtest(
            piece_df, piece_sig,
            stop_atr_mult=stop_mult,
            regime_series=piece_regime,
            allowed_regimes=allowed,
            quality_series=piece_quality,
            quality_min_pct_pos=threshold,
        )

    train_results = []
    for thr in quality_min_pct_pos_grid:
        r = _bt(train_df, train_signal, train_regime, quality_tr, thr)
        train_results.append({
            "threshold":    thr,
            "train_sharpe": round(r["sharpe"], 4),
            "train_trades": r["total_trades"],
        })
        log.info(f"  WF+gate {strategy_name}/{ticker} thr={thr} sharpe={r['sharpe']:.3f} trades={r['total_trades']}")

    train_results.sort(key=lambda x: x["train_sharpe"], reverse=True)
    best = train_results[0]
    test = _bt(test_df, test_signal, test_regime, quality_te, best["threshold"])

    return {
        "success":         True,
        "strategy":        strategy_name,
        "ticker":          ticker,
        "params_used":     merged_params,
        "best_threshold":  best["threshold"],
        "train_sharpe":    best["train_sharpe"],
        "train_trades":    best["train_trades"],
        "test_sharpe":     round(test["sharpe"], 4),
        "test_trades":     test["total_trades"],
        "test_wr":         test["win_rate"],
        "test_dd":         test["max_drawdown"],
        "overfit_gap":     round(best["train_sharpe"] - test["sharpe"], 4),
        "all_train_results": train_results,
    }


# ---------------------------------------------------------------------------
# embedding-based regime quality gate (proof-of-concept infrastructure)
# ---------------------------------------------------------------------------

def precompute_regime_quality(
    ticker: str,
    df: pd.DataFrame,
    step: int = 60,
    k: int = 20,
    cache_dir: Path = Path("vector_stores/.cache"),
    force: bool = False,
) -> pd.DataFrame:
    """
    for every `step` bars, query regime_store.find_similar() and persist:
        timestamp, regime, confidence, fwd_pct_positive, fwd_mean
    results are cached to parquet — subsequent calls are free.

    cost: 1 OpenAI embedding call + 1 chroma query per row. for SPY with
    step=60 over 3y that's ~4800 rows ≈ 15-25 minutes of latency one time.

    usage:
        df = load_ticker("SPY", ...)
        quality = precompute_regime_quality("SPY", df)
        # later:
        run_backtest(df, signal, quality_series=quality.reindex(df.index, method="ffill"))
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_regime_quality_step{step}.parquet"

    if cache_path.exists() and not force:
        log.info(f"loading cached regime quality for {ticker} from {cache_path}")
        return pd.read_parquet(cache_path)

    from vector_stores.regime_store import RegimeStore, WINDOW_BARS
    store = RegimeStore()

    rows = []
    for i in range(WINDOW_BARS, len(df), step):
        window = df.iloc[i - WINDOW_BARS : i]
        try:
            r     = store.find_similar(ticker, window, k=k)
            stats = r.get("forward_return_stats", {})
            rows.append({
                "timestamp":        df.index[i],
                "regime":           r.get("regime", "unknown"),
                "confidence":       r.get("confidence", 0.0),
                "fwd_pct_positive": stats.get("pct_positive", 0.5),
                "fwd_mean":         stats.get("mean", 0.0),
            })
        except Exception as e:
            log.warning(f"  quality query failed at {df.index[i]}: {e}")
        if i % (step * 100) == 0:
            log.info(f"  {ticker} regime quality: {i}/{len(df)} bars")

    if not rows:
        raise RuntimeError(
            f"no regime-quality rows produced for {ticker} — every find_similar() "
            f"call failed. check OPENAI_API_KEY and the regime store's contents."
        )

    out = pd.DataFrame(rows).set_index("timestamp")
    out.to_parquet(cache_path)
    log.info(f"saved {len(out)} regime quality rows to {cache_path}")
    return out


# ---------------------------------------------------------------------------
# trade-log export — write every fill out to a CSV the user can audit
# ---------------------------------------------------------------------------

def dump_trades_to_csv(
    trades: list,
    strategy_key: str,
    ticker: str,
    regime_series: Optional[pd.Series] = None,
    out_dir: Path = Path("results/trades"),
) -> Optional[Path]:
    """
    write the trades list returned by run_backtest to a CSV file. each row is
    a single round-trip (entry+exit) so the user can confirm the simulator's
    PnL by hand.

    columns include:
      entry_date / entry_time / exit_date / exit_time — easy human read
      side, shares, entry_px, exit_px
      gross_cost     — shares * entry_px (what you committed)
      gross_proceeds — shares * exit_px  (what you received at exit)
      pnl, return_pct, hold_minutes, hold_bars
      regime_at_entry — if regime_series provided

    these are derived columns: they let you spot anomalies (e.g. negative
    "gross_proceeds" or zero shares) without joining back to the original
    trades list.
    """
    if not trades:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{strategy_key}_{ticker}_trades.csv"

    df = pd.DataFrame(trades).copy()

    # ensure timestamps are pandas Timestamps
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True)

    # ET-localized date/time columns — the format a human auditor reads
    et = df["entry_ts"].dt.tz_convert("America/New_York")
    xt = df["exit_ts" ].dt.tz_convert("America/New_York")
    df["entry_date"] = et.dt.strftime("%Y-%m-%d")
    df["entry_time"] = et.dt.strftime("%H:%M:%S")
    df["exit_date"]  = xt.dt.strftime("%Y-%m-%d")
    df["exit_time"]  = xt.dt.strftime("%H:%M:%S")

    # dollar columns
    df["gross_cost"]     = (df["shares"] * df["entry_px"]).round(2)
    df["gross_proceeds"] = (df["shares"] * df["exit_px"]).round(2)

    # hold duration in minutes — easier to scan than bars_held
    df["hold_minutes"]   = ((df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 60).round(1)
    df = df.rename(columns={"bars_held": "hold_bars", "pct_return": "return_pct"})

    if regime_series is not None:
        def _regime_for(ts):
            try:
                return regime_series.loc[ts]
            except Exception:
                return "unknown"
        df["regime_at_entry"] = df["entry_ts"].apply(_regime_for)

    # write columns in a human-friendly order; drop the raw timestamps since
    # entry_date/time + exit_date/time cover the same info more readably
    ordered = [
        "entry_date", "entry_time", "exit_date", "exit_time",
        "side", "shares", "entry_px", "exit_px",
        "gross_cost", "gross_proceeds", "pnl", "return_pct",
        "hold_bars", "hold_minutes",
    ]
    if "regime_at_entry" in df.columns:
        ordered.append("regime_at_entry")
    # tolerate older trade dicts that may lack some columns
    ordered = [c for c in ordered if c in df.columns]
    df[ordered].to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# main backtesting agent
# ---------------------------------------------------------------------------

class BacktestingAgent:
    """
    runs backtests on 1m parquet data for any strategy in the registry.
    called by the orchestrator when a new strategy needs validation.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.log      = logging.getLogger("backtesting_agent")

    def run(self, task: dict) -> dict:
        strategy    = task.get("payload", {})
        name        = strategy.get("name", "rsi_reversion").lower().replace(" ", "_")
        params      = strategy.get("params", {})
        tickers     = strategy.get("tickers", ["SPY", "QQQ"])
        start       = strategy.get("start", "2022-01-01")
        end         = strategy.get("end",   "2025-01-01")
        resample    = strategy.get("resample_to", None)
        dump_trades = bool(strategy.get("dump_trades", False))   # write CSV per ticker
        apply_mkt   = bool(strategy.get("apply_market_regime", False))   # gate by SPY bull/bear/vol

        # find matching strategy signal function. prefer exact match, then
        # substring match — otherwise "foo" matches "foo_v2" before "foo_v2"
        # gets its own chance (dict iteration order = insertion order).
        strategy_key = None
        if name in STRATEGIES:
            strategy_key = name
        else:
            for key in STRATEGIES:
                if key in name or name in key:
                    strategy_key = key
                    break

        if not strategy_key:
            # fail explicitly — silently aliasing to rsi_reversion produced
            # meaningless results for novel strategies from autonomous_agent.
            # downstream agents (risk, code) should treat this as "needs a
            # signal function implementation before it can be validated".
            self.log.warning(f"unknown strategy '{name}' — no signal function registered")
            return {
                "success":         False,
                "reason":          f"no signal function registered for strategy '{name}'",
                "needs_code_agent": True,
                "requested_name":  name,
                "requested_params": params,
            }

        entry          = STRATEGIES[strategy_key]
        fn             = entry[0]
        default_params = entry[1]
        strategy_meta  = get_strategy_meta(strategy_key)
        merged_params  = {**default_params, **params}
        stop_mult      = merged_params.get("stop_atr_mult", ATR_STOP_MULT)
        disable_stop   = bool(merged_params.get("disable_atr_stop", False))
        max_hold       = merged_params.get("max_hold_bars")
        allowed        = STRATEGY_REGIME_AFFINITY.get(strategy_key)
        allowed_market = STRATEGY_MARKET_AFFINITY.get(strategy_key) if apply_mkt else None

        # SPY-derived bull/bear/high_vol regime — computed once, reindexed
        # per-ticker. only loaded when apply_market_regime is set.
        spy_for_regime = None
        if apply_mkt and allowed_market is not None:
            try:
                spy_for_regime = load_ticker(
                    "SPY", data_dir=self.data_dir, start=start, end=end, session="regular",
                )
                self.log.info(f"  market regime gate ENABLED for {strategy_key}: allowed={allowed_market}")
            except FileNotFoundError:
                self.log.warning("SPY parquet missing — market regime gating disabled")
                allowed_market = None

        all_results = {}

        if strategy_meta.get("kind") == "multi":
            # cross-asset: load all data_tickers, compute signal once, backtest
            # only the tradeable ticker. caller's `tickers` field is ignored —
            # the strategy itself dictates which tickers it needs.
            data_tickers  = strategy_meta["data_tickers"]
            tradeable     = strategy_meta["tradeable_ticker"]
            self.log.info(f"backtesting {strategy_key} (multi-asset) | "
                          f"data={data_tickers} tradeable={tradeable} | {start} to {end}")
            try:
                dfs = {t: load_ticker(t, data_dir=self.data_dir, start=start, end=end,
                                      session="regular", resample_to=resample)
                       for t in data_tickers}
                signal   = fn(dfs, merged_params)
                df       = dfs[tradeable]
                regime_s = regime_label_series(df)
                results  = run_backtest(df, signal, stop_atr_mult=stop_mult,
                                        regime_series=regime_s, allowed_regimes=allowed,
                                        disable_atr_stop=disable_stop, max_hold_bars=max_hold)
                regimes  = backtest_by_regime(results, df, signal, stop_atr_mult=stop_mult,
                                              regime_series=regime_s, allowed_regimes=allowed)
                results["regime_breakdown"] = regimes
                if dump_trades and results.get("trades"):
                    csv_path = dump_trades_to_csv(
                        results["trades"], strategy_key, tradeable, regime_s,
                    )
                    if csv_path:
                        self.log.info(f"  wrote {len(results['trades'])} trades to {csv_path}")
                        results["trades_csv"] = str(csv_path)
                results.pop("trades", None)
                results.pop("equity_curve", None)
                all_results[tradeable] = results
                self.log.info(
                    f"{tradeable} (multi) | sharpe={results['sharpe']:.2f} | "
                    f"dd={results['max_drawdown']:.2%} | wr={results['win_rate']:.2%} | "
                    f"trades={results['total_trades']}"
                )
            except FileNotFoundError as e:
                self.log.warning(f"multi-asset: parquet missing — {e}")
            except Exception as e:
                self.log.exception(f"multi-asset backtest failed — {e}")

        else:
            self.log.info(f"backtesting {strategy_key} | tickers={tickers} | {start} to {end}")
            for ticker in tickers:
                try:
                    df = load_ticker(
                        ticker,
                        data_dir  = self.data_dir,
                        start     = start,
                        end       = end,
                        session   = "regular",
                        resample_to = resample,
                    )

                    if len(df) < 100:
                        self.log.warning(f"{ticker}: not enough data ({len(df)} bars)")
                        continue

                    signal   = fn(df, merged_params)
                    regime_s = regime_label_series(df)
                    mkt_s    = (market_regime_for_df(spy_for_regime, df)
                                if spy_for_regime is not None else None)

                    results = run_backtest(
                        df, signal,
                        stop_atr_mult=stop_mult,
                        regime_series=regime_s,
                        allowed_regimes=allowed,
                        disable_atr_stop=disable_stop,
                        max_hold_bars=max_hold,
                        market_regime_series=mkt_s,
                        allowed_market_regimes=allowed_market,
                    )
                    regimes = backtest_by_regime(
                        results, df, signal,
                        stop_atr_mult=stop_mult,
                        regime_series=regime_s,
                        allowed_regimes=allowed,
                    )

                    results["regime_breakdown"] = regimes
                    if dump_trades and results.get("trades"):
                        csv_path = dump_trades_to_csv(
                            results["trades"], strategy_key, ticker, regime_s,
                        )
                        if csv_path:
                            self.log.info(f"  wrote {len(results['trades'])} trades to {csv_path}")
                            results["trades_csv"] = str(csv_path)
                    results.pop("trades", None)        # don't store full trade list in summary
                    results.pop("equity_curve", None)

                    all_results[ticker] = results

                    self.log.info(
                        f"{ticker} | sharpe={results['sharpe']:.2f} | "
                        f"dd={results['max_drawdown']:.2%} | "
                        f"wr={results['win_rate']:.2%} | "
                        f"trades={results['total_trades']}"
                    )

                except FileNotFoundError:
                    self.log.warning(f"{ticker}: parquet file not found")
                except Exception as e:
                    self.log.exception(f"{ticker}: backtest failed — {e}")

        if not all_results:
            return {"success": False, "reason": "no results produced"}

        # aggregate across tickers
        sharpes   = [r["sharpe"]       for r in all_results.values()]
        drawdowns = [r["max_drawdown"] for r in all_results.values()]
        winrates  = [r["win_rate"]     for r in all_results.values()]
        trades    = [r["total_trades"] for r in all_results.values()]

        aggregate = {
            "sharpe":       round(float(np.mean(sharpes)), 4),
            "max_drawdown": round(float(np.mean(drawdowns)), 6),
            "win_rate":     round(float(np.mean(winrates)), 4),
            "total_trades": int(np.sum(trades)),
            "calmar":       round(float(np.mean([r["calmar"] for r in all_results.values()])), 4),
        }

        return {
            "success":          True,
            "strategy":         strategy_key,
            "params":           merged_params,
            "tickers_tested":   list(all_results.keys()),
            "per_ticker":       all_results,
            "aggregate":        aggregate,
        }


# ---------------------------------------------------------------------------
# standalone runner — run all strategies on all tickers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    Path("logs").mkdir(exist_ok=True)

    agent = BacktestingAgent(data_dir=DATA_DIR)

    print("=" * 60)
    print("  backtesting agent — running all strategies")
    print("=" * 60)
    print()

    all_scores = []

    for strategy_name, entry in STRATEGIES.items():
        default_params = entry[1]
        if not default_params.get("active", True):
            print(f"strategy: {strategy_name}  [INACTIVE — skipped]")
            print()
            continue
        print(f"strategy: {strategy_name}")
        print("-" * 40)

        result = agent.run({
            "payload": {
                "name":    strategy_name,
                "tickers": ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"],
                "start":   "2022-01-01",
                "end":     "2025-01-01",
            }
        })

        if result["success"]:
            agg = result["aggregate"]
            print(f"  sharpe        : {agg['sharpe']:.4f}")
            print(f"  max drawdown  : {agg['max_drawdown']:.2%}")
            print(f"  win rate      : {agg['win_rate']:.2%}")
            print(f"  total trades  : {agg['total_trades']}")
            print(f"  calmar        : {agg['calmar']:.4f}")
            print()

            # per ticker breakdown
            for ticker, r in result["per_ticker"].items():
                print(f"  {ticker:<6} sharpe={r['sharpe']:>6.2f}  "
                      f"dd={r['max_drawdown']:>7.2%}  "
                      f"wr={r['win_rate']:>5.1%}  "
                      f"trades={r['total_trades']:>5}")

                # regime breakdown
                if r.get("regime_breakdown"):
                    for regime, rm in r["regime_breakdown"].items():
                        print(f"         [{regime:<7}] sharpe={rm['sharpe']:>6.2f}  "
                              f"trades={rm['total_trades']:>4}")
            print()

            all_scores.append({
                "strategy": strategy_name,
                "aggregate": agg,
            })
        else:
            print(f"  failed: {result.get('reason')}")
            print()

    # summary ranking
    print("=" * 60)
    print("  strategy ranking by sharpe")
    print("=" * 60)
    ranked = sorted(all_scores, key=lambda x: x["aggregate"]["sharpe"], reverse=True)
    for i, s in enumerate(ranked, 1):
        agg = s["aggregate"]
        print(f"  {i}. {s['strategy']:<20} sharpe={agg['sharpe']:>6.2f}  "
              f"dd={agg['max_drawdown']:>7.2%}  wr={agg['win_rate']:>5.1%}")

    # save results
    output_path = Path("results/backtest_results.json")
    with open(output_path, "w") as f:
        json.dump({"strategies": all_scores, "run_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(f"\n  results saved to {output_path}")
