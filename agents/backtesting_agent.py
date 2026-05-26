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
COMMISSION  = 0.0005   # 0.05% per trade (realistic for retail)
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


# which regimes each strategy is allowed to fire in. derived from the
# strategy's structural assumption (trend-following vs mean-reverting).
# RSI/VWAP narrowed to "chop" only — the "mean_reversion" bucket is the
# heuristic's fallback class and still contains too much directional drift
# for a snapback trade to work reliably.
STRATEGY_REGIME_AFFINITY = {
    "rsi_reversion":  {"chop"},
    "vwap_reversion": {"chop"},
    "orb":            {"breakout", "trending"},
    "momentum":       {"trending", "breakout"},
    "bb_squeeze":     {"breakout"},
    "ema_crossover":  {"trending"},
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
STRATEGIES = {
    "rsi_reversion":  (signals_rsi_reversion,  {"rsi_period": 14, "oversold": 30, "overbought": 70, "stop_atr_mult": 3.5, "active": False}),
    "vwap_reversion": (signals_vwap_reversion,  {"threshold_pct": 0.003, "stop_atr_mult": 3.5,                         "active": False}),
    "orb":            (signals_orb,             {"orb_minutes": 15, "stop_atr_mult": 1.5,                              "active": True}),
    "momentum":       (signals_momentum,        {"lookback_bars": 20, "volume_zscore_min": 1.0, "stop_atr_mult": 1.5,  "active": True}),
    "bb_squeeze":     (signals_bb_squeeze,      {"bb_period": 20, "bb_std": 2.0, "kc_mult": 1.5, "stop_atr_mult": 2.0, "active": True}),
    "ema_crossover":  (signals_ema_crossover,   {"fast_period": 9, "slow_period": 21, "stop_atr_mult": 1.5,            "active": True}),
}


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
    shares   = 0
    last_exit_bar = -REENTRY_COOLDOWN_BARS   # allow first entry immediately

    close = df["close"]

    for i in range(1, len(df)):
        ts  = df.index[i]
        c   = close.iloc[i]
        sig = int(signal.iloc[i])
        prev_sig = int(signal.iloc[i - 1])

        # ATR stop: override sig to flat if the open position has moved
        # adversely by >= ATR_STOP_MULT * ATR. Existing exit branch then fires.
        if pos != 0:
            atr_val = atr_series.iloc[i]
            if not pd.isna(atr_val) and atr_val > 0:
                adverse = (entry_px - c) * pos   # > 0 when losing
                if adverse >= stop_atr_mult * atr_val:
                    sig = 0

        # entry — only if cooldown elapsed AND regime is compatible AND the
        # embedding-based quality gate (if provided) confirms the direction.
        # open positions are not force-exited on regime/quality change.
        regime_ok = (
            allowed_regimes is None
            or regime_series is None
            or regime_series.iloc[i] in allowed_regimes
        )
        quality_ok = True
        if quality_series is not None and sig != 0 and i < len(quality_series):
            pct_pos = quality_series["fwd_pct_positive"].iloc[i]
            if not pd.isna(pct_pos):
                if sig == 1:
                    quality_ok = pct_pos >= quality_min_pct_pos
                else:  # sig == -1
                    quality_ok = pct_pos <= (1.0 - quality_min_pct_pos)
        if pos == 0 and sig != 0 and regime_ok and quality_ok and (i - last_exit_bar) >= REENTRY_COOLDOWN_BARS:
            pos      = sig
            entry_px = c * (1 + SLIPPAGE * sig)   # slip in direction of trade
            shares   = int((capital * position_size_pct) / entry_px)
            entry_ts = ts
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
    fn, default_params = STRATEGIES[strategy_name]
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

    fn, default_params = STRATEGIES[strategy_name]
    stop_mult = default_params.get("stop_atr_mult", ATR_STOP_MULT)
    allowed   = STRATEGY_REGIME_AFFINITY.get(strategy_name)

    df = load_ticker(ticker, data_dir=data_dir, start=start, end=end, session="regular")
    cut       = int(len(df) * train_pct)
    train_df  = df.iloc[:cut]
    test_df   = df.iloc[cut:]

    quality_df  = precompute_regime_quality(ticker, df, step=60)
    quality_all = quality_df.reindex(df.index, method="ffill")
    quality_tr  = quality_all.iloc[:cut]
    quality_te  = quality_all.iloc[cut:]

    train_signal = fn(train_df, default_params)
    test_signal  = fn(test_df,  default_params)
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

        # find matching strategy signal function
        strategy_key = None
        for key in STRATEGIES:
            if key in name or name in key:
                strategy_key = key
                break

        if not strategy_key:
            strategy_key = "rsi_reversion"   # default fallback
            self.log.warning(f"unknown strategy '{name}', defaulting to rsi_reversion")

        fn, default_params = STRATEGIES[strategy_key]
        merged_params      = {**default_params, **params}

        self.log.info(f"backtesting {strategy_key} | tickers={tickers} | {start} to {end}")

        all_results = {}
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

                signal    = fn(df, merged_params)
                stop_mult = merged_params.get("stop_atr_mult", ATR_STOP_MULT)
                regime_s  = regime_label_series(df)
                allowed   = STRATEGY_REGIME_AFFINITY.get(strategy_key)

                results = run_backtest(
                    df, signal,
                    stop_atr_mult=stop_mult,
                    regime_series=regime_s,
                    allowed_regimes=allowed,
                )
                regimes = backtest_by_regime(
                    results, df, signal,
                    stop_atr_mult=stop_mult,
                    regime_series=regime_s,
                    allowed_regimes=allowed,
                )

                results["regime_breakdown"] = regimes
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

    for strategy_name, (_, default_params) in STRATEGIES.items():
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
