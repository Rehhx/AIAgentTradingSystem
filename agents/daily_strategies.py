"""
agents/daily_strategies.py
--------------------------
DAILY / MULTI-DAY strategies — the search space the 1-minute engine never
touched. The project's whole history (LESSONS.md, champions.json) is intraday
1m bars, where 6bps round-trip cost dominates any signal (loss magnitude tracks
trade count almost perfectly). On daily holds, the same 6bps is amortized over a
multi-week move and becomes negligible, so the economics flip.

This module is SELF-CONTAINED and authoritative for the board numbers. It does
its own $100k portfolio accounting rather than routing through the intraday
engine (whose ATR stops / intraday regime gates would distort a daily strategy).
Costs and starting capital are imported from backtesting_agent so the basis is
identical (round-trip = 2*(COMMISSION+SLIPPAGE) = 6bps; INITIAL_CAP = $100k).

Strategies are LONG/FLAT (position in {0, 1}) — clean to deploy on Alpaca paper
(no borrow) and the honest unit for a board pitch.

  rsi2_meanrev      Connors RSI(2): buy short-term dips above the 200d trend.
  donchian          20-day-high breakout, exit 10-day low (Turtle-lite).
  trend_5020        50/200 SMA trend filter (long when 50>200).

A "book" is an equal-weight, daily-rebalanced portfolio of one strategy across a
universe. The "blended book" is the equal-weight combination of all three books.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.backtesting_agent import COMMISSION, SLIPPAGE, INITIAL_CAP
from data.loader import load_ticker

# round-trip cost in fractional terms: each side pays COMMISSION + SLIPPAGE.
RT_COST = 2 * (COMMISSION + SLIPPAGE)        # 0.0006 = 6 bps
SIDE_COST = COMMISSION + SLIPPAGE            # 0.0003 = 3 bps per side
TRADING_DAYS = 252

# a diversified, liquid default universe — indices + gold + large caps across
# sectors so no single name drives the result.
DEFAULT_UNIVERSE = ["SPY", "QQQ", "GLD", "MSFT", "JPM", "AAPL", "AMZN", "GOOGL"]

# the locked LIVE deployable universe: 10 liquid, sector-diversified quality
# names (index ETFs + gold + mega-cap tech + financials + healthcare + energy).
# Splits a balance between the 6-name book's return (~10.4% CAGR) and the full
# S&P 500's diversification: blended book ~Sharpe 1.23, 9.3% CAGR, -11.8% DD.
QUALITY_UNIVERSE = ["SPY", "QQQ", "GLD", "MSFT", "AAPL", "GOOGL", "AMZN",
                    "JPM", "UNH", "XOM"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

_DAILY_CACHE: dict[str, pd.DataFrame] = {}


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """resample 1m OHLCV bars to daily OHLCV (one row per UTC calendar day)."""
    out = pd.DataFrame({
        "open":  df["open"].resample("1D").first(),
        "high":  df["high"].resample("1D").max(),
        "low":   df["low"].resample("1D").min(),
        "close": df["close"].resample("1D").last(),
        "volume": df["volume"].resample("1D").sum(),
    }).dropna()
    return out


# default to SPLIT/DIVIDEND-ADJUSTED daily bars (yfinance) so split stocks
# (GOOGL, AAPL, AMZN, TSLA, NVDA...) are safe. The local 1m parquet is RAW and
# carries unadjusted splits — verified via runners/verify_trades_vs_yfinance.py.
# Set DAILY_USE_ADJUSTED=0 to force the raw parquet.
import os as _os
USE_ADJUSTED = _os.getenv("DAILY_USE_ADJUSTED", "1") != "0"


def daily_bars(ticker: str) -> pd.DataFrame:
    if ticker in _DAILY_CACHE:
        return _DAILY_CACHE[ticker]
    df = None
    if USE_ADJUSTED:
        try:
            from data.sp500 import load_daily
            df = load_daily([ticker], start="2016-01-01").get(ticker)
        except Exception:
            df = None
    if df is None or len(df) < 50:
        df = to_daily(load_ticker(ticker))      # raw parquet fallback
    _DAILY_CACHE[ticker] = df
    return df


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# signal generators — each returns a daily long/flat position series {0.0, 1.0}
# aligned to d.index. NO shift here; the backtester shifts by one day so a
# signal computed from today's close is entered at tomorrow's close.
# ---------------------------------------------------------------------------

def sig_rsi2_meanrev(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = params or {}
    entry_rsi = p.get("entry_rsi", 10)
    exit_rsi  = p.get("exit_rsi", 70)
    trend_sma = p.get("trend_sma", 200)
    c = d["close"]
    r = _rsi(c, p.get("rsi_period", 2))
    above_trend = c > c.rolling(trend_sma).mean()
    enter = (r < entry_rsi) & above_trend
    exit_ = r > exit_rsi
    return _state_machine(enter, exit_, c.index)


def sig_donchian(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = params or {}
    entry_n = p.get("entry_lookback", 20)
    exit_n  = p.get("exit_lookback", 10)
    c = d["close"]
    hi = c.rolling(entry_n).max()
    lo = c.rolling(exit_n).min()
    enter = c >= hi
    exit_ = c <= lo
    return _state_machine(enter, exit_, c.index)


def sig_trend_5020(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    p = params or {}
    fast = p.get("fast", 50)
    slow = p.get("slow", 200)
    c = d["close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)


def _state_machine(enter: pd.Series, exit_: pd.Series, index) -> pd.Series:
    """walk enter/exit booleans into a held long/flat position series."""
    enter = enter.to_numpy()
    exit_ = exit_.to_numpy()
    out = np.zeros(len(index))
    state = 0
    for i in range(len(index)):
        if state == 0 and enter[i]:
            state = 1
        elif state == 1 and exit_[i]:
            state = 0
        out[i] = state
    return pd.Series(out, index=index, dtype=float)


STRATEGIES_DAILY = {
    "rsi2_meanrev": sig_rsi2_meanrev,
    "donchian":     sig_donchian,
    "trend_5020":   sig_trend_5020,
}

# Deployment parameters — single source of truth for what we actually trade.
# RSI-2 uses the WALK-FORWARD-OPTIMIZED settings (entry_rsi=30/exit_rsi=50) that
# clear the desk's >=100 trades/year floor while staying profitable out-of-sample
# (OOS Sharpe 1.09, -9.7% max DD, 112 trades/yr). Origin: runners/walk_forward_daily.py.
# {} means "use the signal function's built-in defaults".
DEPLOY_PARAMS = {
    "rsi2_meanrev": {"rsi_period": 2, "entry_rsi": 30, "exit_rsi": 50, "trend_sma": 100},
    "donchian":     {},
    "trend_5020":   {},
}


# ---------------------------------------------------------------------------
# CANDIDATE strategies — new mechanisms under evaluation. Kept separate from the
# deployed STRATEGIES_DAILY so research doesn't disturb the live blend. Promote
# into STRATEGIES_DAILY only if they improve the blended book (see strategy_lab).
# ---------------------------------------------------------------------------

def sig_turn_of_month(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Long over the turn of the month: the last `pre` trading days of a month
    plus the first `post` of the next. Captures the documented month-end /
    start-of-month inflow effect. Near-zero correlation to trend/reversion."""
    p = params or {}
    pre, post = p.get("pre", 1), p.get("post", 3)
    idx = d.index
    m = idx.tz_localize(None).to_period("M") if getattr(idx, "tz", None) else idx.to_period("M")
    rank = pd.Series(1, index=idx).groupby(m).cumsum()           # 1..N within month
    count = pd.Series(1, index=idx).groupby(m).transform("sum")
    long = (rank <= post) | (rank > (count - pre))
    return long.astype(float)


def sig_zscore_revert(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Mean reversion on the z-score of close vs its N-day mean, gated by a
    long-term trend filter. Longer lookback than RSI-2 (different horizon)."""
    p = params or {}
    lb = p.get("lookback", 20)
    entry_z, exit_z = p.get("entry_z", -2.0), p.get("exit_z", 0.0)
    trend_sma = p.get("trend_sma", 200)
    c = d["close"]
    mu = c.rolling(lb).mean()
    sd = c.rolling(lb).std(ddof=0).replace(0, np.nan)
    z = (c - mu) / sd
    above = c > c.rolling(trend_sma).mean()
    enter = (z < entry_z) & above
    exit_ = z > exit_z
    return _state_machine(enter, exit_, c.index)


def sig_trend_band(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """50/200 trend with an ANTI-WHIPSAW band: go long only when the fast MA is
    `band` above the slow MA, and exit only when it's `band` below — holding
    through small crosses. Reduces the buy-high/sell-low chop of a bare SMA cross
    in V-shaped markets (Q4-2018, COVID)."""
    p = params or {}
    fast, slow, band = p.get("fast", 50), p.get("slow", 200), p.get("band", 0.03)
    c = d["close"]
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    return _state_machine(f > s * (1 + band), f < s * (1 - band), c.index)


def sig_trend_multi(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Multi-speed trend: graded exposure = fraction of fast/medium/slow SMA
    crosses that are long. The fast pair re-enters a recovery weeks before the
    slow 50/200 (catching V-snapbacks like early 2019 / mid-2020); the slow pair
    damps whipsaw. Diversifies the lookback instead of betting on one timing."""
    p = params or {}
    speeds = p.get("speeds", [(20, 100), (50, 200), (100, 300)])
    c = d["close"]
    sigs = [(c.rolling(f).mean() > c.rolling(s).mean()).astype(float) for f, s in speeds]
    return sum(sigs) / len(sigs)          # 0.0, 0.33, 0.67, 1.0 graded exposure


def sig_recovery(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Recovery-thrust: catch the START of a bull run. When price RECLAIMS its
    50-day average after having been below its 200-day (a snapback off a low —
    fired early-2019 and spring-2020), go long and hold for `hold_days` to ride
    the recovery. Targets the V-recoveries the slow 50/200 trend re-enters late."""
    p = params or {}
    hold = int(p.get("hold_days", 120))
    c = d["close"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    reclaim = (c > sma50) & (c.shift(1) <= sma50.shift(1))          # crosses up through 50d
    recently_below = (c < sma200).rolling(30).max().fillna(0).astype(bool)
    thrust = (reclaim & recently_below).to_numpy()
    pos, left = np.zeros(len(c)), 0
    for i in range(len(c)):
        if thrust[i]:
            left = hold
        if left > 0:
            pos[i] = 1.0
            left -= 1
    return pd.Series(pos, index=c.index, dtype=float)


def sig_abs_momentum(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Absolute (time-series) momentum: hold long while trailing N-day return is
    positive. Slower, longer-horizon trend than the 50/200 SMA cross."""
    p = params or {}
    lb = p.get("lookback", 126)            # ~6 months
    thr = p.get("threshold", 0.0)
    c = d["close"]
    mom = c / c.shift(lb) - 1
    return (mom > thr).astype(float)


def sig_capitulation(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Capitulation dip-buyer: buys EXTREME oversold WITHOUT a trend filter, so
    it can catch crash bottoms (Dec-2018, Mar-2020) that trend-gated RSI-2 sits
    out. Optional drop confirmation (price well below a recent high) keeps it to
    genuine washouts, not every wobble. Exits into the bounce."""
    p = params or {}
    entry_rsi = p.get("entry_rsi", 5)        # extreme oversold
    exit_rsi  = p.get("exit_rsi", 55)
    drop_pct  = p.get("drop_pct", 0.07)      # require >=7% below the recent high
    high_lb   = p.get("high_lb", 10)
    c = d["close"]
    r = _rsi(c, p.get("rsi_period", 2))
    enter = (r < entry_rsi)
    if drop_pct > 0:
        enter = enter & (c < c.rolling(high_lb).max() * (1 - drop_pct))
    exit_ = r > exit_rsi
    return _state_machine(enter, exit_, c.index)


def sig_pead(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Post-Earnings-Announcement Drift (price-proxy). An earnings BEAT shows up
    as a large up-gap on a volume spike; the stock then drifts up for weeks. So:
    detect (1-day return > gap_pct) AND (volume > vol_mult x its 20d average) as
    the 'beat' event, then hold long for `hold_days` to ride the drift. Event-
    driven and largely uncorrelated to continuous price factors. Long/flat."""
    p = params or {}
    gap_pct  = p.get("gap_pct", 0.05)
    vol_mult = p.get("vol_mult", 2.0)
    hold     = int(p.get("hold_days", 40))
    c, v = d["close"], d["volume"]
    ret = c.pct_change()
    event = ((ret > gap_pct) & (v > v.rolling(20).mean() * vol_mult)).to_numpy()
    pos = np.zeros(len(c))
    left = 0
    for i in range(len(c)):
        if event[i]:
            left = hold                     # (re)start the drift-hold window on a fresh beat
        if left > 0:
            pos[i] = 1.0
            left -= 1
    return pd.Series(pos, index=c.index, dtype=float)


CANDIDATE_STRATEGIES = {
    "turn_of_month": sig_turn_of_month,
    "zscore_revert": sig_zscore_revert,
    "abs_momentum":  sig_abs_momentum,
    "capitulation":  sig_capitulation,
    "pead":          sig_pead,
    "trend_multi":   sig_trend_multi,
    "recovery":      sig_recovery,
}


# ---------------------------------------------------------------------------
# CROSS-SECTIONAL book — a portfolio-level signal (ranks the universe each day),
# not a per-ticker sleeve. Long-only: hold the top-k by score, equal weight.
#   mode="momentum": score = trailing return (skip the most recent month to
#       avoid short-term reversal contamination) -> hold relative winners.
#   mode="reversal": score = -trailing short return -> hold relative losers
#       (cross-sectional mean reversion).
# ---------------------------------------------------------------------------

def _xs_trades(W: pd.DataFrame, panel: pd.DataFrame) -> list:
    """reconstruct per-ticker holding spells from the weight matrix for
    win-rate / trade-count (the $PnL itself comes from the costed return series)."""
    trades = []
    idx = panel.index
    for t in W.columns:
        held = (W[t].to_numpy() > 0)
        px = panel[t].to_numpy()
        in_pos, entry_i = False, None
        for i in range(len(held)):
            if not in_pos and held[i]:
                in_pos, entry_i = True, i
            elif in_pos and not held[i]:
                if not np.isnan(px[entry_i]) and px[entry_i] > 0 and not np.isnan(px[i]):
                    trades.append({"entry": idx[entry_i], "exit": idx[i],
                                   "ret": px[i] / px[entry_i] - 1 - RT_COST,
                                   "bars": i - entry_i, "ticker": t})
                in_pos = False
        if in_pos and not np.isnan(px[entry_i]) and px[entry_i] > 0:
            trades.append({"entry": idx[entry_i], "exit": idx[-1],
                           "ret": px[-1] / px[entry_i] - 1 - RT_COST,
                           "bars": len(held) - 1 - entry_i, "ticker": t, "open": True})
    return trades


def backtest_cross_sectional(universe=None, mode: str = "momentum",
                             lookback: int = 126, skip: int = 21, k: int = 3,
                             market_filter: bool = False, market_ticker: str = "SPY",
                             market_sma: int = 200, market_band: float = 0.0,
                             label: str | None = None) -> dict:
    """equal-weight long-only cross-sectional book over the universe, $100k base.

    market_filter=True turns this into DUAL MOMENTUM: hold the top-k by relative
    strength ONLY while the market (market_ticker) is above its market_sma-day
    average; otherwise go to cash. This sidesteps bear-market crashes and slashes
    the drawdown of a raw relative-strength book."""
    universe = universe or QUALITY_UNIVERSE
    closes = {}
    for t in universe:
        try:
            closes[t] = daily_bars(t)["close"]
        except Exception:
            continue
    panel = pd.concat(closes, axis=1)
    panel.columns = list(closes)
    rets = panel.pct_change()

    if mode == "reversal":
        score = -(panel / panel.shift(lookback) - 1)          # losers rank highest
    else:                                                      # momentum (skip recent month)
        score = panel.shift(skip) / panel.shift(skip + lookback) - 1

    cols = list(panel.columns)
    W = pd.DataFrame(0.0, index=panel.index, columns=cols)
    for i in range(len(panel.index)):
        row = score.iloc[i].dropna()
        if len(row) < k:
            continue
        top = row.nlargest(k).index
        W.iloc[i, [cols.index(t) for t in top]] = 1.0 / k

    if market_filter:                                         # dual-momentum gate
        try:
            mkt = daily_bars(market_ticker)["close"]
            sma = mkt.rolling(market_sma).mean()
            if market_band > 0:                               # anti-whipsaw hysteresis
                ratio = mkt / sma
                above = _state_machine(ratio > (1 + market_band),
                                       ratio < (1 - market_band), mkt.index).astype(bool)
            else:
                above = (mkt > sma)
            above = above.reindex(W.index).ffill().fillna(False)
            W = W.mul(above.astype(float), axis=0)            # cash when market < SMA
        except Exception:
            pass

    W = W.shift(1).fillna(0.0)                                 # decide on close, enter next day

    gross = (W * rets).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    port = gross - turnover * SIDE_COST
    m = _metrics_from_returns(port, _xs_trades(W, panel), label or f"xs_{mode}")
    m["universe"] = cols
    m["mode"], m["lookback"], m["skip"], m["k"] = mode, lookback, skip, k
    m["market_filter"] = market_filter
    return m


# ---------------------------------------------------------------------------
# per-sleeve (single ticker) backtest -> net daily return series + trade list
# ---------------------------------------------------------------------------

def sleeve_returns(d: pd.DataFrame, sig_fn, params: dict | None = None):
    """returns (net_daily_return: Series, trades: list[dict]) for one ticker.

    Cost: SIDE_COST charged on every position change (entry and exit), so a
    full round trip pays RT_COST. Returns are close-to-close; position is
    shifted one day to remove lookahead (signal from today's close is entered
    at tomorrow's close)."""
    c = d["close"]
    pos = sig_fn(d, params).shift(1).fillna(0.0)
    ret = c.pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())          # first bar entry counts
    net = pos * ret - turn * SIDE_COST
    net.name = None

    # per-trade reconstruction for win-rate / trade-count
    trades = []
    in_pos = False
    entry_i = None
    p = pos.to_numpy()
    idx = c.index
    px = c.to_numpy()
    for i in range(len(p)):
        if not in_pos and p[i] > 0:
            in_pos, entry_i = True, i
        elif in_pos and p[i] == 0:
            entry_px = float(px[entry_i])
            exit_px  = float(px[i])
            pnl_ret = exit_px / entry_px - 1 - RT_COST
            trades.append({
                "entry": idx[entry_i], "exit": idx[i],
                "entry_px": entry_px, "exit_px": exit_px,
                "ret": pnl_ret, "bars": i - entry_i, "open": False,
            })
            in_pos = False
    if in_pos:  # mark open position to last bar (counts as a trade for stats)
        entry_px, exit_px = float(px[entry_i]), float(px[-1])
        trades.append({
            "entry": idx[entry_i], "exit": idx[-1],
            "entry_px": entry_px, "exit_px": exit_px,
            "ret": exit_px / entry_px - 1 - RT_COST, "bars": len(p) - 1 - entry_i,
            "open": True,
        })
    return net, trades


# ---------------------------------------------------------------------------
# portfolio backtest (equal-weight, daily rebalance) -> $100k book + metrics
# ---------------------------------------------------------------------------

def _metrics_from_returns(port_ret: pd.Series, trades: list, label: str) -> dict:
    port_ret = port_ret.fillna(0.0)
    eq = INITIAL_CAP * (1 + port_ret).cumprod()
    final = float(eq.iloc[-1]) if len(eq) else float(INITIAL_CAP)
    std = port_ret.std()
    sharpe = float(port_ret.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    n_days = int((port_ret != 0).sum())
    years = len(port_ret) / TRADING_DAYS if len(port_ret) else 1
    cagr = float((final / INITIAL_CAP) ** (1 / years) - 1) if years > 0 and final > 0 else 0.0
    wins = sum(1 for t in trades if t["ret"] > 0)
    n_tr = len(trades)
    win_rate = wins / n_tr if n_tr else 0.0
    avg_hold = float(np.mean([t["bars"] for t in trades])) if trades else 0.0
    return {
        "book": label,
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 4),
        "max_drawdown": round(dd, 4),
        "win_rate": round(win_rate, 4),
        "total_trades": n_tr,
        "pnl_dollars": round(final - INITIAL_CAP, 2),
        "final_capital": round(final, 2),
        "total_return": round(final / INITIAL_CAP - 1, 4),
        "avg_hold_days": round(avg_hold, 1),
        "exposure_pct": round(float((port_ret != 0).mean()), 3),
        "n_obs": len(port_ret),
        "_equity": eq,
        "_returns": port_ret,
    }


def backtest_book(sig_fn, universe=None, params: dict | None = None,
                  label: str = "book") -> dict:
    """equal-weight, daily-rebalanced $100k portfolio of one strategy across the
    universe. Returns a metrics dict + per-ticker breakdown."""
    universe = universe or DEFAULT_UNIVERSE
    sleeves, all_trades, per_ticker = {}, [], {}
    for t in universe:
        try:
            d = daily_bars(t)
        except Exception:
            continue
        net, trades = sleeve_returns(d, sig_fn, params)
        sleeves[t] = net
        for tr in trades:
            tr["ticker"] = t
        all_trades.extend(trades)
        per_ticker[t] = _metrics_from_returns(net, trades, t)
    if not sleeves:
        return {"book": label, "error": "no data"}
    # mean(axis=1) skips NaN: a ticker only contributes on dates it has data,
    # so a parquet that ends early drops out of the equal-weight rather than
    # being treated as a 0%-return cash sleeve (which would bias results).
    port_ret = pd.concat(sleeves.values(), axis=1).mean(axis=1)
    m = _metrics_from_returns(port_ret, all_trades, label)
    m["per_ticker"] = {t: {k: v for k, v in pm.items() if not k.startswith("_")}
                       for t, pm in per_ticker.items()}
    m["universe"] = list(sleeves.keys())
    return m


def backtest_blended(universe=None, params: dict | None = None,
                     label: str = "blended_book", weights: dict | None = None) -> dict:
    """capital-weighted combination of the three strategy books. Daily portfolio
    return is the weighted average of the book return series.

    weights: {strategy_name: weight}. Defaults to equal weight. Weights are
    normalized to sum to 1, so e.g. {"trend_5020": 0.5, "rsi2_meanrev": 0.3,
    "donchian": 0.2} tilts capital toward trend. A weight of 0 drops a sleeve."""
    universe = universe or DEFAULT_UNIVERSE
    params = params or {}
    books, ret_by_name, all_trades = {}, {}, []
    for name, fn in STRATEGIES_DAILY.items():
        w = 1.0 if weights is None else float(weights.get(name, 0.0))
        if w <= 0:
            continue
        b = backtest_book(fn, universe, params.get(name), label=name)
        if "error" in b:
            continue
        books[name] = b
        ret_by_name[name] = b["_returns"]
        # collect per-sleeve trades (scaled-in proportion handled at $ level only;
        # WR/trade-count is a property of the signal, so count every round-trip)
        for t in universe:
            try:
                d = daily_bars(t)
            except Exception:
                continue
            _, trs = sleeve_returns(d, fn, (params.get(name)))
            all_trades.extend(trs)
    if not ret_by_name:
        return {"book": label, "error": "no books"}
    names = list(ret_by_name)
    w = np.array([1.0 if weights is None else float(weights.get(n, 0.0)) for n in names])
    w = w / w.sum()
    panel = pd.concat([ret_by_name[n] for n in names], axis=1)
    panel.columns = names
    port_ret = (panel * w).sum(axis=1, min_count=1)
    m = _metrics_from_returns(port_ret, all_trades, label)
    m["components"] = {name: {k: v for k, v in b.items() if not k.startswith("_")
                              and k != "per_ticker"}
                       for name, b in books.items()}
    m["weights"] = {n: round(float(wi), 3) for n, wi in zip(names, w)}
    m["universe"] = universe
    return m


# ---------------------------------------------------------------------------
# walk-forward: sequential contiguous folds (shows regime stability, not 1 split)
# ---------------------------------------------------------------------------

def backtest_long_short(universe=None, mode: str = "momentum", lookback: int = 252,
                        skip: int = 21, k: int = 50, borrow_annual: float = 0.02,
                        label: str | None = None) -> dict:
    """DOLLAR-NEUTRAL long/short book: each day go +50% long the top-k by signal
    and -50% short the bottom-k (net beta ~0), $100k base. Profits from the
    winner-minus-loser SPREAD regardless of market direction — a genuinely
    uncorrelated stream. Charges 6bps on turnover + a borrow cost on the short leg.
      mode="momentum": long winners / short losers (12-1 momentum factor)
      mode="reversal": long recent losers / short recent winners (short-term reversal)"""
    universe = universe or QUALITY_UNIVERSE
    closes = {}
    for t in universe:
        try:
            closes[t] = daily_bars(t)["close"]
        except Exception:
            continue
    panel = pd.concat(closes, axis=1); panel.columns = list(closes)
    rets = panel.pct_change()
    if mode == "reversal":
        score = -(panel / panel.shift(lookback) - 1)
    else:
        score = panel.shift(skip) / panel.shift(skip + lookback) - 1

    cols = list(panel.columns)
    W = pd.DataFrame(0.0, index=panel.index, columns=cols)
    for i in range(len(panel.index)):
        row = score.iloc[i].dropna()
        if len(row) < 2 * k:
            continue
        longs = row.nlargest(k).index
        shorts = row.nsmallest(k).index
        W.iloc[i, [cols.index(t) for t in longs]] = 0.5 / k
        W.iloc[i, [cols.index(t) for t in shorts]] = -0.5 / k
    W = W.shift(1).fillna(0.0)

    gross = (W * rets).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    borrow = W.clip(upper=0).abs().sum(axis=1) * (borrow_annual / TRADING_DAYS)
    port = gross - turnover * SIDE_COST - borrow
    m = _metrics_from_returns(port, [], label or f"ls_{mode}")
    # market exposure: beta + correlation to SPY (the value of neutrality)
    try:
        spy = daily_bars("SPY")["close"].pct_change().reindex(port.index).fillna(0.0)
        v = spy.var()
        m["beta_to_spy"] = round(float((port.cov(spy) / v) if v else 0.0), 3)
        m["corr_to_spy"] = round(float(port.corr(spy)), 3)
    except Exception:
        pass
    m["mode"], m["k"] = mode, k
    return m


def vol_target(returns: pd.Series, target_vol: float = 0.12, window: int = 20,
               max_leverage: float = 1.0) -> pd.Series:
    """volatility-targeting overlay: scale each day's exposure so realized vol
    tracks target_vol. Uses YESTERDAY's vol estimate (no lookahead). Capped at
    max_leverage (1.0 = de-risk only, never lever). Cuts drawdowns sharply in
    high-vol regimes (crashes) while leaving calm periods near full exposure."""
    r = returns.fillna(0.0)
    rv = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / rv.replace(0, np.nan)).clip(upper=max_leverage)
    scale = scale.shift(1).fillna(0.0)
    return r * scale


def walk_forward_folds(returns: pd.Series, n_folds: int = 5) -> list[dict]:
    """split the return series into n contiguous folds and report Sharpe + PnL
    on each. A strategy with edge should be positive in most folds, not just
    one lucky window."""
    returns = returns.fillna(0.0)
    fold_len = len(returns) // n_folds
    out = []
    for k in range(n_folds):
        seg = returns.iloc[k * fold_len:(k + 1) * fold_len] if k < n_folds - 1 \
            else returns.iloc[k * fold_len:]
        if len(seg) < 10 or seg.std() == 0:
            out.append({"fold": k + 1, "sharpe": 0.0, "return_pct": 0.0})
            continue
        sh = seg.mean() / seg.std() * np.sqrt(TRADING_DAYS)
        out.append({
            "fold": k + 1,
            "start": str(seg.index[0].date()),
            "end": str(seg.index[-1].date()),
            "sharpe": round(float(sh), 3),
            "return_pct": round(float((1 + seg).prod() - 1), 4),
        })
    return out


# ---------------------------------------------------------------------------
# integration: register daily strategies into the engine's STRATEGIES registry
# so they flow through the pipeline + risk_agent like any other strategy.
# ---------------------------------------------------------------------------

def make_engine_adapter(sig_fn):
    """wrap a daily signal generator as a signals(df, params) function the
    intraday engine can consume. Resamples the incoming 1m bars to daily,
    computes the daily long/flat position, SHIFTS IT ONE DAY (so day D's
    intraday bars trade on the signal known at D-1's close — no lookahead),
    then forward-fills onto the 1m index."""
    def signals(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
        params = params or {}
        d = to_daily(df)
        if len(d) < 220:                       # need 200d SMA warmup
            return pd.Series(0, index=df.index, dtype=int)
        daily_pos = sig_fn(d, params).shift(1)              # enter next day
        s = daily_pos.reindex(df.index, method="ffill").fillna(0.0)
        return s.astype(int)
    return signals


def register_daily_strategies(overwrite: bool = False) -> list[str]:
    """register daily_rsi2_meanrev / daily_donchian / daily_trend_5020 into
    STRATEGIES with ATR-stop and max-hold disabled (daily strategies manage
    their own exits)."""
    from agents.backtesting_agent import register_strategy
    registered = []
    for name, fn in STRATEGIES_DAILY.items():
        key = f"daily_{name}"
        try:
            register_strategy(
                key, make_engine_adapter(fn),
                default_params={"disable_atr_stop": True, "max_hold_bars": None,
                                "active": True},
                overwrite=overwrite,
            )
            registered.append(key)
        except ValueError:
            pass  # already registered
    return registered


def split_metrics(returns: pd.Series, train_frac: float = 0.7) -> dict:
    """70/30 in-sample / out-of-sample Sharpe."""
    returns = returns.fillna(0.0)
    s = int(len(returns) * train_frac)
    def sh(r):
        r = r.dropna()
        return round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)), 3) if r.std() > 0 else 0.0
    return {"train_sharpe": sh(returns.iloc[:s]), "test_sharpe": sh(returns.iloc[s:])}
