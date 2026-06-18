"""
agents/lab_strategies.py
------------------------
ORIGINAL strategy mechanisms invented in-house for the autonomous agent lab.

These are deliberately NOT the textbook sleeves in daily_strategies.py (RSI-2,
Bollinger, Donchian, 52-week-high, PEAD ...). Every signal here is a first-
principles construction with its own parameters: volatility coils, drawdown
ladders, price acceleration, path-monotonicity, ATR-stretch gravity, internal
breadth thrust, etc. Where a mechanism rhymes with something well known it is
re-cast with a different state machine and a distinct parameter set so it is a
genuinely separate return stream - the thing the ensemble bench rewards is
DECORRELATION, not another copy of trend/reversion we already own.

Contract (same as daily_strategies): each sig_* takes daily OHLCV `d` and
returns a position series in [0, 1] aligned to d.index. NO shift here - the
backtester shifts one day, so a signal off today's close is entered tomorrow.

Each strategy is owned by one of the 12 lab "agents" (LAB_AGENTS). The agent
researches the hypothesis, the orchestrator (runners/agent_lab.py) builds,
validates and dry-run-executes it, and a human approves or rejects the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents.daily_strategies import _state_machine


# ---------------------------------------------------------------------------
# small in-house primitives (no public-library indicators)
# ---------------------------------------------------------------------------

def _atr(d: pd.DataFrame, n: int) -> pd.Series:
    """average true range - our own, not imported."""
    h, l, c = d["high"], d["low"], d["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n // 2)).mean()


def _hold(events, hold: int, index) -> pd.Series:
    """latch a 1.0 position for `hold` bars from each True event (re-armable)."""
    ev = np.asarray(events, dtype=bool)
    pos = np.zeros(len(index))
    left = 0
    for i in range(len(index)):
        if ev[i]:
            left = hold
        if left > 0:
            pos[i] = 1.0
            left -= 1
    return pd.Series(pos, index=index, dtype=float)


# ===========================================================================
# 12 original mechanisms
# ===========================================================================

def sig_coil_release(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """VOLATILITY COIL RELEASE. When a short range compresses far inside the
    long range (a 'coil') and price then sits in the upper half of that coil
    above trend, energy releases upward - ride it for `hold` days. We trade the
    compression->expansion transition, not a level."""
    p = params or {}
    coil, ref = p.get("coil", 8), p.get("ref", 60)
    tight, hold, trend = p.get("tight_pct", 0.30), p.get("hold", 10), p.get("trend", 100)
    c = d["close"]
    rng_s = d["high"].rolling(coil).max() - d["low"].rolling(coil).min()
    rng_l = (d["high"].rolling(ref).max() - d["low"].rolling(ref).min()).replace(0, np.nan)
    comp = rng_s / rng_l
    coiled = comp < comp.rolling(ref, min_periods=ref // 2).quantile(tight)
    mid = (d["high"].rolling(coil).max() + d["low"].rolling(coil).min()) / 2
    above = c > c.rolling(trend, min_periods=trend // 2).mean()
    fire = (coiled & (c > mid) & above).fillna(False)
    return _hold(fire.to_numpy(), hold, c.index)


def sig_drawdown_ladder(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """DRAWDOWN LADDER. In a secular uptrend, accumulate in graded rungs as the
    drop from the recent peak deepens (the deeper the dip, the larger the
    position), and bleed off as it recovers. A continuous dip-accumulator, not
    an on/off dip-buyer - so its exposure path is unlike any trend sleeve."""
    p = params or {}
    rungs = p.get("rungs", [0.04, 0.08, 0.13])
    high_lb, trend = p.get("high_lb", 100), p.get("trend", 250)
    c = d["close"]
    peak = c.rolling(high_lb, min_periods=20).max()
    dd = c / peak - 1.0                                   # <= 0
    secular = (c > c.rolling(trend, min_periods=trend // 2).mean()).astype(float)
    expo = pd.Series(0.0, index=c.index)
    step = 1.0 / len(rungs)
    for r in rungs:
        expo = expo + step * ((-dd) >= r).astype(float)
    return (expo.clip(0, 1) * secular).fillna(0.0)


def sig_velocity_flip(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """ACCELERATION FLIP. Take the 2nd derivative of smoothed price (velocity,
    then its change = acceleration). Enter when acceleration crosses from
    negative to positive above trend - the inflection that PRECEDES a trend
    cross - and hold. Catches turns the 50/200 confirms weeks late."""
    p = params or {}
    smooth, slope = p.get("smooth", 10), p.get("slope", 5)
    trend, hold = p.get("trend", 100), p.get("hold", 12)
    c = d["close"]
    sm = c.rolling(smooth).mean()
    vel = sm - sm.shift(slope)
    acc = vel - vel.shift(slope)
    above = c > c.rolling(trend, min_periods=trend // 2).mean()
    flip = (acc > 0) & (acc.shift(1) <= 0) & above
    return _hold(flip.fillna(False).to_numpy(), hold, c.index)


def sig_trend_persistence(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """PATH PERSISTENCE. Long only when the path is MONOTONE up - a high
    fraction of up-days over the window AND positive drift AND above trend.
    Bets on the smoothness/quality of the climb, not its slope, so it stands
    aside in choppy 'going-nowhere-violently' tapes a pure trend filter holds."""
    p = params or {}
    win, thr, trend = p.get("win", 20), p.get("persist_thr", 0.62), p.get("trend", 120)
    c = d["close"]
    frac_up = (c.diff() > 0).astype(float).rolling(win).mean()
    drift = c / c.shift(win) - 1.0
    above = c > c.rolling(trend, min_periods=trend // 2).mean()
    return ((frac_up > thr) & (drift > 0) & above).fillna(False).astype(float)


def sig_vol_regime_switch(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """VOLATILITY-REGIME ALLOCATOR. Graded exposure = 1 - (realized-vol
    percentile): nearly full in the calmest decile, scaled down through the
    bands, ~flat in the stormiest - gated by a long trend so it never loads a
    calm downtrend. A continuous risk-state dial, decorrelated by construction."""
    p = params or {}
    vol_lb, ref, trend = p.get("vol_lb", 20), p.get("ref", 252), p.get("trend", 200)
    c = d["close"]
    rv = c.pct_change().rolling(vol_lb).std()
    pct = rv.rolling(ref, min_periods=ref // 2).rank(pct=True)      # 0 calm .. 1 stormy
    above = (c > c.rolling(trend, min_periods=trend // 2).mean()).astype(float)
    return ((1.0 - pct).clip(0, 1) * above).fillna(0.0)


def sig_gap_fade_revert(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """OVERNIGHT GAP-DOWN FADE. In an uptrend, when the OPEN gaps down more than
    k sigma below the prior close (panic open), fade it - buy the weakness and
    hold a few days for the intraday/next-day snapback. Uses the open vs prior
    close, a different information event than any close-to-close sleeve."""
    p = params or {}
    gap_k, trend = p.get("gap_k", 1.1), p.get("trend", 150)
    hold, vol_lb = p.get("hold", 3), p.get("vol_lb", 20)
    o, c = d["open"], d["close"]
    gap = o / c.shift(1) - 1.0
    sig = c.pct_change().rolling(vol_lb).std()
    above = c > c.rolling(trend, min_periods=trend // 2).mean()
    fire = (gap < -gap_k * sig) & above
    return _hold(fire.fillna(False).to_numpy(), hold, c.index)


def sig_breadth_thrust_self(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """INTERNAL BREADTH THRUST. Treat one name's own recent days as a
    'breadth' panel: the fraction of the last n days that closed above a short
    MA. When that internal breadth SURGES from weak to strong (a thrust), the
    move has broad internal support - ride it. A self-referential thrust signal."""
    p = params or {}
    n, ma = p.get("n", 20), p.get("ma", 10)
    lo, hi, hold = p.get("lo", 0.30), p.get("hi", 0.75), p.get("hold", 15)
    c = d["close"]
    frac = (c > c.rolling(ma).mean()).astype(float).rolling(n).mean()
    thrust = (frac > hi) & (frac.shift(n // 2) < lo)
    return _hold(thrust.fillna(False).to_numpy(), hold, c.index)


def sig_mean_gravity(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """ATR-STRETCH GRAVITY. Measure how far price is stretched BELOW a long
    anchor mean in ATR units; when the stretch exceeds g ATRs in an uptrend,
    'gravity' pulls it back - enter, exit on the return to the anchor. Distance
    measured in volatility units, not percent or sigma - its own metric."""
    p = params or {}
    anchor, g = p.get("anchor", 100), p.get("g", 2.5)
    atr, trend = p.get("atr", 20), p.get("trend", 200)
    c = d["close"]
    anc = c.rolling(anchor, min_periods=anchor // 2).mean()
    stretch = (anc - c) / _atr(d, atr)
    above = c > c.rolling(trend, min_periods=trend // 2).mean()
    enter = (stretch > g) & above
    exit_ = c >= anc
    return _state_machine(enter.fillna(False), exit_.fillna(False), c.index)


def sig_streak_reversal(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """DOWN-STREAK OVERREACTION. After k CONSECUTIVE down-closes (a rare run) in
    a secular uptrend, short-term selling has overshot - go long for `hold` days.
    Pure path-counting, no oscillator: counts the actual losing streak length."""
    p = params or {}
    k, trend, hold = p.get("k", 4), p.get("trend", 200), p.get("hold", 5)
    c = d["close"]
    down = c.diff() < 0
    streak = down.groupby((~down).cumsum()).cumsum()      # consecutive down count
    above = c > c.rolling(trend, min_periods=trend // 2).mean()
    fire = (streak >= k) & above
    return _hold(fire.fillna(False).to_numpy(), hold, c.index)


def sig_expansion_breakout(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """RANGE-EXPANSION IGNITION. The mirror of the coil: when today's true range
    blows out past m ATRs AND the close finishes in the top third of the day on
    an up-bar above trend, a move has IGNITED - ride the expansion for `hold`
    days. Trades the volatility breakout, not a price level."""
    p = params or {}
    m, atr = p.get("m", 1.8), p.get("atr", 20)
    trend, hold = p.get("trend", 100), p.get("hold", 8)
    h, l, c = d["high"], d["low"], d["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    span = (h - l).replace(0, np.nan)
    pos_in_day = (c - l) / span
    above = c > c.rolling(trend, min_periods=trend // 2).mean()
    fire = (tr > m * _atr(d, atr)) & (pos_in_day > 0.66) & (c > pc) & above
    return _hold(fire.fillna(False).to_numpy(), hold, c.index)


def sig_slope_quality(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """TREND R-SQUARED. Fit a line to log-price over the window; exposure is
    GRADED by the fit's R^2 when the slope is up (0 below a quality floor, ramps
    to 1 at a perfect line). Buys clean, statistically straight up-trends and
    refuses ragged ones - a trend-QUALITY dial, not a trend-direction switch."""
    p = params or {}
    win, r2_min = p.get("win", 40), p.get("r2_min", 0.55)
    lc = np.log(d["close"])
    x = np.arange(win, dtype=float)
    xc = x - x.mean()
    xden = (xc ** 2).sum()

    def _fit(y):
        if np.isnan(y).any():
            return 0.0
        slope = (xc * (y - y.mean())).sum() / xden
        yhat = y.mean() + slope * xc
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return r2 if slope > 0 else 0.0

    q = lc.rolling(win).apply(_fit, raw=True)
    return ((q - r2_min) / (1.0 - r2_min)).clip(0, 1).fillna(0.0)


def sig_dual_horizon_agree(d: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """DUAL-HORIZON AGREEMENT. Two horizons must agree: the SLOW horizon must be
    in an uptrend AND the FAST horizon must have just turned up out of a dip
    (recent fast return was negative, today closed up). 'Buy dips, but only in
    confirmed uptrends, and only once they actually turn.' Two-clock confirmation."""
    p = params or {}
    slow, fast_lb = p.get("slow", 200), p.get("fast_lb", 10)
    dip, hold = p.get("dip", -0.04), p.get("hold", 10)
    c = d["close"]
    slow_up = c > c.rolling(slow, min_periods=slow // 2).mean()
    fast_ret = c / c.shift(fast_lb) - 1.0
    turn = (fast_ret.shift(1) < dip) & (c.diff() > 0)
    return _hold((slow_up & turn).fillna(False).to_numpy(), hold, c.index)


# ===========================================================================
# registry: 12 strategies, their tuned (distinct) params, and the agent roster
# ===========================================================================

LAB_STRATEGIES = {
    "coil_release":       sig_coil_release,
    "drawdown_ladder":    sig_drawdown_ladder,
    "velocity_flip":      sig_velocity_flip,
    "trend_persistence":  sig_trend_persistence,
    "vol_regime_switch":  sig_vol_regime_switch,
    "gap_fade_revert":    sig_gap_fade_revert,
    "breadth_thrust_self": sig_breadth_thrust_self,
    "mean_gravity":       sig_mean_gravity,
    "streak_reversal":    sig_streak_reversal,
    "expansion_breakout": sig_expansion_breakout,
    "slope_quality":      sig_slope_quality,
    "dual_horizon_agree": sig_dual_horizon_agree,
}

# distinct parameter set per strategy (the "different parameters" requirement) -
# each is hand-set to its own horizon so no two share a recipe.
LAB_PARAMS = {
    "coil_release":       {"coil": 8,  "ref": 60,  "tight_pct": 0.30, "hold": 10, "trend": 100},
    "drawdown_ladder":    {"rungs": [0.04, 0.08, 0.13], "high_lb": 100, "trend": 250},
    "velocity_flip":      {"smooth": 10, "slope": 5, "trend": 100, "hold": 12},
    "trend_persistence":  {"win": 20, "persist_thr": 0.62, "trend": 120},
    "vol_regime_switch":  {"vol_lb": 20, "ref": 252, "trend": 200},
    "gap_fade_revert":    {"gap_k": 1.1, "trend": 150, "hold": 3, "vol_lb": 20},
    "breadth_thrust_self": {"n": 20, "ma": 10, "lo": 0.30, "hi": 0.75, "hold": 15},
    "mean_gravity":       {"anchor": 100, "g": 2.5, "atr": 20, "trend": 200},
    "streak_reversal":    {"k": 4, "trend": 200, "hold": 5},
    "expansion_breakout": {"m": 1.8, "atr": 20, "trend": 100, "hold": 8},
    "slope_quality":      {"win": 40, "r2_min": 0.55},
    "dual_horizon_agree": {"slow": 200, "fast_lb": 10, "dip": -0.04, "hold": 10},
}

# the 12 agents. Each owns one mechanism, carries the research thesis it explores,
# and belongs to a family (drives grouping + the constellation colours).
LAB_AGENTS = [
    {"agent": "coil-scout",     "strategy": "coil_release",       "family": "volatility",
     "thesis": "compressed ranges store energy that releases upward - trade the transition"},
    {"agent": "ladder-keeper",  "strategy": "drawdown_ladder",    "family": "reversion",
     "thesis": "accumulate deeper-into-the-dip in a secular uptrend, scale out on recovery"},
    {"agent": "inflection",     "strategy": "velocity_flip",      "family": "trend",
     "thesis": "price acceleration turns before the trend cross - buy the inflection"},
    {"agent": "pathwise",       "strategy": "trend_persistence",  "family": "trend",
     "thesis": "the smoothness of a climb, not its slope, predicts continuation"},
    {"agent": "regime-dial",    "strategy": "vol_regime_switch",  "family": "volatility",
     "thesis": "size inversely to the volatility percentile - full when calm, flat when stormy"},
    {"agent": "nightfade",      "strategy": "gap_fade_revert",    "family": "reversion",
     "thesis": "panic gap-down opens in an uptrend overshoot and snap back"},
    {"agent": "breadth-int",    "strategy": "breadth_thrust_self", "family": "trend",
     "thesis": "a name's own internal breadth thrust signals broad-based ignition"},
    {"agent": "gravity",        "strategy": "mean_gravity",       "family": "reversion",
     "thesis": "stretch below a long anchor in ATR units mean-reverts to the anchor"},
    {"agent": "streakwatch",    "strategy": "streak_reversal",    "family": "reversion",
     "thesis": "rare consecutive down-streaks in an uptrend are short-term overreactions"},
    {"agent": "ignition",       "strategy": "expansion_breakout", "family": "volatility",
     "thesis": "a true-range blow-out closing strong ignites a multi-day move"},
    {"agent": "straightline",   "strategy": "slope_quality",      "family": "trend",
     "thesis": "grade exposure by the R-squared of the up-trend - buy clean, refuse ragged"},
    {"agent": "two-clocks",     "strategy": "dual_horizon_agree", "family": "structure",
     "thesis": "act only when a slow uptrend and a fast turn-up agree"},
]

FAMILY_COLOR = {"reversion": "cyan", "trend": "up", "volatility": "bench", "structure": "violet"}
