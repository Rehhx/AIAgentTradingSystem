"""
runners/crash_sentinel.py
-------------------------
RESEARCH: a FAST crash-detection overlay the current book lacks. The deployed
early-warning (SPY < 50d AND 20d-vol > 20%) is a price/realized-vol signal that
LAGS a fast crash. The VIX term structure inverts at the *start* of acute stress:
when spot VIX rises above 3-month VIX (VIX/VIX3M >= 1, "backwardation"), the market
is pricing near-term panic. That's a documented, faster sentinel.

We test a regime-agnostic de-risk overlay: hold SPY in calm term structure, rotate
to T-bills when the sentinel fires. Compared honestly against:
  - SPY buy & hold
  - the deployed early-warning overlay
  - the two combined

Reports CAGR/Sharpe/maxDD, the drawdown through each FAST crash, whipsaw cost in
calm years, and a 5-fold walk-forward. $100k, 6 bps per signal flip, 2007-2026.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import _metrics_from_returns, TRADING_DAYS

FLIP_COST = 0.0006
CRASHES = {"2008 GFC": ("2008-09-01", "2008-11-20"), "2010 flash": ("2010-04-23", "2010-07-02"),
           "2011 EU": ("2011-07-22", "2011-10-03"), "2015 Aug": ("2015-08-10", "2015-08-25"),
           "2018 Feb (Volmageddon)": ("2018-01-26", "2018-02-09"), "2018 Q4": ("2018-10-01", "2018-12-24"),
           "COVID": ("2020-02-19", "2020-03-23"), "2022 bear": ("2022-01-03", "2022-10-12")}


def _series(t, start="2007-07-01"):
    s = yf.Ticker(t).history(start=start, end="2026-06-03", auto_adjust=True)["Close"]
    s.index = (s.index.tz_localize(None) if s.index.tz is None else s.index.tz_convert("UTC").tz_localize(None)).normalize()
    return s


def _overlay(spy_ret, cash_ret, position):
    """position in {0,1} held with 1-day lag; charge cost on each flip."""
    pos = position.shift(1).fillna(1.0)
    turn = pos.diff().abs().fillna(0.0)
    return pos * spy_ret + (1 - pos) * cash_ret - turn * FLIP_COST


def _report(name, ret, spy_ret):
    m = _metrics_from_returns(ret, [], name)
    # 5-fold walk-forward: positive folds
    fl = np.array_split(ret.values, 5)
    pos = sum(1 for f in fl if f.sum() > 0)
    return m, pos


def main():
    print("pulling SPY + VIX term structure (2007-2026) ...\n")
    spy = _series("SPY"); vix = _series("^VIX"); vix3m = _series("^VIX3M")
    irx = (_series("^IRX") / 100.0).clip(lower=0).fillna(0.02)
    idx = spy.index.intersection(vix.index).intersection(vix3m.index)
    spy, vix, vix3m, irx = [x.reindex(idx).ffill() for x in (spy, vix, vix3m, irx)]
    spy_ret = spy.pct_change().fillna(0.0)
    cash_ret = (1 + irx) ** (1 / TRADING_DAYS) - 1                       # daily T-bill yield

    ts = vix / vix3m                                                     # term structure ratio
    sentinel_on = (ts >= 1.0).astype(float)                             # backwardation = stress
    # deployed early-warning: SPY<50d AND 20d realized vol > 20%
    ew_on = ((spy < spy.rolling(50).mean()) &
             (spy_ret.rolling(20).std() * np.sqrt(TRADING_DAYS) > 0.20)).astype(float)

    strategies = {
        "SPY buy & hold":              pd.Series(1.0, index=idx),
        "Early-warning (deployed)":    1 - ew_on,
        "VIX-term-structure sentinel": 1 - sentinel_on,
        "Sentinel OR early-warning":   1 - np.maximum(sentinel_on, ew_on),
    }
    rets = {n: (spy_ret if n == "SPY buy & hold" else _overlay(spy_ret, cash_ret, p))
            for n, p in strategies.items()}

    print("=" * 84)
    print("CRASH SENTINEL vs BUY-HOLD vs DEPLOYED EARLY-WARNING  (2007-2026, $100k, 6bps/flip)")
    print("=" * 84)
    print(f"  {'strategy':28s} {'$100k->':>11s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'WF':>5s} {'days off':>9s}")
    print("  " + "-" * 80)
    for n in strategies:
        m, wf = _report(n, rets[n], spy_ret)
        off = (strategies[n] < 1).mean() if n != "SPY buy & hold" else 0.0
        print(f"  {n:28s} ${m['final_capital']:>10,.0f} {m['cagr']:>6.1%} {m['sharpe']:>7.2f} "
              f"{m['max_drawdown']:>7.1%} {wf}/5 {off:>8.0%}")

    print("\n  DRAWDOWN THROUGH EACH FAST CRASH (lower = better protection):")
    print(f"    {'window':24s} {'buy&hold':>9s} {'early-warn':>11s} {'sentinel':>9s} {'combined':>9s}")
    for nm, (a, b) in CRASHES.items():
        def dd(r):
            w = (1 + r.loc[a:b]).cumprod()
            return (w / w.cummax() - 1).min() if len(w) else 0.0
        print(f"    {nm:24s} {dd(rets['SPY buy & hold']):>9.1%} {dd(rets['Early-warning (deployed)']):>11.1%} "
              f"{dd(rets['VIX-term-structure sentinel']):>9.1%} {dd(rets['Sentinel OR early-warning']):>9.1%}")

    print("\n  CALM-YEAR WHIPSAW COST (sentinel return vs buy-hold; negative = sentinel lagged):")
    calm = [2013, 2014, 2017, 2019, 2021, 2024]
    ys = (1 + rets["SPY buy & hold"]).groupby(idx.year).prod() - 1
    yc = (1 + rets["VIX-term-structure sentinel"]).groupby(idx.year).prod() - 1
    print("   ", "  ".join(f"{y}:{(yc.get(y,0)-ys.get(y,0))*100:+.1f}pt" for y in calm))

    print("\n" + "=" * 84)
    print("READ")
    print("=" * 84)
    print("""  The sentinel earns its place ONLY if it cuts fast-crash drawdowns more than the
  deployed early-warning while not bleeding too much in calm years. If 'combined'
  beats 'early-warning' on crash drawdown at similar CAGR/Sharpe, it's additive --
  wire it as a second, faster trigger on the de-risk overlay. If it just whipsaws,
  reject it (like the 30 before). The numbers above decide, not the story.""")


if __name__ == "__main__":
    main()
