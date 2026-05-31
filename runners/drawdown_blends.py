"""
runners/drawdown_blends.py
--------------------------
How low can we push the drawdown -- through a real GFC? Blends the core equity
engine (2005-2026) with the managed-futures crisis-alpha engine at several ratios
and reports CAGR / Sharpe / max drawdown / the 2008 GFC + 2022 returns. The honest
lever for lower crisis drawdowns is MORE managed futures (it's +ve in crashes), at
the cost of return. This shows the whole tradeoff curve across the worst bears.
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
from runners.extended_backtest import core_engine

MF = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "VNQ"]


def mf_long():
    def f(t):
        s = yf.Ticker(t).history(start="2006-06-01", end="2026-06-01", auto_adjust=True)["Close"]
        s.index = s.index.tz_localize(None) if s.index.tz is None else s.index.tz_convert("UTC").tz_localize(None)
        return s
    C = pd.DataFrame({t: f(t) for t in MF}).sort_index(); R = C.pct_change()
    sig = (np.sign(C / C.shift(21) - 1) + np.sign(C / C.shift(63) - 1) + np.sign(C / C.shift(252) - 1)) / 3
    vol = R.rolling(60).std(); Wg = (sig / vol).div((sig / vol).abs().sum(axis=1).replace(0, np.nan), axis=0)
    core = (Wg.shift(1) * R).sum(axis=1) - Wg.diff().abs().sum(axis=1).fillna(0) * 0.0006
    conv = sig.abs().mean(axis=1).clip(0, 1).shift(1).fillna(0)
    rv = (core * conv).rolling(20).std() * np.sqrt(TRADING_DAYS)
    scale = (0.12 / rv.replace(0, np.nan)).clip(upper=1.5).shift(1).fillna(0)
    return (core * conv * scale).fillna(0)


def main():
    print("building core equity (2005-) + managed futures (2006-) through the GFC (~2 min) ...")
    eq, spy = core_engine()
    eq.index = eq.index.tz_localize(None); spy.index = spy.index.tz_localize(None)
    mf = mf_long()
    idx = eq.index.intersection(mf.index)
    eq, mf, sret = eq.reindex(idx).fillna(0), mf.reindex(idx).fillna(0), spy.reindex(idx).fillna(0)
    print(f"  common period {idx[0].date()}..{idx[-1].date()}\n")

    print(f"  {'growth / crisis blend':24s} {'CAGR':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'2008 GFC':>9s} {'2022':>7s}")
    print("  " + "-" * 68)
    for w in (1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0):
        b = w * eq + (1 - w) * mf
        m = _metrics_from_returns(b, [], "x")
        gfc = (1 + b.loc["2007-10-01":"2009-03-09"]).prod() - 1
        y = (1 + b).groupby(b.index.year).prod() - 1
        lbl = f"{int(w*100)}/{int((1-w)*100)} growth/crisis"
        print(f"  {lbl:24s} {m['cagr']:6.1%} {m['sharpe']:7.2f} {m['max_drawdown']:7.1%} "
              f"{gfc:+9.1%} {y.get(2022,0):+7.1%}")
    sm = _metrics_from_returns(sret, [], "spy")
    print(f"  {'S&P 500 (reference)':24s} {sm['cagr']:6.1%} {sm['sharpe']:7.2f} {sm['max_drawdown']:7.1%} "
          f"{(1+sret.loc['2007-10-01':'2009-03-09']).prod()-1:+9.1%}")
    print("\n  Read: more crisis-alpha => lower drawdown (esp. 2008) but lower CAGR. Pick the point")
    print("  on the curve that matches the board's risk tolerance; ~50-60% growth roughly halves")
    print("  the GFC drawdown vs growth-only while keeping a solid return.")


if __name__ == "__main__":
    main()
