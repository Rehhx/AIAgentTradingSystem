"""
runners/pairs_statarb.py  (market-neutral research)
---------------------------------------------------
The whole book is LONG-biased — it has no engine whose return is independent of
market direction. This tests dollar-neutral statistical-arbitrage PAIRS trading on
economically-linked names: when the spread (log price ratio) diverges, short the
rich leg / long the cheap leg, bet on convergence, exit when it reverts.

A market-neutral sleeve earns a seat only if it (a) has LOW correlation to SPY,
(b) is positive in BOTH bull and bear years (regime-agnostic), and (c) clears a
useful Sharpe after costs. Needs shorting -> Account 2. 6 bps/leg, 2010-2026.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from agents.daily_strategies import _metrics_from_returns, vol_target, walk_forward_folds, TRADING_DAYS

PAIRS = [("KO", "PEP"), ("HD", "LOW"), ("V", "MA"), ("GS", "MS"), ("XOM", "CVX"),
         ("GLD", "GDX"), ("XLE", "XOP"), ("EWA", "EWC"), ("MCD", "YUM"), ("WMT", "TGT")]
LOOKBACK, ENTRY, EXIT, COST = 60, 2.0, 0.5, 0.0006


def _px(t):
    s = yf.Ticker(t).history(start="2009-06-01", end="2026-06-03", auto_adjust=True)["Close"]
    s.index = (s.index.tz_localize(None) if s.index.tz is None else s.index.tz_convert("UTC").tz_localize(None)).normalize()
    return s


def pair_returns(a, b):
    pa, pb = _px(a), _px(b)
    idx = pa.index.intersection(pb.index)
    pa, pb = pa.reindex(idx), pb.reindex(idx)
    if len(idx) < LOOKBACK + 60:
        return None
    spread = np.log(pa) - np.log(pb)
    z = (spread - spread.rolling(LOOKBACK).mean()) / spread.rolling(LOOKBACK).std()
    # position on the spread: +1 = long A / short B (when z<-ENTRY, spread cheap), -1 = opposite
    pos = pd.Series(0.0, index=idx)
    cur = 0.0
    for i in range(len(idx)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = cur; continue
        if cur == 0.0:
            if zi >= ENTRY:  cur = -1.0          # spread rich -> short A / long B
            elif zi <= -ENTRY: cur = 1.0         # spread cheap -> long A / short B
        elif abs(zi) <= EXIT:
            cur = 0.0                             # reverted -> flat
        pos.iloc[i] = cur
    ra, rb = pa.pct_change().fillna(0), pb.pct_change().fillna(0)
    # dollar-neutral: +0.5 long A / -0.5 short B  (scaled by pos sign)
    legA, legB = 0.5 * pos.shift(1).fillna(0), -0.5 * pos.shift(1).fillna(0)
    turn = (legA.diff().abs() + legB.diff().abs()).fillna(0)
    return (legA * ra + legB * rb - turn * COST).rename(f"{a}/{b}")


def main():
    print("pulling pairs + building dollar-neutral spreads (2010-2026) ...\n")
    series = []
    for a, b in PAIRS:
        try:
            r = pair_returns(a, b)
            if r is not None:
                series.append(r)
                print(f"  {a}/{b:5s} ok")
        except Exception as e:
            print(f"  {a}/{b}: skip ({e})")
    panel = pd.concat(series, axis=1, sort=True).fillna(0.0)
    book = panel.mean(axis=1)                      # equal-weight the pairs
    book_vt = vol_target(book, target_vol=0.10, max_leverage=2.0)   # market-neutral can run hotter

    spy = _px("SPY").pct_change().reindex(book.index).fillna(0)
    corr = float(book.corr(spy))
    m = _metrics_from_returns(book_vt, [], "pairs")
    folds = walk_forward_folds(book_vt, 5); pos = sum(1 for f in folds if f["sharpe"] > 0)

    print("\n" + "=" * 70)
    print("MARKET-NEUTRAL PAIRS STAT-ARB  (10 pairs, dollar-neutral, vol-target 10%)")
    print("=" * 70)
    print(f"  CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']:.2f} | maxDD {m['max_drawdown']:.1%} "
          f"| corr to SPY {corr:+.2f} | WF {pos}/5")

    print("\n  REGIME TEST (return by year; should be ~regime-agnostic):")
    yb = (1 + book_vt).groupby(book_vt.index.year).prod() - 1
    ys = (1 + spy).groupby(spy.index.year).prod() - 1
    for y in yb.index:
        tag = " <-bear" if ys.get(y, 0) < 0 else ""
        print(f"    {y}  pairs {yb[y]:>+6.1%}   SPY {ys.get(y,0):>+6.1%}{tag}")

    print("\n" + "=" * 70)
    verdict = ("ADD — low corr, regime-agnostic" if (abs(corr) < 0.3 and m["sharpe"] >= 0.7 and pos >= 4)
               else "WEAK/REJECT — pairs alpha may have decayed")
    print(f"  Verdict: {verdict}")
    print("  A market-neutral sleeve's value is corr ~0 to the book + positive in bear years.")
    print("  If Sharpe is low or it bleeds in recent years, pairs alpha has been arbitraged away.")


if __name__ == "__main__":
    main()
