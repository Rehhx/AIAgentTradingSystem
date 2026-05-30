"""
runners/crisis_alpha_test.py
----------------------------
Can we counter the negative years (2018, 2022)? The only thing that PROFITS in a
sustained bear is a strategy that can go SHORT: a long/short trend-following
(managed-futures) sleeve across liquid ETFs. It goes long uptrending assets and
short downtrending ones, so in 2022 (stocks+bonds down, dollar up) it makes money.

Tests the L/S trend sleeve standalone (year-by-year, esp. 2018/2022) and blended
into the deployed book at a few weights -- does it shrink/eliminate the down years,
and at what cost to the bull years? $100k, 6 bps, adjusted data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import daily_bars, RT_COST, _metrics_from_returns, walk_forward_folds, TRADING_DAYS
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor, crypto_trend
from runners.lowvol_defensive import make_defensive

ASSETS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC", "UUP", "EEM"]   # equities, bonds, gold, commodities, USD, EM
LOOKBACK = 63        # 3-month time-series momentum


def ls_trend():
    """long/short time-series momentum, inverse-vol weighted, gross exposure ~1."""
    closes = {}
    for t in ASSETS:
        try:
            c = daily_bars(t)["close"]
            if len(c) > 300:
                closes[t] = c
        except Exception:
            pass
    C = pd.DataFrame(closes).sort_index()
    R = C.pct_change()
    sig = np.sign(C / C.shift(LOOKBACK) - 1)               # +1 uptrend, -1 downtrend
    vol = R.rolling(60).std()
    raw = sig / vol                                        # inverse-vol risk weighting
    w = raw.div(raw.abs().sum(axis=1), axis=0)             # gross exposure ~1
    port = (w.shift(1) * R).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    return (port - turn * RT_COST).fillna(0)


def yearly(s):
    return (1 + s).groupby(s.index.year).prod() - 1


def main():
    print("building deployed book + long/short trend (managed-futures) sleeve ...\n")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    cr = crypto_trend().reindex(idx).fillna(0)
    book = overlays(combo * 0.95 + cr * 0.05, idx).fillna(0)       # deployed (crypto-armed)
    ls = ls_trend().reindex(idx).fillna(0)

    m = _metrics_from_returns(ls, [], "ls")
    pos = sum(1 for f in walk_forward_folds(ls, 5) if f["sharpe"] > 0)
    print(f"L/S trend sleeve standalone: Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1%} | DD {m['max_drawdown']:.1%} | WF {pos}/5")
    corr = float(book.corr(ls))
    print(f"  correlation to the deployed book: {corr:+.2f}  (negative/low = good crisis hedge)\n")

    yb = yearly(book); yl = yearly(ls)
    print("YEAR-BY-YEAR: deployed book vs + L/S trend at 15% / 25% (focus on the down years):")
    print(f"  {'year':>4s} {'deployed':>9s} {'L/S only':>9s} {'+15% L/S':>9s} {'+25% L/S':>9s}")
    for w15, w25 in [(0.15, 0.25)]:
        pass
    for y in yb.index:
        b, l = yb[y], yl.get(y, 0.0)
        c15 = 0.85 * b + 0.15 * l
        c25 = 0.75 * b + 0.25 * l
        mark = "  <-- was negative" if b < 0 else ""
        print(f"  {y:>4d} {b:+9.1%} {l:+9.1%} {c15:+9.1%} {c25:+9.1%}{mark}")

    for w in (0.15, 0.25):
        cb = (1 - w) * book + w * ls
        mc = _metrics_from_returns(cb, [], "x")
        negs = (yearly(cb) < 0).sum()
        print(f"\n  book + {w:.0%} L/S: Sharpe {mc['sharpe']:.2f} | CAGR {mc['cagr']:.1%} | DD {mc['max_drawdown']:.1%} "
              f"| negative years: {negs}")


if __name__ == "__main__":
    main()
