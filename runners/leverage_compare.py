"""
runners/leverage_compare.py
---------------------------
Honest cost of the NO-MARGIN decision. Backtests the EXACT deployed portfolio_full
book (6 price sleeves at W + 10% defensive low-vol, with the vol-target + early-
warning + cash overlays) at several leverage caps, so we can see precisely what we
give up by capping at 1.0x (margin-call-proof) vs the old 1.8x.

maxlev = 1.0 means vol-target can only DE-RISK (never borrow) -> gross <= 100% ->
the book can never be margin-called. Everything above 1.0x uses margin in calm
markets. This is the same book the live rebalancer trades (no-crypto base).

$100k base, 6 bps round-trip, 2016-2026 deployed window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import _metrics_from_returns
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor
from runners.lowvol_defensive import make_defensive

LEVS = (1.0, 1.2, 1.5, 1.8)
BEARS = {"2018 Q4": ("2018-10-01", "2018-12-24"), "COVID": ("2020-02-19", "2020-03-23"),
         "2022 bear": ("2022-01-01", "2022-10-12")}


def main():
    print("building the deployed portfolio_full book (6 sleeves + 10% low-vol) ~1-2 min ...\n")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10

    books = {L: overlays(combo, idx, vt=0.17, maxlev=L).fillna(0) for L in LEVS}

    print("=" * 76)
    print("DEPLOYED BOOK vs LEVERAGE CAP  (2016-2026, $100k, 6bps)")
    print("=" * 76)
    print(f"  {'leverage cap':18s} {'$100k ->':>12s}  {'CAGR':>6s}  {'Sharpe':>6s}  {'maxDD':>7s}  {'margin?':>8s}")
    print("  " + "-" * 66)
    for L in LEVS:
        m = _metrics_from_returns(books[L], [], f"L{L}")
        margin = "NONE" if L <= 1.0 else f"up to {L:.1f}x"
        tag = "  <- LIVE (no-margin)" if L == 1.0 else ("  <- old config" if L == 1.8 else "")
        print(f"  {'<= ' + f'{L:.1f}x':18s} ${m['final_capital']:>11,.0f}  {m['cagr']:>6.1%}  "
              f"{m['sharpe']:>6.2f}  {m['max_drawdown']:>7.1%}  {margin:>8s}{tag}")

    print("\n  YEAR-BY-YEAR (no-margin 1.0x vs old 1.8x):")
    y10 = (1 + books[1.0]).groupby(books[1.0].index.year).prod() - 1
    y18 = (1 + books[1.8]).groupby(books[1.8].index.year).prod() - 1
    print(f"    {'year':6s} {'1.0x':>8s} {'1.8x':>8s}")
    for y in y10.index:
        print(f"    {y:<6d} {y10[y]:>+8.1%} {y18.get(y, 0):>+8.1%}")

    print("\n  BEAR-WINDOW BEHAVIOR (total return through each stress; lower leverage = smaller loss):")
    print(f"    {'window':14s} {'1.0x':>8s} {'1.8x':>8s}")
    for nm, (a, b) in BEARS.items():
        def tot(bk):
            w = (1 + bk.loc[a:b]).prod() - 1
            return w
        print(f"    {nm:14s} {tot(books[1.0]):>+8.1%} {tot(books[1.8]):>+8.1%}")

    m10 = _metrics_from_returns(books[1.0], [], "x")
    m18 = _metrics_from_returns(books[1.8], [], "x")
    print("\n" + "=" * 76)
    print("THE HONEST TRADE")
    print("=" * 76)
    print(f"""  Going from 1.8x -> 1.0x (no-margin) on the deployed window:
    CAGR   {m18['cagr']:.1%}  ->  {m10['cagr']:.1%}   ({(m10['cagr']-m18['cagr'])*100:+.1f} pts)
    Sharpe {m18['sharpe']:.2f}  ->  {m10['sharpe']:.2f}
    maxDD  {m18['max_drawdown']:.1%}  ->  {m10['max_drawdown']:.1%}

  You give up the calm-market leverage boost, but the book now CANNOT be margin-
  called (gross <= 100% always) and the drawdown shrinks. Note: this is the benign
  2016-2026 window -- the honest worst case is still ~-32% (2008 GFC, extended_backtest.py),
  which 1.0x also reduces vs 1.8x. Trailing stops at 20% are catastrophe-only.""")


if __name__ == "__main__":
    main()
