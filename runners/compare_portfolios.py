"""
runners/compare_portfolios.py
-----------------------------
Side-by-side of the two deployed accounts + their combination:
  Account 1 (Growth)        : portfolio_full + 5% crypto (long equity)
  Account 2 (Crisis-alpha)  : managed_futures (long/short trend, conviction-scaled)
Shows PnL / CAGR / Sharpe / DD, the correlation between them, and year-by-year so
the diversification (one wins when the other loses) is visible. $100k each.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, daily_bars
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor, crypto_trend
from runners.lowvol_defensive import make_defensive
from runners.managed_futures import mf_returns


def row(label, r):
    m = _metrics_from_returns(r, [], label)
    pos = sum(1 for f in walk_forward_folds(r, 5) if f["sharpe"] > 0)
    print(f"  {label:26s} ${m['pnl_dollars']:>10,.0f} | {m['cagr']:6.1%} | {m['sharpe']:5.2f} | "
          f"{m['max_drawdown']:6.1%} | {pos}/5")
    return m


def main():
    print("building both accounts (~1-2 min) ...\n")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    cr = crypto_trend().reindex(idx).fillna(0)

    acct1 = overlays(combo * 0.95 + cr * 0.05, idx).fillna(0)        # Growth
    acct2 = mf_returns(0.12, 1.5)[0].reindex(idx).fillna(0)          # Crisis-alpha (managed futures)
    c70 = 0.70 * acct1 + 0.30 * acct2
    c50 = 0.50 * acct1 + 0.50 * acct2
    spy = daily_bars("SPY")["close"].reindex(idx).pct_change().fillna(0)

    print(f"  {'portfolio':26s} {'$ PnL':>11s} | {'CAGR':>6s} | {'Shrp':>5s} | {'maxDD':>6s} | WF")
    print("  " + "-" * 70)
    m1 = row("Account 1 (Growth)", acct1)
    m2 = row("Account 2 (Crisis-alpha)", acct2)
    row("Combined 70/30", c70)
    row("Combined 50/50", c50)
    row("S&P 500 (benchmark)", spy)

    corr = float(acct1.corr(acct2))
    print(f"\n  correlation Account 1 <-> Account 2: {corr:+.2f}  (low/negative = real diversification)")

    print("\n  YEAR-BY-YEAR (the point: when one loses, the other tends to win):")
    print(f"    {'year':>4s} {'Acct1 Growth':>13s} {'Acct2 Crisis':>13s} {'Combined70/30':>14s} {'S&P':>7s}")
    y1 = (1 + acct1).groupby(acct1.index.year).prod() - 1
    y2 = (1 + acct2).groupby(acct2.index.year).prod() - 1
    yc = (1 + c70).groupby(c70.index.year).prod() - 1
    ys = (1 + spy).groupby(spy.index.year).prod() - 1
    for y in y1.index:
        flag = "  <-- market down" if ys[y] < 0 else ""
        print(f"    {y:>4d} {y1[y]:+13.1%} {y2[y]:+13.1%} {yc[y]:+14.1%} {ys[y]:+7.1%}{flag}")


if __name__ == "__main__":
    main()
