"""
runners/deployed_backtest.py
----------------------------
Full backtest of the LIVE deployed config (portfolio_full 7 sleeves + 5% crypto
sleeve, vol-target 17%/1.8x, early-warning, lowvol->BIL): headline PnL/Sharpe/DD,
year-by-year ROI vs S&P 500, and 5-fold walk-forward. Also prints the no-crypto
baseline for honesty (crypto's contribution is front-loaded in the 2017 bull).
$100k base, 6 bps round-trip, split/dividend-adjusted data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, split_metrics, daily_bars, INITIAL_CAP
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor, crypto_trend
from runners.lowvol_defensive import make_defensive


def headline(label, r):
    m = _metrics_from_returns(r, [], label)
    s = split_metrics(r)
    pos = sum(1 for f in walk_forward_folds(r, 5) if f["sharpe"] > 0)
    print(f"\n=== {label} ===")
    print(f"  $100,000 -> ${m['final_capital']:,.0f}  (+${m['pnl_dollars']:,.0f}, +{m['total_return']*100:.0f}%)")
    print(f"  CAGR {m['cagr']:.1%} | Sharpe {m['sharpe']} | max DD {m['max_drawdown']:.1%} "
          f"| in-sample {s['train_sharpe']:+.2f} -> OOS {s['test_sharpe']:+.2f} | WF {pos}/5 folds positive")
    return m


def main():
    print("building the deployed crypto-armed book (~1-2 min) ...")
    panel = build_base()
    idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    cr = crypto_trend().reindex(idx).fillna(0)

    book = overlays(combo * 0.95 + cr * 0.05, idx)          # DEPLOYED (crypto-armed)
    nocrypto = overlays(combo, idx)                          # baseline for honesty
    spy = daily_bars("SPY")["close"].reindex(idx).pct_change().fillna(0)

    headline("DEPLOYED: portfolio_full + 5% crypto", book)
    headline("baseline: portfolio_full (no crypto)", nocrypto)

    # year-by-year ROI
    yb = (1 + book).groupby(book.index.year).prod() - 1
    yn = (1 + nocrypto).groupby(nocrypto.index.year).prod() - 1
    ys = (1 + spy).groupby(spy.index.year).prod() - 1
    eq = INITIAL_CAP
    print("\n=== YEAR-BY-YEAR RETURN ON INVESTMENT ===")
    print(f"  {'year':>4s} {'deployed':>9s} {'no-crypto':>10s} {'S&P 500':>9s} {'$ (deployed)':>14s}")
    for y in yb.index:
        eq *= (1 + yb[y])
        star = " (partial)" if y in (yb.index[0], yb.index[-1]) else ""
        print(f"  {y:>4d} {yb[y]:+9.1%} {yn[y]:+10.1%} {ys[y]:+9.1%} {eq:>14,.0f}{star}")

    print("\n  walk-forward folds (deployed):")
    for f in walk_forward_folds(book, 5):
        print(f"    {f.get('start','?')[:7]}..{f.get('end','?')[:7]}: return {f['return_pct']:+6.1%} | Sharpe {f['sharpe']:+.2f}")

    print("\n  NOTE: crypto's lift is front-loaded in the 2017 BTC bull (won't repeat at that")
    print("  scale); the no-crypto column is the conservative base case. Crypto only deploys")
    print("  when BTC's 6-month trend is positive — it sits in cash otherwise.")


if __name__ == "__main__":
    main()
