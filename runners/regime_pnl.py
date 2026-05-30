"""
runners/regime_pnl.py
---------------------
Where the money is made/lost: PnL of the deployed Growth book (Account 1) broken
out by market regime, plus the headline backtest of all the best books. $100k base.
Regime PnL is attributed by log-growth share (so the parts reconcile to the total).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from agents.daily_strategies import _metrics_from_returns, walk_forward_folds, daily_bars, INITIAL_CAP, TRADING_DAYS
from runners.diversifier_screen import build_base, overlays, W
from runners.final_tests import lowvol_factor, crypto_trend
from runners.lowvol_defensive import make_defensive
from runners.managed_futures import mf_returns


def regimes(idx):
    spy = daily_bars("SPY")["close"].reindex(idx)
    sret = spy.pct_change().fillna(0)
    up = spy > spy.rolling(200).mean()
    calm = (sret.rolling(20).std() * np.sqrt(TRADING_DAYS)) < 0.20
    reg = pd.Series("", index=idx)
    reg[up & calm] = "Bull . calm"; reg[up & ~calm] = "Bull . stormy"
    reg[~up & calm] = "Bear . calm"; reg[~up & ~calm] = "Bear . stormy"
    return reg, sret


def main():
    print("building books (~1-2 min) ...\n")
    panel = build_base(); idx = panel.index
    lvd = make_defensive(lowvol_factor()).reindex(idx).fillna(0)
    combo = sum(panel[c].fillna(0) * W[c] for c in W) * 0.90 + lvd * 0.10
    cr = crypto_trend().reindex(idx).fillna(0)
    book = overlays(combo * 0.95 + cr * 0.05, idx).fillna(0)        # Account 1 (Growth)
    mf = mf_returns(0.12, 1.5)[0].reindex(idx).fillna(0)            # Account 2 (Crisis-alpha)
    c70 = 0.70 * book + 0.30 * mf
    reg, sret = regimes(idx)

    total_pnl = INITIAL_CAP * ((1 + book).prod() - 1)
    total_log = np.log1p(book).sum()
    order = ["Bull . calm", "Bull . stormy", "Bear . calm", "Bear . stormy"]
    print("=== PnL BY MARKET REGIME — Account 1 (Growth book), $100k base ===")
    print(f"  {'regime':16s} {'% days':>7s} {'book ret':>9s} {'SPY ret':>8s} {'$ PnL share':>12s}")
    print("  " + "-" * 58)
    for r in order:
        m = reg == r
        if m.sum() == 0:
            continue
        bret = (1 + book[m]).prod() - 1
        spyret = (1 + sret[m]).prod() - 1
        share = float(np.log1p(book[m]).sum() / total_log) if total_log != 0 else 0
        print(f"  {r:16s} {m.mean():7.0%} {bret:+9.1%} {spyret:+8.1%} {share*total_pnl:>+12,.0f}")
    print(f"  {'TOTAL':16s} {'100%':>7s} {(1+book).prod()-1:+9.1%} {(1+sret).prod()-1:+8.1%} {total_pnl:>+12,.0f}")

    print("\n=== BACKTEST OF THE BEST BOOKS (2016-2026, $100k base, 6bps) ===")
    print(f"  {'book':30s} {'$ PnL':>11s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'WF':>4s}")
    print("  " + "-" * 72)
    for label, r in [("Account 1 - Growth (eq+crypto)", book),
                     ("Account 2 - Crisis-alpha (CTA)", mf),
                     ("Combined 70/30 (all-weather)", c70),
                     ("S&P 500 benchmark", sret)]:
        mm = _metrics_from_returns(r, [], label)
        pos = sum(1 for f in walk_forward_folds(r, 5) if f["sharpe"] > 0)
        print(f"  {label:30s} {mm['pnl_dollars']:>+11,.0f} {mm['cagr']:>7.1%} "
              f"{mm['sharpe']:>7.2f} {mm['max_drawdown']:>7.1%} {pos}/5")


if __name__ == "__main__":
    main()
