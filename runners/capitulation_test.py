"""
runners/capitulation_test.py
----------------------------
Backtest the capitulation sleeve, focused on whether it lifts the weak 2018-2020
stretch (the two V-shaped crashes the trend-gated book missed). Compares it to
standard RSI-2, slices the Dec-2018 and COVID recoveries, and checks whether
adding it to the portfolio improves the 2018-2020 fold + passes the gate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_capitulation,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds, split_metrics,
)
from data.sp500 import sp500_tickers


def window(r, lo, hi):
    s = r[(r.index >= lo) & (r.index <= hi)]
    tot = (1 + s).prod() - 1
    sh = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
    return tot, sh, int((s != 0).sum())


def metrics(r, label):
    m = _metrics_from_returns(r, [], label)
    return m


def main():
    U = QUALITY_UNIVERSE
    print("\nbuilding sleeves ...")
    cap = backtest_book(sig_capitulation, U, label="cap")["_returns"]
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"], label="rsi")["_returns"]

    # 1) capitulation standalone
    m = metrics(cap, "capitulation")
    s = split_metrics(cap)
    print(f"\n== capitulation sleeve (quality-10, entry RSI(2)<5, >=7% below 10d high) ==")
    print(f"  full: Sharpe {m['sharpe']} | CAGR {m['cagr']:.1%} | maxDD {m['max_drawdown']:.1%} "
          f"| trades {m['total_trades']} | OOS Sharpe {s['test_sharpe']:+.2f}")
    print("  walk-forward folds:")
    for f in walk_forward_folds(cap, 5):
        print(f"    {f.get('start','?')[:7]}..{f.get('end','?')[:7]}: {f['return_pct']:+.1%} (SR {f['sharpe']:+.2f})")

    # 2) the 2018-2020 window + the two specific crash recoveries
    print("\n== does it fix 2018-2020? (capitulation vs standard RSI-2) ==")
    for lo, hi, lbl in [("2018-01-01", "2020-06-30", "2018-2020 full"),
                        ("2018-12-01", "2019-04-30", "Dec-2018 crash+recovery"),
                        ("2020-02-15", "2020-06-30", "COVID crash+recovery")]:
        ct, csh, cn = window(cap, lo, hi)
        rt, rsh, rn = window(rsi, lo, hi)
        print(f"  {lbl:26s}  capitulation {ct:+6.1%} ({cn} days active)  |  std RSI-2 {rt:+6.1%}")

    # 3) does adding it to the portfolio improve the 2018-2020 fold + pass the gate?
    print("\n== portfolio impact (risk-parity, 4 sleeves vs 5 with capitulation) ==")
    don = backtest_book(sig_donchian, U)["_returns"]
    trd = backtest_book(sig_trend_5020, U)["_returns"]
    xs  = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]

    def build(sleeves):
        panel = pd.concat(sleeves, axis=1, sort=True)
        iv = {c: 1.0 / (panel[c].std() or 1e-9) for c in panel.columns}
        w = np.array([iv[c] for c in panel.columns]); w /= w.sum()
        comb = (panel.fillna(0.0) * w).sum(axis=1)
        return vol_target(comb, 0.16, max_leverage=1.6)

    base = build({"rsi": rsi, "don": don, "trd": trd, "xs": xs})
    plus = build({"rsi": rsi, "don": don, "trd": trd, "xs": xs, "cap": cap})
    for label, r in [("portfolio (4 sleeves)", base), ("portfolio + capitulation (5)", plus)]:
        m = metrics(r, label)
        f2 = [f for f in walk_forward_folds(r, 5)][1]      # the 2018-2020 fold
        print(f"  {label:30s} Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1%} | "
              f"DD {m['max_drawdown']:.1%} | 2018-2020 fold {f2['return_pct']:+.1%}")


if __name__ == "__main__":
    main()
