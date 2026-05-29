"""
runners/long_short_test.py
--------------------------
Backtest the market-neutral long/short sleeve (PnL) and, crucially, measure its
CORRELATION + beta to the market and to our long-only portfolio — that low
correlation is the whole point. Then check whether adding it lifts 2018-2020 and
improves the portfolio's risk-adjusted return.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, backtest_long_short, vol_target,
    _metrics_from_returns, sig_rsi2_meanrev, sig_donchian, sig_trend_5020,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds,
)
from data.sp500 import sp500_tickers


def fold2(r):
    return walk_forward_folds(r, 5)[1]["return_pct"]


def main():
    sp = sp500_tickers()
    print("\n== market-neutral long/short sleeves (full S&P 500, dollar-neutral, $100k) ==")
    print(f"{'sleeve':22s} {'Sharpe':>6s} {'$PnL':>11s} {'CAGR':>7s} {'maxDD':>7s} "
          f"{'beta':>6s} {'corrSPY':>8s} {'2018-20':>8s}")
    print("-" * 80)
    ls = {}
    for lbl, kw in [("LS momentum 12-1", dict(mode="momentum", lookback=252, skip=21, k=50)),
                    ("LS reversal 5d",   dict(mode="reversal", lookback=5, k=50)),
                    ("LS reversal 3d",   dict(mode="reversal", lookback=3, k=50))]:
        m = backtest_long_short(sp, label=lbl, **kw)
        ls[lbl] = m
        print(f"{lbl:22s} {m['sharpe']:6.2f} {m['pnl_dollars']:11,.0f} {m['cagr']:7.1%} "
              f"{m['max_drawdown']:7.1%} {m.get('beta_to_spy',0):6.2f} "
              f"{m.get('corr_to_spy',0):8.2f} {fold2(m['_returns']):+8.1%}")

    best = max(ls, key=lambda k: ls[k]["sharpe"])
    lsr = ls[best]["_returns"]
    print(f"\nbest L/S: {best} (Sharpe {ls[best]['sharpe']}, corr to SPY {ls[best].get('corr_to_spy')})")

    # portfolio impact: add the L/S sleeve to the risk-parity book
    U = QUALITY_UNIVERSE
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    don = backtest_book(sig_donchian, U)["_returns"]
    trd = backtest_book(sig_trend_5020, U)["_returns"]
    xs = backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]

    def build(sleeves):
        panel = pd.concat(sleeves, axis=1, sort=True)
        iv = {c: 1.0 / (panel[c].std() or 1e-9) for c in panel.columns}
        w = np.array([iv[c] for c in panel.columns]); w /= w.sum()
        return vol_target((panel.fillna(0.0) * w).sum(axis=1), 0.16, max_leverage=1.6)

    base = build({"rsi": rsi, "don": don, "trd": trd, "xs": xs})
    plus = build({"rsi": rsi, "don": don, "trd": trd, "xs": xs, "ls": lsr})
    # corr of L/S to the long-only portfolio
    corr_port = round(float(lsr.reindex(base.index).fillna(0).corr(base)), 2)
    print(f"\ncorrelation of {best} to the long-only portfolio: {corr_port}")
    print("\n== portfolio impact (risk-parity, vol-target 16%/1.6x) ==")
    for label, r in [("portfolio (4 sleeves)", base), ("portfolio + L/S (5 sleeves)", plus)]:
        m = _metrics_from_returns(r, [], label)
        print(f"  {label:30s} Sharpe {m['sharpe']:.2f} | CAGR {m['cagr']:.1%} | "
              f"DD {m['max_drawdown']:.1%} | 2018-2020 {fold2(r):+.1%}")


if __name__ == "__main__":
    main()
