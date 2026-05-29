"""
runners/trend_multi_test.py
---------------------------
Does a multi-speed trend (faster recovery participation) fix 2018-2020 WITHOUT
overfitting? Robustness check: it must help 2018-2020 AND not hurt the other
folds. Tests standalone vs single 50/200, then swapped into the portfolio.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_trend_multi, sig_pead,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds, daily_bars,
)
from data.sp500 import sp500_tickers


def folds(r):
    return [f["return_pct"] for f in walk_forward_folds(r, 5)]


def show(label, r):
    m = _metrics_from_returns(r, [], label)
    fl = folds(r)
    print(f"  {label:24s} SR {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | DD {m['max_drawdown']:6.1%}"
          f" | folds " + " ".join(f"{x:+.0%}" for x in fl))
    return r


def main():
    U = QUALITY_UNIVERSE
    print("\n== TREND sleeve: single 50/200 vs multi-speed (folds: 16-18 18-20 20-22 22-24 24-26) ==")
    single = backtest_book(sig_trend_5020, U, label="t")["_returns"]
    show("single 50/200", single)
    multi = backtest_book(sig_trend_multi, U, label="tm")["_returns"]
    show("multi-speed", multi)
    # robustness: faster-only variant for comparison
    show("multi (fast-tilt)", backtest_book(sig_trend_multi, U,
         {"speeds": [(20, 60), (20, 100), (50, 200)]}, label="tmf")["_returns"])

    print("\n== PORTFOLIO: swap trend_5020 -> multi-speed (risk-parity, vol-target 16%/1.6x) ==")
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    don = backtest_book(sig_donchian, U)["_returns"]
    xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]

    def build(trd):
        panel = pd.concat({"rsi": rsi, "don": don, "trd": trd, "xs": xs}, axis=1, sort=True)
        iv = {c: 1.0 / (panel[c].std() or 1e-9) for c in panel.columns}
        w = np.array([iv[c] for c in panel.columns]); w /= w.sum()
        return vol_target((panel.fillna(0.0) * w).sum(axis=1), 0.16, max_leverage=1.6)

    show("portfolio (single trend)", build(single))
    show("portfolio (multi trend)", build(multi))


if __name__ == "__main__":
    main()
