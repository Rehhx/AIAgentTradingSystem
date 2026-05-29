"""
runners/antiwhipsaw_test.py
---------------------------
Does an anti-whipsaw band on the trend + market filters lift 2018-2020 (the
V-shaped-crash drag) without hurting the rest? Tests band widths on the trend
sleeve and the cross-sectional market filter, standalone and in the portfolio.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_trend_band,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds,
)
from data.sp500 import sp500_tickers

U = QUALITY_UNIVERSE


def fold2(r):  # the 2018-2020 fold
    return walk_forward_folds(r, 5)[1]["return_pct"]


def row(label, r, trades=None):
    m = _metrics_from_returns(r, [], label)
    t = f" | trades {trades}" if trades is not None else ""
    print(f"  {label:28s} Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | "
          f"DD {m['max_drawdown']:6.1%} | 2018-2020 {fold2(r):+5.1%}{t}")
    return r


def main():
    print("\n== TREND sleeve: bare 50/200 cross vs anti-whipsaw band ==")
    b0 = backtest_book(sig_trend_5020, U, label="t")
    row("trend (no band)", b0["_returns"], b0["total_trades"])
    for band in (0.02, 0.03, 0.05):
        b = backtest_book(sig_trend_band, U, {"band": band}, label="tb")
        row(f"trend band={band:.0%}", b["_returns"], b["total_trades"])

    print("\n== CROSS-SECTIONAL market filter: hard vs banded hysteresis ==")
    sp = sp500_tickers()
    x0 = backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10, market_filter=True)
    row("xs filter (hard)", x0["_returns"])
    for band in (0.03, 0.05):
        x = backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10,
                                     market_filter=True, market_band=band)
        row(f"xs filter band={band:.0%}", x["_returns"])

    print("\n== PORTFOLIO (risk-parity, vol-target 16%/1.6x): original vs anti-whipsaw ==")
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    don = backtest_book(sig_donchian, U)["_returns"]

    def build(trd, xs):
        panel = pd.concat({"rsi": rsi, "don": don, "trd": trd, "xs": xs}, axis=1, sort=True)
        iv = {c: 1.0 / (panel[c].std() or 1e-9) for c in panel.columns}
        w = np.array([iv[c] for c in panel.columns]); w /= w.sum()
        return vol_target((panel.fillna(0.0) * w).sum(axis=1), 0.16, max_leverage=1.6)

    orig = build(b0["_returns"], x0["_returns"])
    band_trd = backtest_book(sig_trend_band, U, {"band": 0.03}, label="tb")["_returns"]
    band_xs = backtest_cross_sectional(sp, mode="momentum", lookback=252, skip=21, k=10,
                                       market_filter=True, market_band=0.03)["_returns"]
    banded = build(band_trd, band_xs)
    row("portfolio ORIGINAL", orig)
    row("portfolio ANTI-WHIPSAW", banded)


if __name__ == "__main__":
    main()
