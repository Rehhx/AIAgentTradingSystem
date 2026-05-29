"""
runners/multiasset_trend_test.py
--------------------------------
Test multi-asset trend-following ("crisis alpha") as the fix for equity lean
years. Time-series momentum (long while trailing return > 0) across a diversified
ETF basket — equities, bonds, gold, commodities, the dollar, REITs. When equities
chop (2018-2020), bonds/gold trend up, so this book can earn while the equity
book is flat. Also models parking idle cash in T-bills (BIL).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_abs_momentum,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds, daily_bars,
)
from data.sp500 import sp500_tickers

# diversified, liquid, multi-asset-class ETFs (all have 2016+ history)
MULTI = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "VNQ"]


def fold2(r):
    return walk_forward_folds(r, 5)[1]["return_pct"]


def show(label, r, extra=""):
    m = _metrics_from_returns(r, [], label)
    print(f"  {label:30s} Sharpe {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | "
          f"DD {m['max_drawdown']:6.1%} | 2018-2020 {fold2(r):+6.1%}{extra}")
    return r


def main():
    print(f"\n== multi-asset trend (TS-momentum, long/flat) across {len(MULTI)} ETFs ==")
    mat252 = backtest_book(sig_abs_momentum, MULTI, {"lookback": 252})["_returns"]
    mat126 = backtest_book(sig_abs_momentum, MULTI, {"lookback": 126})["_returns"]
    show("multi-asset trend 12mo", mat252)
    show("multi-asset trend 6mo", mat126)
    mat = mat252

    # equity portfolio (the 4 sleeves) for correlation + blend
    U = QUALITY_UNIVERSE
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    don = backtest_book(sig_donchian, U)["_returns"]
    trd = backtest_book(sig_trend_5020, U)["_returns"]
    xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]

    def build(sleeves):
        panel = pd.concat(sleeves, axis=1, sort=True)
        iv = {c: 1.0 / (panel[c].std() or 1e-9) for c in panel.columns}
        w = np.array([iv[c] for c in panel.columns]); w /= w.sum()
        return vol_target((panel.fillna(0.0) * w).sum(axis=1), 0.16, max_leverage=1.6)

    base = build({"rsi": rsi, "don": don, "trd": trd, "xs": xs})
    corr = round(float(mat.reindex(base.index).fillna(0).corr(base)), 2)
    print(f"\n  correlation of multi-asset trend to the equity portfolio: {corr}")

    print("\n== portfolio impact (risk-parity, vol-target 16%/1.6x) ==")
    show("equity portfolio (4 sleeves)", base)
    show("+ multi-asset trend (5)", build({"rsi": rsi, "don": don, "trd": trd, "xs": xs, "mat": mat}))

    # idle-cash yield: park the equity portfolio's cash in T-bills (BIL)
    print("\n== idle-cash yield (park cash in BIL T-bill ETF) ==")
    try:
        bil = daily_bars("BIL")["close"].pct_change().reindex(base.index).fillna(0.0)
        # crude model: book is ~36% cash on avg -> that fraction earns the BIL daily return
        cash_frac = 0.36
        with_cash = base + cash_frac * bil
        show("equity portfolio + cash yield", with_cash)
        print("  (note: 2018-2020 T-bill yields were ~1-2%; today ~4-5%, so the forward boost is larger)")
    except Exception as e:
        print(f"  BIL data unavailable: {e}")


if __name__ == "__main__":
    main()
