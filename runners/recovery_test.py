"""
runners/recovery_test.py
------------------------
Does a recovery-thrust sleeve capture the 2019/2020 bull-run snapbacks the
defensive book missed — without overfitting? Robustness: must help 2018-2020
AND not wreck other folds. Standalone + portfolio impact.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from agents.daily_strategies import (
    backtest_book, backtest_cross_sectional, vol_target, _metrics_from_returns,
    sig_rsi2_meanrev, sig_donchian, sig_trend_5020, sig_recovery,
    DEPLOY_PARAMS, QUALITY_UNIVERSE, walk_forward_folds,
)
from data.sp500 import sp500_tickers


def show(label, r):
    m = _metrics_from_returns(r, [], label)
    fl = [f["return_pct"] for f in walk_forward_folds(r, 5)]
    print(f"  {label:26s} SR {m['sharpe']:5.2f} | CAGR {m['cagr']:6.1%} | DD {m['max_drawdown']:6.1%}"
          f" | folds " + " ".join(f"{x:+.0%}" for x in fl))
    return r


def main():
    U = QUALITY_UNIVERSE
    print("\n== recovery-thrust sleeve (folds: 16-18 18-20 20-22 22-24 24-26) ==")
    for hold in (90, 120, 150):
        show(f"recovery hold={hold}", backtest_book(sig_recovery, U, {"hold_days": hold}, label="rec")["_returns"])

    rec = backtest_book(sig_recovery, U, {"hold_days": 120}, label="rec")["_returns"]
    print("\n== PORTFOLIO: add recovery as a 5th sleeve (risk-parity, vol-target 16%/1.6x) ==")
    rsi = backtest_book(sig_rsi2_meanrev, U, DEPLOY_PARAMS["rsi2_meanrev"])["_returns"]
    don = backtest_book(sig_donchian, U)["_returns"]
    trd = backtest_book(sig_trend_5020, U)["_returns"]
    xs = backtest_cross_sectional(sp500_tickers(), mode="momentum", lookback=252, skip=21, k=10, market_filter=True)["_returns"]

    def build(sleeves):
        panel = pd.concat(sleeves, axis=1, sort=True)
        iv = {c: 1.0 / (panel[c].std() or 1e-9) for c in panel.columns}
        w = np.array([iv[c] for c in panel.columns]); w /= w.sum()
        return vol_target((panel.fillna(0.0) * w).sum(axis=1), 0.16, max_leverage=1.6)

    show("portfolio (4 sleeves)", build({"rsi": rsi, "don": don, "trd": trd, "xs": xs}))
    show("+ recovery (5 sleeves)", build({"rsi": rsi, "don": don, "trd": trd, "xs": xs, "rec": rec}))
    # recovery-heavy variant (overweight the bull-capture sleeve)
    def build_w(sleeves, wmap):
        panel = pd.concat(sleeves, axis=1, sort=True)
        wv = np.array([wmap[c] for c in panel.columns]); wv /= wv.sum()
        return vol_target((panel.fillna(0.0) * wv).sum(axis=1), 0.16, max_leverage=1.6)
    show("+ recovery (overweight)", build_w({"rsi": rsi, "don": don, "trd": trd, "xs": xs, "rec": rec},
         {"rsi": 0.25, "don": 0.2, "trd": 0.15, "xs": 0.1, "rec": 0.3}))


if __name__ == "__main__":
    main()
